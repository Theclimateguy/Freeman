"""Tests for semantic memory and vector-store retrieval."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pytest
import yaml

from freeman.agent.analysispipeline import AnalysisPipeline, AnalysisPipelineConfig
from freeman.interface.cli import main as cli_main
from freeman.memory.knowledgegraph import KGEdge, KGNode, KnowledgeGraph
from freeman.memory.vectorstore import KGVectorStore


@dataclass
class MappingEmbeddingAdapter:
    mapping: dict[str, list[float]]

    def __post_init__(self) -> None:
        self.calls: list[str] = []

    def embed(self, text: str) -> list[float]:
        self.calls.append(text)
        return list(self.mapping[text])


@pytest.fixture
def chroma_client():
    chromadb = pytest.importorskip("chromadb")
    return chromadb.EphemeralClient()


def test_vectorstore_upsert_query_and_delete(tmp_path, chroma_client) -> None:
    store = KGVectorStore(path=tmp_path / "chroma", collection_name="kg_test", client=chroma_client)
    node_a = KGNode(id="a", label="A", content="A", confidence=0.9, embedding=[1.0, 0.0, 0.0])
    node_b = KGNode(id="b", label="B", content="B", confidence=0.4, embedding=[0.9, 0.1, 0.0])

    store.upsert(node_a)
    store.upsert(node_b)

    assert store.query([1.0, 0.0, 0.0], top_k=2) == ["a", "b"]
    assert store.query([1.0, 0.0, 0.0], top_k=2, min_confidence=0.5) == ["a"]

    store.delete("a")

    assert store.query([1.0, 0.0, 0.0], top_k=2) == ["b"]


def test_vectorstore_sync_from_kg_skips_nodes_without_embeddings(tmp_path, chroma_client) -> None:
    store = KGVectorStore(path=tmp_path / "chroma", collection_name="kg_sync", client=chroma_client)
    kg = KnowledgeGraph(json_path=tmp_path / "kg.json", auto_load=False, auto_save=False)
    kg.add_node(KGNode(id="embedded", label="Embedded", content="Embedded", confidence=0.8, embedding=[1.0, 0.0, 0.0]))
    kg.add_node(KGNode(id="empty", label="Empty", content="Empty", confidence=0.8))

    upserted = store.sync_from_kg(kg)

    assert upserted == 1
    assert store.query([1.0, 0.0, 0.0], top_k=2) == ["embedded"]


def test_semantic_query_returns_seed_nodes_plus_neighbors(tmp_path, chroma_client) -> None:
    adapter = MappingEmbeddingAdapter(
        {
            "Water stress is rising.": [1.0, 0.0, 0.0],
            "Reservoir storage is falling.": [0.1, 1.0, 0.0],
            "Commodity demand is rising.": [0.0, 0.0, 1.0],
            "regional water alert": [1.0, 0.0, 0.0],
        }
    )
    store = KGVectorStore(path=tmp_path / "chroma", collection_name="kg_semantic", client=chroma_client)
    kg = KnowledgeGraph(
        json_path=tmp_path / "kg.json",
        auto_load=False,
        auto_save=False,
        llm_adapter=adapter,
        vectorstore=store,
    )
    kg.add_node(KGNode(id="water", label="Water", content="Water stress is rising.", confidence=0.8))
    kg.add_node(KGNode(id="reservoir", label="Reservoir", content="Reservoir storage is falling.", confidence=0.7))
    kg.add_node(KGNode(id="commodity", label="Commodity", content="Commodity demand is rising.", confidence=0.7))
    kg.add_edge(KGEdge(source="water", target="reservoir", relation_type="drives"))

    results = kg.semantic_query("regional water alert", top_k=1)

    assert [node.id for node in results] == ["water", "reservoir"]


def test_pipeline_context_falls_back_to_active_nodes_without_vectorstore(tmp_path) -> None:
    kg = KnowledgeGraph(json_path=tmp_path / "kg.json", auto_load=False, auto_save=False)
    kg.add_node(KGNode(id="a", label="A", content="A", confidence=0.8))
    kg.add_node(KGNode(id="b", label="B", content="B", confidence=0.7))
    kg.add_node(KGNode(id="c", label="C", content="C", confidence=0.6))
    kg.add_node(KGNode(id="archived", label="Archived", content="Archived", confidence=0.1, status="archived"))
    pipeline = AnalysisPipeline(
        knowledge_graph=kg,
        config=AnalysisPipelineConfig(retrieval_top_k=1, max_context_nodes=2),
    )

    context = pipeline._get_context_nodes("unused")

    assert [node.id for node in context] == ["a", "b"]


def test_semantic_query_without_vectorstore_uses_embeddings(tmp_path) -> None:
    adapter = MappingEmbeddingAdapter(
        {
            "Heat adaptation lowers migration losses.": [1.0, 0.0, 0.0],
            "Power demand rises during cold snaps.": [0.0, 1.0, 0.0],
            "adaptation migration": [1.0, 0.0, 0.0],
        }
    )
    kg = KnowledgeGraph(
        json_path=tmp_path / "kg.json",
        auto_load=False,
        auto_save=False,
        llm_adapter=adapter,
    )
    kg.add_node(KGNode(id="adapt", label="Adapt", content="Heat adaptation lowers migration losses.", confidence=0.8))
    kg.add_node(KGNode(id="power", label="Power", content="Power demand rises during cold snaps.", confidence=0.9))

    results = kg.semantic_query("adaptation migration", top_k=1)

    assert [node.id for node in results] == ["adapt"]


def test_legacy_nodes_are_embedded_lazily_on_first_access(tmp_path, chroma_client) -> None:
    graph_path = tmp_path / "kg.json"
    payload = {
        "backend": "networkx-json",
        "json_path": str(graph_path),
        "nodes": [
            {
                "id": "legacy",
                "label": "Legacy",
                "node_type": "claim",
                "content": "Legacy claim",
                "confidence": 0.8,
                "status": "active",
                "evidence": [],
                "sources": [],
                "metadata": {},
                "created_at": "2026-03-27T00:00:00+00:00",
                "updated_at": "2026-03-27T00:00:00+00:00",
                "archived_at": None,
            }
        ],
        "edges": [],
    }
    graph_path.write_text(json.dumps(payload), encoding="utf-8")
    adapter = MappingEmbeddingAdapter({"Legacy claim": [0.6, 0.6, 0.0]})
    store = KGVectorStore(path=tmp_path / "chroma", collection_name="kg_legacy", client=chroma_client)
    kg = KnowledgeGraph(
        json_path=graph_path,
        auto_load=True,
        auto_save=False,
        llm_adapter=adapter,
        vectorstore=store,
    )

    node = kg.get_node("legacy")

    assert node is not None
    assert node.embedding == [0.6, 0.6, 0.0]
    assert store.query([0.6, 0.6, 0.0], top_k=1) == ["legacy"]


def test_update_node_reembeds_when_content_changes(tmp_path, chroma_client) -> None:
    adapter = MappingEmbeddingAdapter(
        {
            "Old content": [1.0, 0.0, 0.0],
            "New content": [0.0, 1.0, 0.0],
        }
    )
    store = KGVectorStore(path=tmp_path / "chroma", collection_name="kg_update", client=chroma_client)
    kg = KnowledgeGraph(
        json_path=tmp_path / "kg.json",
        auto_load=False,
        auto_save=False,
        llm_adapter=adapter,
        vectorstore=store,
    )
    kg.add_node(KGNode(id="claim", label="Claim", content="Old content", confidence=0.8))

    kg.update_node(KGNode(id="claim", label="Claim", content="New content", confidence=0.8))

    updated = kg.get_node("claim", lazy_embed=False)
    assert updated is not None
    assert updated.embedding == [0.0, 1.0, 0.0]
    assert store.query([0.0, 1.0, 0.0], top_k=1) == ["claim"]


def test_analysis_pipeline_uses_semantic_retrieval_with_cap(tmp_path, chroma_client) -> None:
    adapter = MappingEmbeddingAdapter(
        {
            "Signal A": [1.0, 0.0, 0.0],
            "Neighbor B": [0.0, 1.0, 0.0],
            "Neighbor C": [0.0, 0.0, 1.0],
            "signal": [1.0, 0.0, 0.0],
        }
    )
    store = KGVectorStore(path=tmp_path / "chroma", collection_name="kg_pipeline", client=chroma_client)
    kg = KnowledgeGraph(
        json_path=tmp_path / "kg.json",
        auto_load=False,
        auto_save=False,
        llm_adapter=adapter,
        vectorstore=store,
    )
    kg.add_node(KGNode(id="a", label="A", content="Signal A", confidence=0.9))
    kg.add_node(KGNode(id="b", label="B", content="Neighbor B", confidence=0.9))
    kg.add_node(KGNode(id="c", label="C", content="Neighbor C", confidence=0.9))
    kg.add_edge(KGEdge(source="a", target="b", relation_type="supports"))
    kg.add_edge(KGEdge(source="a", target="c", relation_type="supports"))
    pipeline = AnalysisPipeline(
        knowledge_graph=kg,
        config=AnalysisPipelineConfig(retrieval_top_k=1, max_context_nodes=2),
    )

    context = pipeline._get_context_nodes("signal")

    assert [node.id for node in context] == ["a", "b"]


def test_cli_kg_reindex_populates_embeddings_and_creates_chroma_dir(tmp_path, monkeypatch, capsys) -> None:
    pytest.importorskip("chromadb")
    graph_path = tmp_path / "kg.json"
    chroma_path = tmp_path / "chroma_db"
    config_path = tmp_path / "config.yaml"
    payload = {
        "backend": "networkx-json",
        "json_path": str(graph_path),
        "nodes": [
            {
                "id": "legacy",
                "label": "Legacy",
                "node_type": "claim",
                "content": "CLI legacy claim",
                "confidence": 0.8,
                "status": "active",
                "evidence": [],
                "sources": [],
                "metadata": {},
                "created_at": "2026-03-27T00:00:00+00:00",
                "updated_at": "2026-03-27T00:00:00+00:00",
                "archived_at": None,
            }
        ],
        "edges": [],
    }
    graph_path.write_text(json.dumps(payload), encoding="utf-8")
    config_path.write_text(
        yaml.safe_dump(
            {
                "memory": {
                    "json_path": str(graph_path),
                    "vector_store": {
                        "enabled": True,
                        "backend": "chroma",
                        "path": str(chroma_path),
                        "collection": "kg_nodes",
                    },
                    "embedding_model": "text-embedding-3-small",
                    "retrieval_top_k": 15,
                    "max_context_nodes": 30,
                    "session_log_path": str(tmp_path / "sessions"),
                }
            }
        ),
        encoding="utf-8",
    )

    adapter = MappingEmbeddingAdapter({"CLI legacy claim": [0.4, 0.5, 0.6]})
    monkeypatch.setattr("freeman.interface.cli._build_embedding_adapter", lambda config, use_stub=False: (adapter, "mock"))

    exit_code = cli_main(["--config-path", str(config_path), "kg-reindex", "--batch-size", "1"])
    output = json.loads(capsys.readouterr().out)
    reloaded = KnowledgeGraph(json_path=graph_path, auto_load=True, auto_save=False)
    node = reloaded.get_node("legacy", lazy_embed=False)

    assert exit_code == 0
    assert output["reembedded"] == 1
    assert output["synced"] == 1
    assert chroma_path.exists()
    assert node is not None
    assert node.embedding == [0.4, 0.5, 0.6]


def test_cli_query_uses_semantic_retrieval_without_vectorstore(tmp_path, monkeypatch, capsys) -> None:
    graph_path = tmp_path / "kg.json"
    config_path = tmp_path / "config.yaml"
    payload = {
        "backend": "networkx-json",
        "json_path": str(graph_path),
        "nodes": [
            {
                "id": "adapt",
                "label": "Adapt",
                "node_type": "claim",
                "content": "Heat adaptation lowers migration losses.",
                "confidence": 0.8,
                "status": "active",
                "evidence": [],
                "sources": [],
                "metadata": {},
                "embedding": [],
                "created_at": "2026-03-27T00:00:00+00:00",
                "updated_at": "2026-03-27T00:00:00+00:00",
                "archived_at": None,
            },
            {
                "id": "power",
                "label": "Power",
                "node_type": "claim",
                "content": "Power demand rises during cold snaps.",
                "confidence": 0.9,
                "status": "active",
                "evidence": [],
                "sources": [],
                "metadata": {},
                "embedding": [],
                "created_at": "2026-03-27T00:00:00+00:00",
                "updated_at": "2026-03-27T00:00:00+00:00",
                "archived_at": None,
            },
        ],
        "edges": [],
    }
    graph_path.write_text(json.dumps(payload), encoding="utf-8")
    config_path.write_text(
        yaml.safe_dump(
            {
                "memory": {
                    "json_path": str(graph_path),
                    "embedding_provider": "hashing",
                    "retrieval_top_k": 5,
                    "max_context_nodes": 10,
                    "session_log_path": str(tmp_path / "sessions"),
                }
            }
        ),
        encoding="utf-8",
    )
    adapter = MappingEmbeddingAdapter(
        {
            "Heat adaptation lowers migration losses.": [1.0, 0.0, 0.0],
            "Power demand rises during cold snaps.": [0.0, 1.0, 0.0],
            "adaptation migration": [1.0, 0.0, 0.0],
        }
    )
    monkeypatch.setattr("freeman.interface.cli._build_embedding_adapter", lambda config, use_stub=False: (adapter, "mock"))

    exit_code = cli_main(["--config-path", str(config_path), "query", "--text", "adaptation migration", "--limit", "1"])
    output = json.loads(capsys.readouterr().out)

    assert exit_code == 0
    assert output["semantic"] is True
    assert output["count"] == 1
    assert output["matches"][0]["id"] == "adapt"
