from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from freeman.interface.factory import build_knowledge_graph
from freeman.memory.knowledgegraph import KGEdge, KGNode, KnowledgeGraph


def _graph(tmp_path: Path, backend_name: str) -> KnowledgeGraph:
    return KnowledgeGraph(
        json_path=tmp_path / f"{backend_name}.json",
        sqlite_path=tmp_path / f"{backend_name}.db",
        backend_name=backend_name,
        auto_load=False,
        auto_save=True,
    )


def _projection(graph: KnowledgeGraph) -> dict[str, object]:
    return {
        "nodes": [(node.id, node.node_type, node.status, round(node.confidence, 3)) for node in graph.nodes(lazy_embed=False)],
        "edges": [(edge.source, edge.target, edge.relation_type, round(edge.confidence, 3)) for edge in graph.edges()],
        "active_claims": [node.id for node in graph.query(node_type="claim", status="active")],
    }


@pytest.mark.parametrize("backend_name", ["json", "sqlite"])
def test_knowledge_graph_backend_roundtrip_parity(tmp_path, backend_name) -> None:
    graph = _graph(tmp_path, backend_name)
    graph.add_node(KGNode(id="climate:enso", label="ENSO", node_type="claim", confidence=0.9))
    graph.add_node(KGNode(id="risk:drought", label="Drought risk", node_type="risk", confidence=0.7))
    graph.add_edge(KGEdge(source="climate:enso", target="risk:drought", relation_type="causes", confidence=0.8))
    expected = _projection(graph)

    reloaded = KnowledgeGraph(
        json_path=tmp_path / f"{backend_name}.json",
        sqlite_path=tmp_path / f"{backend_name}.db",
        backend_name=backend_name,
        auto_load=True,
        auto_save=False,
    )

    assert _projection(reloaded) == expected


@pytest.mark.parametrize("backend_name", ["json", "sqlite"])
def test_knowledge_graph_backend_transaction_rolls_back(tmp_path, backend_name) -> None:
    graph = _graph(tmp_path, backend_name)
    graph.add_node(KGNode(id="baseline", label="Baseline", confidence=0.9))

    with pytest.raises(RuntimeError):
        with graph.transaction():
            graph.add_node(KGNode(id="partial", label="Partial", confidence=0.9))
            graph.save()
            raise RuntimeError("rollback")

    reloaded = KnowledgeGraph(
        json_path=tmp_path / f"{backend_name}.json",
        sqlite_path=tmp_path / f"{backend_name}.db",
        backend_name=backend_name,
        auto_load=True,
        auto_save=False,
    )
    assert reloaded.get_node("baseline", lazy_embed=False) is not None
    assert reloaded.get_node("partial", lazy_embed=False) is None


def test_sqlite_backend_creates_indexed_tables(tmp_path) -> None:
    graph = _graph(tmp_path, "sqlite")
    graph.add_node(KGNode(id="node:1", label="Node", node_type="claim", confidence=0.9))
    graph.add_node(KGNode(id="node:2", label="Node 2", node_type="risk", confidence=0.8))
    graph.add_edge(KGEdge(source="node:1", target="node:2", relation_type="causes"))

    with sqlite3.connect(tmp_path / "sqlite.db") as conn:
        tables = {row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        indexes = {row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type='index'")}
        node_count = conn.execute("SELECT COUNT(*) FROM nodes").fetchone()[0]
        edge_count = conn.execute("SELECT COUNT(*) FROM edges").fetchone()[0]

    assert {"nodes", "edges"}.issubset(tables)
    assert {"idx_nodes_type", "idx_nodes_status", "idx_edges_source", "idx_edges_target"}.issubset(indexes)
    assert node_count == 2
    assert edge_count == 1


def test_build_knowledge_graph_selects_sqlite_backend_from_config(tmp_path) -> None:
    config_path = tmp_path / "config.yaml"
    config = {
        "memory": {
            "backend": "sqlite",
            "json_path": "./kg.json",
            "sqlite_path": "./kg.db",
        }
    }
    graph = build_knowledge_graph(config, config_path=config_path, auto_load=False, auto_save=True)

    graph.add_node(KGNode(id="configured", label="Configured", confidence=0.9))

    assert graph.backend_name == "sqlite"
    assert graph.sqlite_path == tmp_path / "kg.db"
    assert graph.json_path == tmp_path / "kg.json"
    assert graph.sqlite_path.exists()
    assert not graph.json_path.exists()
