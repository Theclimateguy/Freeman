"""Knowledge graph with a NetworkX + JSON backend."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
import json
from pathlib import Path
import re
from typing import TYPE_CHECKING, Any, Dict, List, Sequence

import networkx as nx
import yaml

from freeman.utils import deep_copy_jsonable, json_ready

if TYPE_CHECKING:
    from freeman.memory.vectorstore import KGVectorStore

ACTIVE_THRESHOLD = 0.60
UNCERTAIN_THRESHOLD = 0.30
REVIEW_THRESHOLD = 0.15


def _now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _status_for_confidence(confidence: float) -> str:
    if confidence >= ACTIVE_THRESHOLD:
        return "active"
    if confidence >= UNCERTAIN_THRESHOLD:
        return "uncertain"
    if confidence >= REVIEW_THRESHOLD:
        return "review"
    return "archived"


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _memory_config(config_path: str | Path | None = None) -> Dict[str, Any]:
    config_file = Path(config_path) if config_path is not None else _repo_root() / "config.yaml"
    if config_file.exists():
        payload = yaml.safe_load(config_file.read_text(encoding="utf-8")) or {}
        return payload.get("memory", {})
    return {}


def _default_json_path(config_path: str | Path | None = None) -> Path:
    memory_cfg = _memory_config(config_path)
    json_path = memory_cfg.get("json_path")
    if json_path:
        candidate = Path(json_path)
        base = Path(config_path).resolve().parent if config_path is not None else _repo_root()
        return candidate if candidate.is_absolute() else (base / candidate).resolve()
    return (_repo_root() / "runs" / "memory" / "knowledge_graph.json").resolve()


@dataclass
class KGNode:
    """Knowledge graph node."""

    id: str
    label: str
    node_type: str = "claim"
    content: str = ""
    confidence: float = 0.5
    status: str = ""
    evidence: List[str] = field(default_factory=list)
    sources: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    embedding: List[float] = field(default_factory=list)
    created_at: str = field(default_factory=_now_iso)
    updated_at: str = field(default_factory=_now_iso)
    archived_at: str | None = None

    def __post_init__(self) -> None:
        self.confidence = float(min(max(self.confidence, 0.0), 1.0))
        if not self.status:
            self.status = _status_for_confidence(self.confidence)
        self.evidence = list(self.evidence)
        self.sources = list(self.sources)
        self.metadata = deep_copy_jsonable(self.metadata)
        self.embedding = [float(value) for value in self.embedding]

    def snapshot(self) -> Dict[str, Any]:
        return json_ready(
            {
                "id": self.id,
                "label": self.label,
                "node_type": self.node_type,
                "content": self.content,
                "confidence": self.confidence,
                "status": self.status,
                "evidence": self.evidence,
                "sources": self.sources,
                "metadata": self.metadata,
                "embedding": self.embedding,
                "created_at": self.created_at,
                "updated_at": self.updated_at,
                "archived_at": self.archived_at,
            }
        )

    @classmethod
    def from_snapshot(cls, data: Dict[str, Any]) -> "KGNode":
        return cls(
            id=data["id"],
            label=data["label"],
            node_type=data.get("node_type", "claim"),
            content=data.get("content", ""),
            confidence=float(data.get("confidence", 0.5)),
            status=data.get("status", ""),
            evidence=list(data.get("evidence", [])),
            sources=list(data.get("sources", [])),
            metadata=deep_copy_jsonable(data.get("metadata", {})),
            embedding=list(data.get("embedding", [])),
            created_at=data.get("created_at", _now_iso()),
            updated_at=data.get("updated_at", _now_iso()),
            archived_at=data.get("archived_at"),
        )


@dataclass
class KGEdge:
    """Knowledge graph edge."""

    source: str
    target: str
    relation_type: str
    confidence: float = 0.5
    weight: float = 1.0
    id: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: str = field(default_factory=_now_iso)
    updated_at: str = field(default_factory=_now_iso)

    def __post_init__(self) -> None:
        self.confidence = float(min(max(self.confidence, 0.0), 1.0))
        self.weight = float(self.weight)
        self.metadata = deep_copy_jsonable(self.metadata)

    def snapshot(self) -> Dict[str, Any]:
        return json_ready(
            {
                "id": self.id,
                "source": self.source,
                "target": self.target,
                "relation_type": self.relation_type,
                "confidence": self.confidence,
                "weight": self.weight,
                "metadata": self.metadata,
                "created_at": self.created_at,
                "updated_at": self.updated_at,
            }
        )

    @classmethod
    def from_snapshot(cls, data: Dict[str, Any]) -> "KGEdge":
        return cls(
            id=data.get("id", ""),
            source=data["source"],
            target=data["target"],
            relation_type=data["relation_type"],
            confidence=float(data.get("confidence", 0.5)),
            weight=float(data.get("weight", 1.0)),
            metadata=deep_copy_jsonable(data.get("metadata", {})),
            created_at=data.get("created_at", _now_iso()),
            updated_at=data.get("updated_at", _now_iso()),
        )


class KnowledgeGraph:
    """NetworkX-backed knowledge graph with optional semantic retrieval."""

    def __init__(
        self,
        *,
        json_path: str | Path | None = None,
        config_path: str | Path | None = None,
        auto_load: bool = True,
        auto_save: bool = True,
        llm_adapter: Any | None = None,
        vectorstore: KGVectorStore | None = None,
    ) -> None:
        self.json_path = Path(json_path).resolve() if json_path is not None else _default_json_path(config_path)
        self.auto_save = auto_save
        self.llm_adapter = llm_adapter
        self.vectorstore = vectorstore
        self.graph = nx.MultiDiGraph()
        if auto_load and self.json_path.exists():
            self.load()

    def add_node(self, node: KGNode) -> None:
        """Insert a node and embed it if an adapter is available."""

        if node.id in self.graph:
            self.update_node(node)
            return
        prepared = self._prepare_node(node, previous=None)
        self.graph.add_node(prepared.id, **prepared.snapshot())
        self._sync_vectorstore(prepared)
        self._maybe_save()

    def update_node(self, node: KGNode) -> None:
        """Update an existing node and re-embed if content changed."""

        previous = self.get_node(node.id, lazy_embed=False)
        if previous is None:
            self.add_node(node)
            return
        prepared = self._prepare_node(node, previous=previous)
        node_store = self.graph.nodes[prepared.id]
        node_store.clear()
        node_store.update(prepared.snapshot())
        self._sync_vectorstore(prepared)
        self._maybe_save()

    def get_node(self, node_id: str, *, lazy_embed: bool = True) -> KGNode | None:
        """Return a node by id, lazily re-embedding legacy nodes if possible."""

        if node_id not in self.graph:
            return None
        return self._deserialize_node(dict(self.graph.nodes[node_id]), lazy_embed=lazy_embed)

    def nodes(self, *, lazy_embed: bool = False) -> List[KGNode]:
        """Return all nodes."""

        return [self._deserialize_node(dict(attrs), lazy_embed=lazy_embed) for _, attrs in self.graph.nodes(data=True)]

    def add_edge(self, edge: KGEdge) -> None:
        """Insert or replace an edge."""

        edge.updated_at = _now_iso()
        if not edge.id:
            edge.id = f"{edge.source}:{edge.relation_type}:{edge.target}:{self.graph.number_of_edges(edge.source, edge.target)}"
        self.graph.add_edge(edge.source, edge.target, key=edge.id, **edge.snapshot())
        self._maybe_save()

    def remove_edge(self, edge_id: str) -> None:
        """Remove one edge by id if present."""

        for source, target, key in list(self.graph.edges(keys=True)):
            if key == edge_id:
                self.graph.remove_edge(source, target, key=key)
                self._maybe_save()
                return

    def remove_node(self, node_id: str) -> None:
        """Remove one node and all incident edges."""

        if node_id not in self.graph:
            return
        if self.vectorstore is not None:
            self.vectorstore.delete(node_id)
        self.graph.remove_node(node_id)
        self._maybe_save()

    def edges(self) -> List[KGEdge]:
        """Return all edges."""

        return [KGEdge.from_snapshot(dict(attrs)) for _, _, _, attrs in self.graph.edges(keys=True, data=True)]

    def get_edge(self, edge_id: str) -> KGEdge | None:
        """Return one edge by id if present."""

        for _source, _target, key, attrs in self.graph.edges(keys=True, data=True):
            if str(attrs.get("id", key)) == str(edge_id):
                return KGEdge.from_snapshot(dict(attrs))
        return None

    def query(
        self,
        *,
        text: str | None = None,
        status: str | None = None,
        node_type: str | None = None,
        min_confidence: float | None = None,
        metadata_filters: Dict[str, Any] | None = None,
        metadata_contains: Dict[str, Sequence[Any] | Any] | None = None,
    ) -> List[KGNode]:
        """Return nodes matching the provided filters."""

        text_query = text.lower() if text else None
        results: List[KGNode] = []
        for node in self.nodes(lazy_embed=False):
            if status is not None and node.status != status:
                continue
            if node_type is not None and node.node_type != node_type:
                continue
            if min_confidence is not None and node.confidence < min_confidence:
                continue
            if metadata_filters is not None and not self._metadata_matches(node.metadata, metadata_filters):
                continue
            if metadata_contains is not None and not self._metadata_contains(node.metadata, metadata_contains):
                continue
            if text_query is not None:
                haystack = " ".join(
                    [
                        node.id,
                        node.label,
                        node.content,
                        " ".join(node.evidence),
                        " ".join(node.sources),
                        json.dumps(node.metadata, sort_keys=True),
                    ]
                ).lower()
                if text_query not in haystack:
                    continue
            results.append(node)
        return sorted(results, key=lambda item: (-item.confidence, item.id))

    def _metadata_matches(self, metadata: Dict[str, Any], filters: Dict[str, Any]) -> bool:
        """Return whether all metadata filter values match exactly."""

        for key, expected in filters.items():
            actual = self._metadata_lookup(metadata, key)
            if actual != expected:
                return False
        return True

    def _metadata_contains(
        self,
        metadata: Dict[str, Any],
        filters: Dict[str, Sequence[Any] | Any],
    ) -> bool:
        """Return whether metadata collections contain at least one requested value."""

        for key, expected in filters.items():
            actual = self._metadata_lookup(metadata, key)
            if isinstance(actual, str):
                actual_values = {actual}
            elif isinstance(actual, Sequence):
                actual_values = {value for value in actual if not isinstance(value, (dict, list))}
            else:
                actual_values = {actual}
            if isinstance(expected, str):
                expected_values = {expected}
            elif isinstance(expected, Sequence):
                expected_values = {value for value in expected if not isinstance(value, (dict, list))}
            else:
                expected_values = {expected}
            if actual_values.isdisjoint(expected_values):
                return False
        return True

    def _metadata_lookup(self, metadata: Dict[str, Any], key: str) -> Any:
        """Resolve dotted metadata paths against a node payload."""

        value: Any = metadata
        for part in str(key).split("."):
            if not isinstance(value, dict) or part not in value:
                return None
            value = value[part]
        return value

    def semantic_query(self, query_text: str, top_k: int = 15) -> List[KGNode]:
        """Return semantically relevant nodes plus 1-hop graph neighbors."""

        top_k = max(int(top_k), 0)
        if top_k <= 0:
            return []
        query_text = str(query_text).strip()
        if not query_text:
            return self.query(status="active")[:top_k]

        query_embedding: list[float] | None = None
        if self.llm_adapter is not None:
            query_embedding = [float(value) for value in self.llm_adapter.embed(query_text)]

        candidate_ids = self._semantic_candidate_ids(query_text, query_embedding=query_embedding, top_k=top_k)
        ordered_ids: List[str] = []
        seen = set()
        for node_id in candidate_ids:
            if node_id not in self.graph or node_id in seen:
                continue
            node = self.get_node(node_id)
            if node is None or node.status == "archived":
                continue
            ordered_ids.append(node_id)
            seen.add(node_id)
            for neighbor_id in [*self.graph.predecessors(node_id), *self.graph.successors(node_id)]:
                if neighbor_id in seen:
                    continue
                neighbor = self.get_node(neighbor_id)
                if neighbor is None or neighbor.status == "archived":
                    continue
                ordered_ids.append(neighbor_id)
                seen.add(neighbor_id)
        nodes: List[KGNode] = []
        for node_id in ordered_ids:
            node = self.get_node(node_id)
            if node is not None:
                nodes.append(node)
        return nodes

    def _semantic_candidate_ids(
        self,
        query_text: str,
        *,
        query_embedding: list[float] | None,
        top_k: int,
    ) -> List[str]:
        """Rank active nodes by embedding similarity or lexical-semantic fallback."""

        active_nodes = [node for node in self.nodes(lazy_embed=bool(self.llm_adapter is not None)) if node.status != "archived"]
        if not active_nodes:
            return []

        if self.vectorstore is not None and query_embedding:
            node_ids = self.vectorstore.query(query_embedding, top_k=max(top_k, min(top_k * 3, len(active_nodes))))
            ranked_ids = [node_id for node_id in node_ids if node_id in self.graph]
            if ranked_ids:
                return ranked_ids[:top_k]

        scored: list[tuple[float, str]] = []
        for node in active_nodes:
            score = self._semantic_score_node(query_text, node=node, query_embedding=query_embedding)
            if score <= 0.0:
                continue
            scored.append((score, node.id))
        if not scored:
            return [node.id for node in sorted(active_nodes, key=lambda item: (-item.confidence, item.id))[:top_k]]
        scored.sort(key=lambda item: (-item[0], item[1]))
        return [node_id for _, node_id in scored[:top_k]]

    def _semantic_score_node(
        self,
        query_text: str,
        *,
        node: KGNode,
        query_embedding: list[float] | None,
    ) -> float:
        """Score one node for semantic retrieval."""

        lexical = self._lexical_semantic_score(query_text, node)
        embedding_similarity = self._embedding_similarity(query_embedding, node)
        confidence_bonus = 0.02 * float(node.confidence)
        if embedding_similarity is not None:
            return float((0.75 * embedding_similarity) + (0.25 * lexical) + confidence_bonus)
        return float(lexical + confidence_bonus)

    def _lexical_semantic_score(self, query_text: str, node: KGNode) -> float:
        """Return a deterministic lexical-semantic relevance score."""

        haystack = self._semantic_haystack(node)
        if not haystack:
            return 0.0
        query_text = str(query_text).strip().lower()
        if not query_text:
            return 0.0
        score = 0.0
        if query_text in haystack:
            score += 1.0
        query_tokens = self._semantic_tokens(query_text)
        haystack_token_list = self._semantic_tokens(haystack)
        haystack_tokens = set(haystack_token_list)
        if query_tokens:
            overlap = sum(1 for token in query_tokens if token in haystack_tokens)
            score += 0.6 * (overlap / len(query_tokens))
        query_bigrams = self._semantic_ngrams(query_tokens, size=2)
        haystack_bigrams = self._semantic_ngrams(haystack_token_list, size=2)
        if query_bigrams:
            overlap = len(query_bigrams & haystack_bigrams)
            score += 0.3 * (overlap / len(query_bigrams))
        query_trigrams = self._semantic_ngrams(query_tokens, size=3)
        haystack_trigrams = self._semantic_ngrams(haystack_token_list, size=3)
        if query_trigrams:
            overlap = len(query_trigrams & haystack_trigrams)
            score += 0.2 * (overlap / len(query_trigrams))
        return float(score)

    def _semantic_haystack(self, node: KGNode) -> str:
        """Build one lowercase retrieval document for a node."""

        return " ".join(
            [
                str(node.id),
                str(node.label),
                str(node.content),
                " ".join(str(item) for item in node.evidence),
                " ".join(str(item) for item in node.sources),
                json.dumps(node.metadata, sort_keys=True),
            ]
        ).lower()

    def _semantic_tokens(self, text: str) -> List[str]:
        """Tokenize text for deterministic lexical retrieval."""

        normalized = str(text).lower().replace("_", " ")
        return re.findall(r"[a-z0-9]+", normalized)

    def _semantic_ngrams(self, tokens: Sequence[str], *, size: int) -> set[str]:
        """Return ordered token n-grams."""

        values = [str(token) for token in tokens if str(token)]
        if len(values) < size:
            return set()
        return {" ".join(values[index : index + size]) for index in range(len(values) - size + 1)}

    def _embedding_similarity(self, query_embedding: list[float] | None, node: KGNode) -> float | None:
        """Return cosine similarity between query and node embeddings."""

        if not query_embedding:
            return None
        candidate = node
        if not candidate.embedding and candidate.id in self.graph:
            stored = self.get_node(candidate.id, lazy_embed=True)
            if stored is not None:
                candidate = stored
        if not candidate.embedding or len(candidate.embedding) != len(query_embedding):
            return None
        return self._cosine_similarity(query_embedding, candidate.embedding)

    def _cosine_similarity(self, left: Sequence[float], right: Sequence[float]) -> float | None:
        """Return cosine similarity for two vectors."""

        if not left or not right or len(left) != len(right):
            return None
        left_norm = sum(float(value) * float(value) for value in left) ** 0.5
        right_norm = sum(float(value) * float(value) for value in right) ** 0.5
        if left_norm == 0.0 or right_norm == 0.0:
            return None
        dot = sum(float(a) * float(b) for a, b in zip(left, right, strict=False))
        return float(dot / (left_norm * right_norm))

    def explain_causal_path(self, edge_ids: list[str]) -> list[Any]:
        """Resolve ordered edge ids into human-readable causal steps."""

        from freeman.agent.analysispipeline import CausalStep

        steps: list[CausalStep] = []
        for edge_id in edge_ids:
            edge = self.get_edge(edge_id)
            if edge is None:
                continue
            source_node = self.get_node(edge.source, lazy_embed=False)
            target_node = self.get_node(edge.target, lazy_embed=False)
            metadata = dict(edge.metadata or {})
            steps.append(
                CausalStep(
                    edge_id=str(edge.id),
                    edge_type=str(edge.relation_type),
                    source_id=str(edge.source),
                    target_id=str(edge.target),
                    source_label=self._causal_label(source_node, fallback_id=edge.source),
                    target_label=self._causal_label(target_node, fallback_id=edge.target),
                    confidence=float(edge.confidence),
                    world_step=int(metadata.get("world_step", 0) or 0),
                )
            )
        return steps

    def _causal_label(self, node: KGNode | None, *, fallback_id: str) -> str:
        if node is None:
            return str(fallback_id)
        if node.node_type == "signal_event":
            snippet = str(node.content).strip()
            if snippet:
                compact = " ".join(snippet.split())[:80]
                return f"{node.id} ({compact})"
            return str(node.id)
        if node.content:
            compact = " ".join(str(node.content).split())
            return compact[:100]
        return str(node.label or node.id)

    def archive(self, node_id: str, reason: str = "") -> KGNode:
        """Archive a node in place and remove it from the vector store."""

        node = self.get_node(node_id, lazy_embed=False)
        if node is None:
            raise KeyError(node_id)
        node.status = "archived"
        node.archived_at = _now_iso()
        node.updated_at = node.archived_at
        if reason:
            node.metadata["archive_reason"] = reason
        self.graph.nodes[node_id].clear()
        self.graph.nodes[node_id].update(node.snapshot())
        if self.vectorstore is not None:
            self.vectorstore.delete(node_id)
        self._maybe_save()
        return node

    def archive_node(self, node_id: str, reason: str = "") -> KGNode:
        """Alias for compatibility with the semantic layer requirements."""

        return self.archive(node_id, reason=reason)

    def split_node(
        self,
        node_id: str,
        new_nodes: Sequence[KGNode | Dict[str, Any]],
        *,
        redistribute_edges: bool = True,
    ) -> List[str]:
        """Archive one node and replace it by several more specific nodes."""

        original = self.get_node(node_id, lazy_embed=False)
        if original is None:
            raise KeyError(node_id)

        incoming_edges = [
            KGEdge.from_snapshot(dict(attrs))
            for source, _, _, attrs in self.graph.in_edges(node_id, keys=True, data=True)
            if source != node_id
        ]
        outgoing_edges = [
            KGEdge.from_snapshot(dict(attrs))
            for _, target, _, attrs in self.graph.out_edges(node_id, keys=True, data=True)
            if target != node_id
        ]

        created_ids: List[str] = []
        for index, candidate in enumerate(new_nodes, start=1):
            if isinstance(candidate, KGNode):
                node = candidate
            else:
                payload = dict(candidate)
                node = KGNode(
                    id=payload.get("id", f"{node_id}__split_{index}"),
                    label=payload.get("label", original.label),
                    node_type=payload.get("node_type", original.node_type),
                    content=payload.get("content", original.content),
                    confidence=float(payload.get("confidence", original.confidence)),
                    status=payload.get("status", ""),
                    evidence=list(original.evidence) + list(payload.get("evidence", [])),
                    sources=list(original.sources) + list(payload.get("sources", [])),
                    metadata={**original.metadata, **deep_copy_jsonable(payload.get("metadata", {}))},
                    embedding=list(payload.get("embedding", [])),
                )
            self.add_node(node)
            created_ids.append(node.id)
            self.add_edge(KGEdge(source=node_id, target=node.id, relation_type="split_into", confidence=1.0, weight=1.0))
            if redistribute_edges:
                for edge in incoming_edges:
                    self.add_edge(
                        KGEdge(
                            source=edge.source,
                            target=node.id,
                            relation_type=edge.relation_type,
                            confidence=edge.confidence,
                            weight=edge.weight,
                            metadata=edge.metadata,
                        )
                    )
                for edge in outgoing_edges:
                    self.add_edge(
                        KGEdge(
                            source=node.id,
                            target=edge.target,
                            relation_type=edge.relation_type,
                            confidence=edge.confidence,
                            weight=edge.weight,
                            metadata=edge.metadata,
                        )
                    )

        self.archive(node_id, reason="split")
        return created_ids

    def to_payload(self) -> Dict[str, Any]:
        """Return a JSON-serializable graph payload."""

        return {
            "backend": "networkx-json",
            "json_path": str(self.json_path),
            "nodes": [node.snapshot() for node in self.nodes(lazy_embed=False)],
            "edges": [edge.snapshot() for edge in self.edges()],
        }

    def save(self, path: str | Path | None = None) -> Path:
        """Persist the graph as JSON."""

        target = Path(path).resolve() if path is not None else self.json_path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(self.to_payload(), indent=2, sort_keys=True), encoding="utf-8")
        return target

    def load(self, path: str | Path | None = None) -> None:
        """Load graph data from JSON."""

        source = Path(path).resolve() if path is not None else self.json_path
        payload = json.loads(source.read_text(encoding="utf-8"))
        self.graph = nx.MultiDiGraph()
        for node_payload in payload.get("nodes", []):
            node = KGNode.from_snapshot(node_payload)
            self.graph.add_node(node.id, **node.snapshot())
        for edge_payload in payload.get("edges", []):
            edge = KGEdge.from_snapshot(edge_payload)
            self.graph.add_edge(edge.source, edge.target, key=edge.id, **edge.snapshot())
        if self.vectorstore is not None:
            self.vectorstore.sync_from_kg(self)

    def export_json(self, path: str | Path | None = None) -> Path:
        """Export the graph as JSON."""

        return self.save(path)

    def export_dot(self, path: str | Path | None = None) -> str | Path:
        """Export the graph as Graphviz DOT."""

        lines = ["digraph KnowledgeGraph {"]
        for node in self.nodes(lazy_embed=False):
            label = f"{node.label}\\n[{node.status}, {node.confidence:.2f}]"
            lines.append(f'  "{node.id}" [label="{label}"];')
        for edge in self.edges():
            lines.append(f'  "{edge.source}" -> "{edge.target}" [label="{edge.relation_type} ({edge.confidence:.2f})"];')
        lines.append("}")
        dot = "\n".join(lines)
        if path is None:
            return dot
        target = Path(path).resolve()
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(dot, encoding="utf-8")
        return target

    def export_html(self, path: str | Path | None = None) -> str | Path:
        """Export the graph as a simple D3-backed HTML file."""

        payload = json.dumps(self.to_payload(), sort_keys=True)
        html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Knowledge Graph</title>
  <script src="https://d3js.org/d3.v7.min.js"></script>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 0; background: #f4f1ea; }}
    svg {{ width: 100vw; height: 100vh; }}
    .label {{ font-size: 12px; fill: #1f2933; }}
  </style>
</head>
<body>
  <svg></svg>
  <script>
    const graph = {payload};
    const width = window.innerWidth;
    const height = window.innerHeight;
    const svg = d3.select("svg").attr("viewBox", [0, 0, width, height]);
    const simulation = d3.forceSimulation(graph.nodes)
      .force("link", d3.forceLink(graph.edges).id(d => d.id).distance(140))
      .force("charge", d3.forceManyBody().strength(-360))
      .force("center", d3.forceCenter(width / 2, height / 2));
    const link = svg.append("g").selectAll("line")
      .data(graph.edges).join("line")
      .attr("stroke", "#8b6f47")
      .attr("stroke-opacity", 0.7);
    const node = svg.append("g").selectAll("circle")
      .data(graph.nodes).join("circle")
      .attr("r", d => 10 + 12 * d.confidence)
      .attr("fill", d => d.status === "active" ? "#1d7a6b" : d.status === "review" ? "#cc7a00" : "#6b7280");
    const label = svg.append("g").selectAll("text")
      .data(graph.nodes).join("text")
      .attr("class", "label")
      .text(d => d.label);
    simulation.on("tick", () => {{
      link
        .attr("x1", d => d.source.x)
        .attr("y1", d => d.source.y)
        .attr("x2", d => d.target.x)
        .attr("y2", d => d.target.y);
      node
        .attr("cx", d => d.x)
        .attr("cy", d => d.y);
      label
        .attr("x", d => d.x + 14)
        .attr("y", d => d.y + 4);
    }});
  </script>
</body>
</html>
"""
        if path is None:
            return html
        target = Path(path).resolve()
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(html, encoding="utf-8")
        return target

    def cosine_similarity(self, left: KGNode, right: KGNode) -> float | None:
        """Return cosine similarity between two node embeddings when available."""

        left_node = KGNode.from_snapshot(left.snapshot())
        right_node = KGNode.from_snapshot(right.snapshot())
        if not left_node.embedding and left_node.id in self.graph:
            stored = self.get_node(left_node.id, lazy_embed=True)
            if stored is not None:
                left_node = stored
        if not right_node.embedding and self.llm_adapter is not None and right_node.content:
            right_node.embedding = [float(value) for value in self.llm_adapter.embed(right_node.content)]
        if not left_node.embedding or not right_node.embedding:
            return None
        return self._cosine_similarity(left_node.embedding, right_node.embedding)

    def _prepare_node(self, node: KGNode, previous: KGNode | None) -> KGNode:
        prepared = KGNode.from_snapshot(node.snapshot())
        force_reembed = bool(prepared.metadata.pop("_force_reembed", False))
        content_changed = previous is not None and previous.content != prepared.content
        if previous is not None and not force_reembed and not content_changed and not prepared.embedding and previous.embedding:
            prepared.embedding = list(previous.embedding)
        if prepared.content and self.llm_adapter is not None and (force_reembed or not prepared.embedding or content_changed):
            prepared.embedding = [float(value) for value in self.llm_adapter.embed(prepared.content)]
        prepared.updated_at = _now_iso()
        if previous is not None:
            prepared.created_at = previous.created_at
        return prepared

    def _deserialize_node(self, attrs: Dict[str, Any], *, lazy_embed: bool) -> KGNode:
        node = KGNode.from_snapshot(attrs)
        if lazy_embed and self.llm_adapter is not None and node.content and not node.embedding:
            node.embedding = [float(value) for value in self.llm_adapter.embed(node.content)]
            node.updated_at = _now_iso()
            self.graph.nodes[node.id].clear()
            self.graph.nodes[node.id].update(node.snapshot())
            self._sync_vectorstore(node)
            self._maybe_save()
        return node

    def _sync_vectorstore(self, node: KGNode) -> None:
        if self.vectorstore is None:
            return
        if node.status == "archived":
            self.vectorstore.delete(node.id)
            return
        self.vectorstore.upsert(node)

    def _maybe_save(self) -> None:
        if self.auto_save:
            self.save()


__all__ = ["KGEdge", "KGNode", "KnowledgeGraph"]
