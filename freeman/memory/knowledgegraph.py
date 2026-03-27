"""Knowledge graph with a NetworkX + JSON backend."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

import networkx as nx
import yaml

from freeman.utils import deep_copy_jsonable, json_ready

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


def _default_json_path(config_path: str | Path | None = None) -> Path:
    config_file = Path(config_path) if config_path is not None else _repo_root() / "config.yaml"
    if config_file.exists():
        payload = yaml.safe_load(config_file.read_text(encoding="utf-8")) or {}
        memory_cfg = payload.get("memory", {})
        json_path = memory_cfg.get("json_path")
        if json_path:
            candidate = Path(json_path)
            return candidate if candidate.is_absolute() else (config_file.parent / candidate).resolve()
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
    """NetworkX-backed knowledge graph with JSON persistence."""

    def __init__(
        self,
        *,
        json_path: str | Path | None = None,
        config_path: str | Path | None = None,
        auto_load: bool = True,
        auto_save: bool = True,
    ) -> None:
        self.json_path = Path(json_path).resolve() if json_path is not None else _default_json_path(config_path)
        self.auto_save = auto_save
        self.graph = nx.MultiDiGraph()
        if auto_load and self.json_path.exists():
            self.load()

    def add_node(self, node: KGNode) -> None:
        """Insert or replace a node."""

        node.updated_at = _now_iso()
        self.graph.add_node(node.id, **node.snapshot())
        self._maybe_save()

    def get_node(self, node_id: str) -> KGNode | None:
        """Return a node by id."""

        if node_id not in self.graph:
            return None
        return KGNode.from_snapshot(dict(self.graph.nodes[node_id]))

    def nodes(self) -> List[KGNode]:
        """Return all nodes."""

        return [KGNode.from_snapshot(dict(attrs)) for _, attrs in self.graph.nodes(data=True)]

    def add_edge(self, edge: KGEdge) -> None:
        """Insert or replace an edge."""

        edge.updated_at = _now_iso()
        if not edge.id:
            edge.id = f"{edge.source}:{edge.relation_type}:{edge.target}:{self.graph.number_of_edges(edge.source, edge.target)}"
        self.graph.add_edge(edge.source, edge.target, key=edge.id, **edge.snapshot())
        self._maybe_save()

    def edges(self) -> List[KGEdge]:
        """Return all edges."""

        return [
            KGEdge.from_snapshot(dict(attrs))
            for _, _, _, attrs in self.graph.edges(keys=True, data=True)
        ]

    def query(
        self,
        *,
        text: str | None = None,
        status: str | None = None,
        node_type: str | None = None,
        min_confidence: float | None = None,
    ) -> List[KGNode]:
        """Return nodes matching the provided filters."""

        text_query = text.lower() if text else None
        results: List[KGNode] = []
        for node in self.nodes():
            if status is not None and node.status != status:
                continue
            if node_type is not None and node.node_type != node_type:
                continue
            if min_confidence is not None and node.confidence < min_confidence:
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

    def archive(self, node_id: str, reason: str = "") -> KGNode:
        """Archive a node in place."""

        node = self.get_node(node_id)
        if node is None:
            raise KeyError(node_id)
        node.status = "archived"
        node.archived_at = _now_iso()
        node.updated_at = node.archived_at
        if reason:
            node.metadata["archive_reason"] = reason
        self.graph.nodes[node_id].update(node.snapshot())
        self._maybe_save()
        return node

    def split_node(
        self,
        node_id: str,
        new_nodes: Sequence[KGNode | Dict[str, Any]],
        *,
        redistribute_edges: bool = True,
    ) -> List[str]:
        """Archive one node and replace it by several more specific nodes."""

        original = self.get_node(node_id)
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
                )
            self.add_node(node)
            created_ids.append(node.id)
            self.add_edge(
                KGEdge(
                    source=node_id,
                    target=node.id,
                    relation_type="split_into",
                    confidence=1.0,
                    weight=1.0,
                )
            )
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
            "nodes": [node.snapshot() for node in self.nodes()],
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

    def export_json(self, path: str | Path | None = None) -> Path:
        """Export the graph as JSON."""

        return self.save(path)

    def export_dot(self, path: str | Path | None = None) -> str | Path:
        """Export the graph as Graphviz DOT."""

        lines = ["digraph KnowledgeGraph {"]
        for node in self.nodes():
            label = f"{node.label}\\n[{node.status}, {node.confidence:.2f}]"
            lines.append(f'  "{node.id}" [label="{label}"];')
        for edge in self.edges():
            lines.append(
                f'  "{edge.source}" -> "{edge.target}" [label="{edge.relation_type} ({edge.confidence:.2f})"];'
            )
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

    def _maybe_save(self) -> None:
        if self.auto_save:
            self.save()


__all__ = ["KGEdge", "KGNode", "KnowledgeGraph"]
