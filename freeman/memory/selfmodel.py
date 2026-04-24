"""Structured self-model graph stored on top of the knowledge graph."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
from typing import Any

from freeman.memory.knowledgegraph import KGEdge, KGNode, KnowledgeGraph
from freeman.utils import deep_copy_jsonable

_ENGINE_CALLER_TOKEN = "freeman_consciousness_engine"
_SELF_MODEL_NAMESPACE = "sm:"
_EDGE_METADATA_KEY = "self_model_edge"

VALID_NODE_TYPES = {
    "self_observation",
    "self_capability",
    "self_uncertainty",
    "active_hypothesis",
    "goal_state",
    "attention_focus",
    "identity_trait",
}

VALID_EDGE_TYPES = {
    "supports",
    "contradicts",
    "depends_on",
    "derived_from",
    "focuses_on",
    "serves_goal",
    "revises",
}


def _now() -> datetime:
    return datetime.now(timezone.utc).replace(microsecond=0)


def _to_datetime(value: str | datetime) -> datetime:
    if isinstance(value, datetime):
        dt = value
    else:
        dt = datetime.fromisoformat(str(value))
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc).replace(microsecond=0)


def _datetime_to_iso(value: datetime) -> str:
    return _to_datetime(value).isoformat()


class SelfModelAccessError(Exception):
    """Raised when a non-engine caller attempts to mutate the self-model."""


@dataclass(eq=True)
class SelfModelNode:
    """One node in the structured self-model."""

    node_id: str
    node_type: str
    domain: str | None
    payload: dict[str, Any]
    confidence: float
    created_at: datetime
    updated_at: datetime
    source_trace_id: str | None

    def __post_init__(self) -> None:
        if not str(self.node_id).startswith(_SELF_MODEL_NAMESPACE):
            raise ValueError(f"SelfModelNode id must start with '{_SELF_MODEL_NAMESPACE}': {self.node_id}")
        if self.node_type not in VALID_NODE_TYPES:
            raise ValueError(f"Unsupported self-model node type: {self.node_type}")
        self.payload = deep_copy_jsonable(self.payload)
        self.confidence = float(min(max(self.confidence, 0.0), 1.0))
        self.created_at = _to_datetime(self.created_at)
        self.updated_at = _to_datetime(self.updated_at)
        self.source_trace_id = str(self.source_trace_id) if self.source_trace_id is not None else None
        self.domain = str(self.domain) if self.domain is not None else None

    def to_dict(self) -> dict[str, Any]:
        return {
            "node_id": self.node_id,
            "node_type": self.node_type,
            "domain": self.domain,
            "payload": deep_copy_jsonable(self.payload),
            "confidence": self.confidence,
            "created_at": _datetime_to_iso(self.created_at),
            "updated_at": _datetime_to_iso(self.updated_at),
            "source_trace_id": self.source_trace_id,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SelfModelNode":
        return cls(
            node_id=data["node_id"],
            node_type=data["node_type"],
            domain=data.get("domain"),
            payload=deep_copy_jsonable(data.get("payload", {})),
            confidence=float(data.get("confidence", 0.5)),
            created_at=_to_datetime(data.get("created_at", _now())),
            updated_at=_to_datetime(data.get("updated_at", _now())),
            source_trace_id=data.get("source_trace_id"),
        )

    def to_kg_node(self) -> KGNode:
        return KGNode(
            id=self.node_id,
            label=self.node_type.replace("_", " ").title(),
            node_type=self.node_type,
            content=json.dumps(self.payload, sort_keys=True),
            confidence=self.confidence,
            metadata={
                "self_model": True,
                "domain": self.domain,
                "payload": deep_copy_jsonable(self.payload),
                "source_trace_id": self.source_trace_id,
                "self_model_node": self.to_dict(),
            },
            created_at=_datetime_to_iso(self.created_at),
            updated_at=_datetime_to_iso(self.updated_at),
        )

    @classmethod
    def from_kg_node(cls, node: KGNode) -> "SelfModelNode":
        payload = deep_copy_jsonable(node.metadata.get("self_model_node", {}))
        if payload:
            return cls.from_dict(payload)
        return cls(
            node_id=node.id,
            node_type=node.node_type,
            domain=node.metadata.get("domain"),
            payload=deep_copy_jsonable(node.metadata.get("payload", {})),
            confidence=float(node.confidence),
            created_at=_to_datetime(node.created_at),
            updated_at=_to_datetime(node.updated_at),
            source_trace_id=node.metadata.get("source_trace_id"),
        )


@dataclass(eq=True)
class SelfModelEdge:
    """One edge in the structured self-model."""

    edge_id: str
    edge_type: str
    source_id: str
    target_id: str
    weight: float
    created_at: datetime
    trace_id: str | None

    def __post_init__(self) -> None:
        if self.edge_type not in VALID_EDGE_TYPES:
            raise ValueError(f"Unsupported self-model edge type: {self.edge_type}")
        if not str(self.source_id).startswith(_SELF_MODEL_NAMESPACE):
            raise ValueError(f"SelfModelEdge source must start with '{_SELF_MODEL_NAMESPACE}': {self.source_id}")
        if not str(self.target_id).startswith(_SELF_MODEL_NAMESPACE):
            raise ValueError(f"SelfModelEdge target must start with '{_SELF_MODEL_NAMESPACE}': {self.target_id}")
        self.weight = float(self.weight)
        self.created_at = _to_datetime(self.created_at)
        self.trace_id = str(self.trace_id) if self.trace_id is not None else None

    def to_dict(self) -> dict[str, Any]:
        return {
            "edge_id": self.edge_id,
            "edge_type": self.edge_type,
            "source_id": self.source_id,
            "target_id": self.target_id,
            "weight": self.weight,
            "created_at": _datetime_to_iso(self.created_at),
            "trace_id": self.trace_id,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SelfModelEdge":
        return cls(
            edge_id=data["edge_id"],
            edge_type=data["edge_type"],
            source_id=data["source_id"],
            target_id=data["target_id"],
            weight=float(data.get("weight", 1.0)),
            created_at=_to_datetime(data.get("created_at", _now())),
            trace_id=data.get("trace_id"),
        )

    def to_kg_edge(self) -> KGEdge:
        return KGEdge(
            id=self.edge_id,
            source=self.source_id,
            target=self.target_id,
            relation_type=self.edge_type,
            confidence=1.0,
            weight=self.weight,
            metadata={
                _EDGE_METADATA_KEY: self.to_dict(),
                "self_model": True,
                "trace_id": self.trace_id,
            },
            created_at=_datetime_to_iso(self.created_at),
            updated_at=_datetime_to_iso(self.created_at),
        )

    @classmethod
    def from_kg_edge(cls, edge: KGEdge) -> "SelfModelEdge":
        payload = deep_copy_jsonable(edge.metadata.get(_EDGE_METADATA_KEY, {}))
        if payload:
            return cls.from_dict(payload)
        return cls(
            edge_id=edge.id,
            edge_type=edge.relation_type,
            source_id=edge.source,
            target_id=edge.target,
            weight=float(edge.weight),
            created_at=_to_datetime(edge.created_at),
            trace_id=edge.metadata.get("trace_id"),
        )


class SelfModelGraph:
    """Restricted writer wrapper over a knowledge graph self-model namespace."""

    def __init__(self, knowledge_graph: KnowledgeGraph, caller_token: str | None = None) -> None:
        self.knowledge_graph = knowledge_graph
        self._caller_token = caller_token

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, SelfModelGraph):
            return NotImplemented
        return self.snapshot() == other.snapshot()

    def write(self, node_or_edge: SelfModelNode | SelfModelEdge, caller_token: str) -> None:
        """Persist a self-model node or edge if the caller is authorized."""

        if caller_token != _ENGINE_CALLER_TOKEN:
            raise SelfModelAccessError("Only ConsciousnessEngine may mutate SelfModelGraph.")

        if isinstance(node_or_edge, SelfModelNode):
            self.knowledge_graph.add_node(node_or_edge.to_kg_node())
            return
        if isinstance(node_or_edge, SelfModelEdge):
            self.knowledge_graph.add_edge(node_or_edge.to_kg_edge())
            return
        raise TypeError(f"Unsupported self-model payload: {type(node_or_edge)!r}")

    def get_nodes_by_type(self, node_type: str) -> list[SelfModelNode]:
        """Return all self-model nodes of the requested type."""

        if node_type not in VALID_NODE_TYPES:
            raise ValueError(f"Unsupported self-model node type: {node_type}")
        return sorted(
            [
                SelfModelNode.from_kg_node(node)
                for node in self.knowledge_graph.nodes(lazy_embed=False)
                if node.id.startswith(_SELF_MODEL_NAMESPACE) and node.node_type == node_type
            ],
            key=lambda item: item.node_id,
        )

    def get_edges_by_type(self, edge_type: str) -> list[SelfModelEdge]:
        """Return all self-model edges of the requested type."""

        if edge_type not in VALID_EDGE_TYPES:
            raise ValueError(f"Unsupported self-model edge type: {edge_type}")
        return sorted(
            [
                SelfModelEdge.from_kg_edge(edge)
                for edge in self.knowledge_graph.edges()
                if edge.source.startswith(_SELF_MODEL_NAMESPACE)
                and edge.target.startswith(_SELF_MODEL_NAMESPACE)
                and edge.relation_type == edge_type
            ],
            key=lambda item: item.edge_id,
        )

    def snapshot(self) -> dict[str, Any]:
        """Return a full self-model snapshot."""

        nodes = [
            SelfModelNode.from_kg_node(node).to_dict()
            for node in self.knowledge_graph.nodes(lazy_embed=False)
            if node.id.startswith(_SELF_MODEL_NAMESPACE)
        ]
        edges = [
            SelfModelEdge.from_kg_edge(edge).to_dict()
            for edge in self.knowledge_graph.edges()
            if edge.source.startswith(_SELF_MODEL_NAMESPACE) and edge.target.startswith(_SELF_MODEL_NAMESPACE)
        ]
        return {
            "namespace": _SELF_MODEL_NAMESPACE,
            "nodes": sorted(nodes, key=lambda item: item["node_id"]),
            "edges": sorted(edges, key=lambda item: item["edge_id"]),
        }

    def load_snapshot(self, data: dict[str, Any]) -> None:
        """Replace the current self-model namespace with the provided snapshot."""

        for node_id in list(self.knowledge_graph.graph.nodes):
            if str(node_id).startswith(_SELF_MODEL_NAMESPACE):
                self.knowledge_graph.graph.remove_node(node_id)

        for node_payload in data.get("nodes", []):
            node = SelfModelNode.from_dict(node_payload).to_kg_node()
            self.knowledge_graph.graph.add_node(node.id, **node.snapshot())
        for edge_payload in data.get("edges", []):
            edge = SelfModelEdge.from_dict(edge_payload).to_kg_edge()
            self.knowledge_graph.graph.add_edge(edge.source, edge.target, key=edge.id, **edge.snapshot())

        if self.knowledge_graph.auto_save:
            self.knowledge_graph.save()


__all__ = [
    "SelfModelAccessError",
    "SelfModelEdge",
    "SelfModelGraph",
    "SelfModelNode",
    "VALID_EDGE_TYPES",
    "VALID_NODE_TYPES",
]
