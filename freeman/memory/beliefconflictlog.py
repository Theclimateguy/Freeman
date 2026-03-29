"""Queryable conflict trace memory built on top of the knowledge graph."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List

from freeman.memory.knowledgegraph import KGNode, KnowledgeGraph


@dataclass(frozen=True)
class BeliefConflictRecord:
    """One logged contradiction between momentum and an incoming signal."""

    node_id: str
    domain_id: str
    belief_before: float
    belief_after: float
    signal_direction: str
    signal_strength: float
    signal_source: str
    conflict_reason: str
    resolution: str
    timestamp: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_node(cls, node: KGNode) -> "BeliefConflictRecord":
        metadata = dict(node.metadata)
        signal = metadata.get("signal", {})
        return cls(
            node_id=node.id,
            domain_id=str(metadata.get("domain_id", "")),
            belief_before=float(metadata.get("belief_before", 0.0)),
            belief_after=float(metadata.get("belief_after", 0.0)),
            signal_direction=str(signal.get("direction", "flat")),
            signal_strength=float(signal.get("strength", 0.0)),
            signal_source=str(signal.get("source", metadata.get("signal_source", ""))),
            conflict_reason=str(metadata.get("conflict_reason", "")),
            resolution=str(metadata.get("resolution", "")),
            timestamp=str(metadata.get("timestamp", node.created_at)),
            metadata=metadata,
        )

    def prompt_payload(self) -> Dict[str, Any]:
        """Return a compact JSON-ready conflict payload for prompting."""

        return {
            "domain_id": self.domain_id,
            "belief_before": self.belief_before,
            "belief_after": self.belief_after,
            "signal": {
                "source": self.signal_source,
                "direction": self.signal_direction,
                "strength": self.signal_strength,
            },
            "conflict_reason": self.conflict_reason,
            "resolution": self.resolution,
            "timestamp": self.timestamp,
        }


class BeliefConflictLog:
    """Thin query layer over belief conflict nodes in the KG."""

    def __init__(self, knowledge_graph: KnowledgeGraph) -> None:
        self.knowledge_graph = knowledge_graph

    def record(self, node: KGNode) -> KGNode:
        """Persist a conflict node into the KG."""

        self.knowledge_graph.add_node(node)
        return node

    def query(
        self,
        *,
        domain_id: str | None = None,
        limit: int | None = None,
    ) -> List[BeliefConflictRecord]:
        """Return logged conflicts, optionally restricted to one domain."""

        metadata_filters = {"domain_id": str(domain_id)} if domain_id is not None else None
        nodes = self.knowledge_graph.query(
            node_type="belief_conflict",
            metadata_filters=metadata_filters,
        )
        records = [BeliefConflictRecord.from_node(node) for node in nodes]
        records.sort(key=lambda item: item.timestamp, reverse=True)
        if limit is not None:
            return records[:limit]
        return records


__all__ = ["BeliefConflictLog", "BeliefConflictRecord"]
