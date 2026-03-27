"""Session-log to knowledge-graph reconciliation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

from freeman.memory.knowledgegraph import KGEdge, KGNode, KnowledgeGraph
from freeman.memory.sessionlog import KGDelta, SessionLog


@dataclass
class ConfidenceThresholds:
    """Status thresholds for KG nodes."""

    active: float = 0.60
    uncertain: float = 0.30
    review: float = 0.15


@dataclass
class ReconciliationResult:
    """Summary of a reconciliation run."""

    processed_deltas: int = 0
    merged_node_ids: List[str] = field(default_factory=list)
    split_nodes: Dict[str, List[str]] = field(default_factory=dict)
    archived_node_ids: List[str] = field(default_factory=list)
    conflict_node_ids: List[str] = field(default_factory=list)


class Reconciler:
    """Merge session-log deltas into a persistent knowledge graph."""

    def __init__(
        self,
        *,
        thresholds: ConfidenceThresholds | None = None,
        prior_strength_penalty: float = 0.0,
    ) -> None:
        self.thresholds = thresholds or ConfidenceThresholds()
        self.prior_strength_penalty = float(prior_strength_penalty)

    def beta_update(
        self,
        confidence: float,
        support: int,
        contradiction: int,
        *,
        prior_strength_penalty: float | None = None,
    ) -> float:
        """Apply the spec confidence update and clip to [0, 1]."""

        penalty = self.prior_strength_penalty if prior_strength_penalty is None else float(prior_strength_penalty)
        denominator = support + contradiction
        support_ratio = float(support / denominator) if denominator > 0 else 1.0
        updated = float(confidence) * support_ratio - penalty
        return min(max(updated, 0.0), 1.0)

    def classify_status(self, confidence: float) -> str:
        """Map confidence to the configured status bands."""

        if confidence >= self.thresholds.active:
            return "active"
        if confidence >= self.thresholds.uncertain:
            return "uncertain"
        if confidence >= self.thresholds.review:
            return "review"
        return "archived"

    def reconcile(self, knowledge_graph: KnowledgeGraph, session_log: SessionLog) -> ReconciliationResult:
        """Merge all KG deltas from a session into the graph."""

        result = ReconciliationResult()
        for delta in session_log.kg_deltas:
            self._apply_delta(knowledge_graph, delta, result)
            result.processed_deltas += 1

        for node in knowledge_graph.nodes():
            if node.status == "archived":
                if node.id not in result.archived_node_ids:
                    result.archived_node_ids.append(node.id)
                continue
            node.status = self.classify_status(node.confidence)
            knowledge_graph.add_node(node)
            if node.status == "archived" and node.id not in result.archived_node_ids:
                knowledge_graph.archive(node.id, reason="below_threshold")
                result.archived_node_ids.append(node.id)

        knowledge_graph.save()
        return result

    def _apply_delta(self, knowledge_graph: KnowledgeGraph, delta: KGDelta, result: ReconciliationResult) -> None:
        if delta.operation in {"add_node", "update_node"}:
            incoming = self._node_from_delta(delta)
            incoming.confidence = self.beta_update(incoming.confidence, delta.support, delta.contradiction)
            incoming.status = self.classify_status(incoming.confidence)
            candidate = self._find_candidate_node(knowledge_graph, incoming)
            if candidate is None:
                knowledge_graph.add_node(incoming)
                result.merged_node_ids.append(incoming.id)
                return
            if self._same_claim(candidate, incoming):
                merged = self._merge_nodes(candidate, incoming)
                merged.confidence = self.beta_update(merged.confidence, delta.support, delta.contradiction)
                merged.status = self.classify_status(merged.confidence)
                knowledge_graph.add_node(merged)
                result.merged_node_ids.append(merged.id)
                return
            split_ids = knowledge_graph.split_node(
                candidate.id,
                [
                    {
                        "id": f"{candidate.id}__split_1",
                        "label": candidate.label,
                        "content": candidate.content,
                        "confidence": candidate.confidence,
                        "metadata": {**candidate.metadata, "split_from": candidate.id},
                    },
                    incoming,
                ],
            )
            result.split_nodes[candidate.id] = split_ids
            result.conflict_node_ids.append(candidate.id)
            if candidate.id not in result.archived_node_ids:
                result.archived_node_ids.append(candidate.id)
            return

        if delta.operation == "add_edge":
            edge_payload = delta.payload.get("edge", delta.payload)
            knowledge_graph.add_edge(KGEdge.from_snapshot(edge_payload))
            return

        if delta.operation == "archive_node":
            target_id = delta.target_id or delta.payload.get("id")
            if target_id is None:
                raise ValueError("archive_node delta requires target_id")
            knowledge_graph.archive(target_id, reason=delta.metadata.get("reason", "session_archive"))
            if target_id not in result.archived_node_ids:
                result.archived_node_ids.append(target_id)
            return

        raise ValueError(f"Unsupported KG delta operation: {delta.operation}")

    def _node_from_delta(self, delta: KGDelta) -> KGNode:
        payload = delta.payload.get("node", delta.payload)
        if isinstance(payload, KGNode):
            return payload
        return KGNode.from_snapshot(payload)

    def _find_candidate_node(self, knowledge_graph: KnowledgeGraph, incoming: KGNode) -> KGNode | None:
        claim_key = self._claim_key(incoming)
        for node in knowledge_graph.nodes():
            if node.status == "archived":
                continue
            if node.node_type != incoming.node_type:
                continue
            if self._claim_key(node) == claim_key:
                return node
        return None

    def _claim_key(self, node: KGNode) -> str:
        explicit = node.metadata.get("claim_key")
        if explicit:
            return str(explicit).strip().lower()
        return f"{node.node_type}:{node.label.strip().lower()}"

    def _same_claim(self, existing: KGNode, incoming: KGNode) -> bool:
        return existing.content.strip().lower() == incoming.content.strip().lower()

    def _merge_nodes(self, existing: KGNode, incoming: KGNode) -> KGNode:
        evidence = list(dict.fromkeys([*existing.evidence, *incoming.evidence]))
        sources = list(dict.fromkeys([*existing.sources, *incoming.sources]))
        metadata = {**existing.metadata, **incoming.metadata}
        return KGNode(
            id=existing.id,
            label=incoming.label or existing.label,
            node_type=incoming.node_type or existing.node_type,
            content=incoming.content or existing.content,
            confidence=max(existing.confidence, incoming.confidence),
            evidence=evidence,
            sources=sources,
            metadata=metadata,
            created_at=existing.created_at,
        )


__all__ = ["ConfidenceThresholds", "ReconciliationResult", "Reconciler"]
