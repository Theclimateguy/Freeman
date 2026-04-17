"""Session-log to knowledge-graph reconciliation."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import math
from typing import Dict, List

from freeman_librarian.memory.knowledgegraph import KGEdge, KGNode, KnowledgeGraph
from freeman_librarian.memory.sessionlog import KGDelta, SessionLog


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
    compacted_node_ids: List[str] = field(default_factory=list)
    kg_health: Dict[str, float | int] = field(default_factory=dict)


class Reconciler:
    """Merge session-log deltas into a persistent knowledge graph."""

    def __init__(
        self,
        *,
        thresholds: ConfidenceThresholds | None = None,
        mode: str = "log_odds",
        prior_strength_penalty: float = 0.0,
        prior_strength: float = 1.0,
        w_support: float = 1.0,
        w_conflict: float = 1.0,
        gamma: float = 0.0,
        clip_eps: float = 1.0e-9,
        merge_threshold: float = 0.82,
        compaction_interval: int = 20,
    ) -> None:
        self.thresholds = thresholds or ConfidenceThresholds()
        self.mode = str(mode).strip().lower()
        self.prior_strength_penalty = float(prior_strength_penalty)
        self.prior_strength = max(float(prior_strength), 1.0e-9)
        self.w_support = float(w_support)
        self.w_conflict = float(w_conflict)
        self.gamma = max(float(gamma), 0.0)
        self.clip_eps = min(max(float(clip_eps), 1.0e-12), 1.0e-3)
        self.merge_threshold = float(merge_threshold)
        self.compaction_interval = max(int(compaction_interval), 0)
        if self.mode not in {"legacy", "log_odds"}:
            raise ValueError(f"Unsupported reconciler mode: {mode}")

    @classmethod
    def from_config(cls, config_path: str | None = None) -> "Reconciler":
        """Instantiate a reconciler from config.yaml when available."""

        from pathlib import Path

        import yaml

        config_file = (
            Path(config_path).resolve()
            if config_path is not None
            else Path(__file__).resolve().parents[2] / "config.yaml"
        )
        if not config_file.exists():
            return cls()
        payload = yaml.safe_load(config_file.read_text(encoding="utf-8")) or {}
        reconciler_cfg = ((payload.get("memory") or {}).get("reconciler") or {})
        return cls(
            merge_threshold=float(reconciler_cfg.get("merge_threshold", 0.82)),
            compaction_interval=int(reconciler_cfg.get("compaction_interval", 20)),
        )

    def beta_update(
        self,
        confidence: float,
        support: int,
        contradiction: int,
        *,
        prior_strength_penalty: float | None = None,
    ) -> float:
        """Update confidence using the configured reconciliation mode."""

        if self.mode == "legacy":
            return self._legacy_update(
                confidence,
                support,
                contradiction,
                prior_strength_penalty=prior_strength_penalty,
            )
        return self._log_odds_update(confidence, support, contradiction)

    def _legacy_update(
        self,
        confidence: float,
        support: int,
        contradiction: int,
        *,
        prior_strength_penalty: float | None = None,
    ) -> float:
        """Apply the legacy multiplicative confidence update and clip to [0, 1]."""

        penalty = self.prior_strength_penalty if prior_strength_penalty is None else float(prior_strength_penalty)
        denominator = support + contradiction
        support_ratio = float(support / denominator) if denominator > 0 else 1.0
        updated = float(confidence) * support_ratio - penalty
        return min(max(updated, 0.0), 1.0)

    def _log_odds_update(self, confidence: float, support: int, contradiction: int) -> float:
        """Apply a Bayesian-style log-odds update with exponential forgetting."""

        clipped_confidence = min(max(float(confidence), self.clip_eps), 1.0 - self.clip_eps)
        support_count = max(int(support), 0)
        contradiction_count = max(int(contradiction), 0)
        if support_count == 0 and contradiction_count == 0 and self.gamma == 0.0:
            return clipped_confidence

        current_log_odds = math.log(clipped_confidence / (1.0 - clipped_confidence))
        forgetting_decay = math.exp(-self.gamma)
        evidence_unit = math.log((self.prior_strength + 1.0) / self.prior_strength)
        updated_log_odds = (
            forgetting_decay * current_log_odds
            + self.w_support * support_count * evidence_unit
            - self.w_conflict * contradiction_count * evidence_unit
        )
        return 1.0 / (1.0 + math.exp(-updated_log_odds))

    def classify_status(self, confidence: float) -> str:
        """Map confidence to the configured status bands."""

        if confidence >= self.thresholds.active:
            return "active"
        if confidence >= self.thresholds.uncertain:
            return "uncertain"
        if confidence >= self.thresholds.review:
            return "review"
        return "archived"

    def reconcile(
        self,
        knowledge_graph: KnowledgeGraph,
        session_log: SessionLog,
        *,
        step_index: int | None = None,
        last_compaction_step: int | None = None,
    ) -> ReconciliationResult:
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

        compaction_last_step = int(last_compaction_step) if last_compaction_step is not None else -1
        if self.compaction_interval > 0 and step_index is not None and int(step_index) > 0 and int(step_index) % self.compaction_interval == 0:
            result.compacted_node_ids = self.kg_compact(knowledge_graph)
            compaction_last_step = int(step_index)

        result.kg_health = self.kg_health(knowledge_graph, compaction_last_step)
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
                if knowledge_graph.vectorstore is not None and knowledge_graph.llm_adapter is not None:
                    merged.embedding = []
                    merged.metadata["_force_reembed"] = True
                knowledge_graph.add_node(merged)
                result.merged_node_ids.append(merged.id)
                return
            similarity = self._claim_similarity(knowledge_graph, candidate, incoming)
            if similarity is not None and similarity >= self.merge_threshold:
                merged = self._merge_nodes(candidate, incoming)
                merged.metadata["semantic_merge_similarity"] = similarity
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

    def _claim_similarity(self, knowledge_graph: KnowledgeGraph, existing: KGNode, incoming: KGNode) -> float | None:
        return knowledge_graph.cosine_similarity(existing, incoming)

    def _merge_nodes(self, existing: KGNode, incoming: KGNode) -> KGNode:
        evidence = list(dict.fromkeys([*existing.evidence, *incoming.evidence]))
        sources = list(dict.fromkeys([*existing.sources, *incoming.sources]))
        metadata = {**existing.metadata, **incoming.metadata}
        claim_variants = [text for text in [existing.content.strip(), incoming.content.strip()] if text]
        metadata["claim_variants"] = list(dict.fromkeys([*existing.metadata.get("claim_variants", []), *claim_variants]))
        merged_content = "\n".join(metadata["claim_variants"])
        return KGNode(
            id=existing.id,
            label=incoming.label or existing.label,
            node_type=incoming.node_type or existing.node_type,
            content=merged_content or incoming.content or existing.content,
            confidence=max(existing.confidence, incoming.confidence),
            evidence=evidence,
            sources=sources,
            metadata=metadata,
            created_at=existing.created_at,
        )

    def kg_compact(self, knowledge_graph: KnowledgeGraph) -> List[str]:
        """Merge redundant ``__split_*`` nodes back into their archived parent."""

        compacted: List[str] = []
        split_nodes = [node for node in knowledge_graph.nodes(lazy_embed=False) if "__split_" in node.id]
        for split_node in split_nodes:
            parent_ids = self._split_parents(knowledge_graph, split_node.id)
            if len(parent_ids) != 1:
                continue
            parent_id = parent_ids[0]
            parent = knowledge_graph.get_node(parent_id, lazy_embed=False)
            if parent is None:
                continue
            sibling_ids = self._split_children(knowledge_graph, parent_id)
            if split_node.id not in sibling_ids:
                continue
            if not self._is_redundant_split(knowledge_graph, parent_id, split_node.id, sibling_ids):
                continue
            merged_parent = self._merge_nodes(parent, split_node)
            merged_parent.status = self.classify_status(merged_parent.confidence)
            merged_parent.archived_at = None
            merged_parent.metadata.pop("archive_reason", None)
            compacted_from = list(merged_parent.metadata.get("compacted_split_ids", []))
            compacted_from.append(split_node.id)
            merged_parent.metadata["compacted_split_ids"] = list(dict.fromkeys(compacted_from))
            knowledge_graph.add_node(merged_parent)
            knowledge_graph.remove_node(split_node.id)
            compacted.append(split_node.id)
        return compacted

    def kg_health(self, knowledge_graph: KnowledgeGraph, compaction_last_step: int) -> Dict[str, float | int]:
        """Return a compact health summary for checkpoint/runtime inspection."""

        live_nodes = [node_id for node_id, attrs in knowledge_graph.graph.nodes(data=True) if attrs.get("status") != "archived"]
        if live_nodes:
            avg_node_degree = sum(float(knowledge_graph.graph.degree(node_id)) for node_id in live_nodes) / len(live_nodes)
        else:
            avg_node_degree = 0.0
        split_node_count = sum(1 for node_id in knowledge_graph.graph.nodes if "__split_" in str(node_id))
        return {
            "split_node_count": int(split_node_count),
            "avg_node_degree": float(avg_node_degree),
            "compaction_last_step": int(compaction_last_step),
        }

    def _split_parents(self, knowledge_graph: KnowledgeGraph, node_id: str) -> List[str]:
        return [
            str(source)
            for source, _target, _key, attrs in knowledge_graph.graph.in_edges(node_id, keys=True, data=True)
            if attrs.get("relation_type") == "split_into"
        ]

    def _split_children(self, knowledge_graph: KnowledgeGraph, parent_id: str) -> List[str]:
        return [
            str(target)
            for _source, target, _key, attrs in knowledge_graph.graph.out_edges(parent_id, keys=True, data=True)
            if attrs.get("relation_type") == "split_into"
        ]

    def _is_redundant_split(
        self,
        knowledge_graph: KnowledgeGraph,
        parent_id: str,
        split_node_id: str,
        sibling_ids: List[str],
    ) -> bool:
        outgoing = self._edge_signature_set(knowledge_graph, split_node_id)
        comparison_targets = [parent_id, *[node_id for node_id in sibling_ids if node_id != split_node_id]]
        baseline: set[tuple[str, str]] = set()
        for candidate_id in comparison_targets:
            baseline.update(self._edge_signature_set(knowledge_graph, candidate_id))
        return outgoing.issubset(baseline)

    def _edge_signature_set(self, knowledge_graph: KnowledgeGraph, node_id: str) -> set[tuple[str, str]]:
        return {
            (str(attrs.get("relation_type")), str(target))
            for _source, target, _key, attrs in knowledge_graph.graph.out_edges(node_id, keys=True, data=True)
            if attrs.get("relation_type") != "split_into"
        }

    def verify_causal_path(
        self,
        knowledge_graph: KnowledgeGraph,
        forecast: "Forecast",
        current_signal_id: str | None,
    ) -> dict:
        """Check whether a forecast causal path is still confirmed by the KG."""

        if not forecast.causal_path:
            return {"confirmed": [], "refuted": [], "unknown": []}
        edge_records = [
            {
                "source": str(source),
                "target": str(target),
                "edge_id": str(attrs.get("id", key)),
                "relation_type": str(attrs.get("relation_type", "")),
                "metadata": dict(attrs.get("metadata", {}) or {}),
            }
            for source, target, key, attrs in knowledge_graph.graph.edges(keys=True, data=True)
        ]
        edge_lookup = {record["edge_id"]: record for record in edge_records}
        current_signal_ref = f"signal:{current_signal_id}" if current_signal_id else None
        confirmed: List[str] = []
        refuted: List[str] = []
        unknown: List[str] = []
        refuted_at_node: str | None = None

        for edge_id in forecast.causal_path:
            record = edge_lookup.get(str(edge_id))
            if record is None:
                unknown.append(str(edge_id))
                continue
            if self._causal_edge_refuted(record, edge_records, current_signal_ref):
                refuted.append(str(edge_id))
                if refuted_at_node is None:
                    refuted_at_node = str(record["target"])
                continue
            confirmed.append(str(edge_id))

        payload = {
            "confirmed": confirmed,
            "refuted": refuted,
            "unknown": unknown,
        }
        if refuted_at_node is not None:
            payload["refuted_at_node"] = refuted_at_node
        if current_signal_ref is not None and refuted:
            payload["refutation_signal"] = current_signal_ref
        return payload

    def update_self_model(
        self,
        knowledge_graph: KnowledgeGraph,
        forecast: "Forecast",
        *,
        current_signal_id: str | None = None,
    ) -> KGNode:
        """Accumulate verified forecast errors into a self-observation KG node."""

        if forecast.status != "verified":
            raise ValueError("Self-model updates require a verified forecast.")
        if forecast.actual_prob is None:
            raise ValueError("Verified forecast must include actual_prob.")

        node_id = f"self:forecast_error:{forecast.domain_id}:{forecast.outcome_id}"
        existing = knowledge_graph.get_node(node_id)
        errors = json.loads(existing.metadata.get("errors_json", "[]")) if existing is not None else []
        signed_error = float(forecast.predicted_prob - forecast.actual_prob)
        errors.append(signed_error)
        errors = errors[-50:]
        mean_abs_error = sum(abs(error) for error in errors) / len(errors)
        bias = sum(errors) / len(errors)
        causal_verification = self.verify_causal_path(
            knowledge_graph,
            forecast,
            current_signal_id=current_signal_id,
        )
        node = KGNode(
            id=node_id,
            label=f"Self-model: {forecast.domain_id}/{forecast.outcome_id}",
            node_type="self_observation",
            content=f"n={len(errors)} forecasts; MAE={mean_abs_error:.4f}; bias={bias:+.4f}",
            confidence=0.9,
            metadata={
                "domain_id": forecast.domain_id,
                "outcome_id": forecast.outcome_id,
                "mean_abs_error": mean_abs_error,
                "bias": bias,
                "n_forecasts": len(errors),
                "errors_json": json.dumps(errors),
                "causal_path_confirmed": len(causal_verification["confirmed"]),
                "causal_path_refuted": len(causal_verification["refuted"]),
                "causal_path_unknown": len(causal_verification["unknown"]),
                "causal_path_confirmed_ids": list(causal_verification["confirmed"]),
                "causal_path_refuted_ids": list(causal_verification["refuted"]),
                "causal_path_unknown_ids": list(causal_verification["unknown"]),
            },
        )
        if "refuted_at_node" in causal_verification:
            node.metadata["refuted_at_node"] = causal_verification["refuted_at_node"]
        if "refutation_signal" in causal_verification:
            node.metadata["refutation_signal"] = causal_verification["refutation_signal"]
        knowledge_graph.add_node(node)
        knowledge_graph.save()
        return node

    def _causal_edge_refuted(
        self,
        edge_record: Dict[str, object],
        edge_records: List[Dict[str, object]],
        current_signal_ref: str | None,
    ) -> bool:
        if current_signal_ref is None:
            return False
        relation_type = str(edge_record["relation_type"])
        metadata = dict(edge_record["metadata"])
        for candidate in edge_records:
            if str(candidate["relation_type"]) != relation_type:
                continue
            if str(candidate["source"]) != current_signal_ref:
                continue
            candidate_meta = dict(candidate["metadata"])
            if relation_type == "causes":
                if candidate_meta.get("param_name") != metadata.get("param_name"):
                    continue
                if float(candidate_meta.get("delta_sign", 0.0)) * float(metadata.get("delta_sign", 0.0)) < 0.0:
                    return True
            elif relation_type == "propagates_to":
                if candidate_meta.get("param_name") != metadata.get("param_name"):
                    continue
                if candidate_meta.get("resource_id") != metadata.get("resource_id"):
                    continue
                if float(candidate_meta.get("variable_sign", 0.0)) * float(metadata.get("variable_sign", 0.0)) < 0.0:
                    return True
            elif relation_type == "threshold_exceeded":
                if candidate_meta.get("outcome_id") != metadata.get("outcome_id"):
                    continue
                if candidate_meta.get("resource_id") != metadata.get("resource_id"):
                    continue
                if float(candidate_meta.get("contribution_sign", 0.0)) * float(metadata.get("contribution_sign", 0.0)) < 0.0:
                    return True
        return False


__all__ = ["ConfidenceThresholds", "ReconciliationResult", "Reconciler"]
