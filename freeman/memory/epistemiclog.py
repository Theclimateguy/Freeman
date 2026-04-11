"""Persistent epistemic memory built on top of the knowledge graph."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, List, Sequence

from freeman.core.world import WorldState
from freeman.memory.knowledgegraph import KGNode, KnowledgeGraph

if TYPE_CHECKING:
    from freeman.agent.forecastregistry import Forecast


def normalize_causal_chain(value: Any) -> List[str]:
    """Normalize a causal-chain payload into a stable list of string labels."""

    if value is None:
        return []
    if isinstance(value, str):
        if "->" in value:
            parts = [part.strip() for part in value.split("->")]
            return [part for part in parts if part]
        token = value.strip()
        return [token] if token else []
    if isinstance(value, Sequence):
        return [str(item).strip() for item in value if str(item).strip()]
    return [str(value).strip()]


def infer_domain_family(domain_id: str, metadata: Dict[str, Any] | None = None) -> str:
    """Infer a stable domain family for epistemic memory retrieval."""

    if isinstance(metadata, dict):
        explicit = metadata.get("domain_family")
        if explicit:
            return str(explicit).strip()
    return str(domain_id).strip()


def infer_world_tags(world: WorldState) -> Dict[str, Any]:
    """Return stable retrieval tags for a world."""

    return {
        "domain_family": infer_domain_family(world.domain_id, world.metadata),
        "causal_chain": normalize_causal_chain(world.metadata.get("causal_chain")),
    }


@dataclass(frozen=True)
class EpistemicRecord:
    """One verified forecast remembered for later calibration."""

    node_id: str
    forecast_id: str
    domain_id: str
    outcome_id: str
    domain_family: str
    causal_chain: tuple[str, ...]
    predicted: float
    actual: float
    delta: float
    abs_error: float
    rationale_at_cutoff: str
    timestamp_cutoff: str
    timestamp_resolution: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_node(cls, node: KGNode) -> "EpistemicRecord":
        metadata = dict(node.metadata)
        causal_chain = tuple(normalize_causal_chain(metadata.get("causal_chain")))
        return cls(
            node_id=node.id,
            forecast_id=str(metadata.get("forecast_id", "")),
            domain_id=str(metadata.get("domain_id", "")),
            outcome_id=str(metadata.get("outcome_id", "")),
            domain_family=str(metadata.get("domain_family", "")),
            causal_chain=causal_chain,
            predicted=float(metadata.get("predicted", metadata.get("predicted_prob", 0.0))),
            actual=float(metadata.get("actual", metadata.get("actual_prob", 0.0))),
            delta=float(metadata.get("delta", metadata.get("signed_error", 0.0))),
            abs_error=float(metadata.get("abs_error", 0.0)),
            rationale_at_cutoff=str(metadata.get("rationale_at_cutoff", metadata.get("rationale_at_time", ""))),
            timestamp_cutoff=str(metadata.get("timestamp_cutoff", metadata.get("created_at", ""))),
            timestamp_resolution=str(metadata.get("timestamp_resolution", metadata.get("verified_at", ""))),
            metadata=metadata,
        )

    def prompt_payload(self) -> Dict[str, Any]:
        """Return a compact JSON-ready payload for LLM conditioning."""

        return {
            "domain_id": self.domain_id,
            "outcome_id": self.outcome_id,
            "domain_family": self.domain_family,
            "causal_chain": list(self.causal_chain),
            "predicted": self.predicted,
            "actual": self.actual,
            "delta": self.delta,
            "abs_error": self.abs_error,
            "rationale_at_cutoff": self.rationale_at_cutoff,
            "timestamp_cutoff": self.timestamp_cutoff,
            "timestamp_resolution": self.timestamp_resolution,
        }


class EpistemicLog:
    """Queryable verified-forecast memory stored as KG nodes."""

    def __init__(self, knowledge_graph: KnowledgeGraph) -> None:
        self.knowledge_graph = knowledge_graph

    def record(self, forecast: Forecast) -> KGNode:
        """Persist one verified forecast as an epistemic KG node."""

        from freeman.agent.epistemic import build_epistemic_log_node

        node = build_epistemic_log_node(forecast)
        self.knowledge_graph.add_node(node)
        return node

    def query(
        self,
        *,
        domain_id: str | None = None,
        domain_family: str | None = None,
        causal_chain: Sequence[str] | str | None = None,
        outcome_id: str | None = None,
        limit: int | None = None,
    ) -> List[EpistemicRecord]:
        """Return verified forecast records filtered by domain tags."""

        metadata_filters: Dict[str, Any] = {}
        if domain_id is not None:
            metadata_filters["domain_id"] = str(domain_id)
        if domain_family is not None:
            metadata_filters["domain_family"] = str(domain_family)
        if outcome_id is not None:
            metadata_filters["outcome_id"] = str(outcome_id)
        metadata_contains = None
        chain_values = normalize_causal_chain(causal_chain)
        if chain_values:
            metadata_contains = {"causal_chain": chain_values}
        nodes = self.knowledge_graph.query(
            node_type="epistemic_log",
            metadata_filters=metadata_filters or None,
            metadata_contains=metadata_contains,
        )
        records = [EpistemicRecord.from_node(node) for node in nodes]
        records.sort(key=lambda item: item.timestamp_resolution, reverse=True)
        if limit is not None:
            return records[:limit]
        return records

    def relevant_for_world(self, world: WorldState, *, limit: int = 5) -> List[EpistemicRecord]:
        """Return the most relevant verified errors for a world."""

        tags = infer_world_tags(world)
        candidates = self.query(domain_family=tags["domain_family"])
        if not candidates:
            candidates = self.query()
        chain_tokens = set(tags["causal_chain"])

        def score(record: EpistemicRecord) -> tuple[int, float, str]:
            family_bonus = 1 if record.domain_family == tags["domain_family"] else 0
            chain_overlap = len(chain_tokens.intersection(record.causal_chain))
            return (family_bonus + chain_overlap, record.abs_error, record.timestamp_resolution)

        ranked = sorted(candidates, key=score, reverse=True)
        return ranked[:limit]

    def context_for_world(self, world: WorldState, *, limit: int = 5) -> List[Dict[str, Any]]:
        """Return compact epistemic memory payloads for prompting."""

        return [record.prompt_payload() for record in self.relevant_for_world(world, limit=limit)]

    def domain_mae(self, domain_id: str, *, fallback: float = 0.0) -> float:
        """Return the rolling mean absolute error for one domain when available."""

        self_observations = self.knowledge_graph.query(
            node_type="self_observation",
            metadata_filters={"domain_id": str(domain_id)},
        )
        weighted_abs_error = 0.0
        total_forecasts = 0.0
        for node in self_observations:
            mean_abs_error = float(node.metadata.get("mean_abs_error", 0.0))
            n_forecasts = max(float(node.metadata.get("n_forecasts", 0.0)), 0.0)
            if n_forecasts <= 0.0:
                continue
            weighted_abs_error += mean_abs_error * n_forecasts
            total_forecasts += n_forecasts
        if total_forecasts > 0.0:
            return float(weighted_abs_error / total_forecasts)

        records = self.query(domain_id=domain_id)
        if records:
            return float(sum(record.abs_error for record in records) / len(records))
        return float(fallback)

    def domain_weight(self, domain_id: str, *, fallback: float = 1.0) -> float:
        """Return an inverse-noise epistemic weight from the domain MAE."""

        has_self_observation = bool(
            self.knowledge_graph.query(
                node_type="self_observation",
                metadata_filters={"domain_id": str(domain_id)},
            )
        )
        has_records = bool(self.query(domain_id=domain_id, limit=1))
        if not has_self_observation and not has_records:
            return float(fallback)
        mae = max(self.domain_mae(domain_id, fallback=0.0), 0.0)
        return float(1.0 / (1.0 + mae))


__all__ = [
    "EpistemicLog",
    "EpistemicRecord",
    "infer_domain_family",
    "infer_world_tags",
    "normalize_causal_chain",
]
