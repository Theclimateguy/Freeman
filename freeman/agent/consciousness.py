"""Deterministic consciousness-state primitives and operators for Freeman."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
import math
from typing import Any
from uuid import uuid4

from freeman.memory.knowledgegraph import KnowledgeGraph
from freeman.memory.selfmodel import SelfModelGraph, SelfModelNode, _ENGINE_CALLER_TOKEN
from freeman.utils import deep_copy_jsonable

ENGINE_TOKEN: str = _ENGINE_CALLER_TOKEN

DEFAULT_CONSCIOUSNESS_CONFIG: dict[str, Any] = {
    "idle_scheduler": {
        "threshold": 0.60,
        "weights": {
            "time_since_last_update": 0.25,
            "confidence_gap": 0.35,
            "hypothesis_age": 0.20,
            "attention_deficit": 0.20,
        },
    },
    "operators": {
        "capability_review": {
            "alpha": 2.0,
            "beta": 4.0,
        },
        "attention_rebalance": {
            "w_g": 0.30,
            "w_u": 0.30,
            "w_e": 0.25,
            "w_s": 0.15,
        },
        "trait_consolidation": {
            "lambda_k": 0.95,
            "eta_k": 0.10,
            "beta_k": 0.20,
            "min_delta": 0.01,
        },
    },
}


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


@dataclass(eq=True)
class TraceEvent:
    """One deterministic consciousness transition trace."""

    event_id: str
    timestamp: datetime
    transition_type: str
    trigger_type: str
    operator: str
    pre_state_ref: str
    post_state_ref: str
    input_refs: list[str]
    diff: dict[str, Any]
    rationale: str

    def __post_init__(self) -> None:
        if self.transition_type not in {"external", "internal"}:
            raise ValueError(f"Unsupported transition_type: {self.transition_type}")
        if self.trigger_type not in {"signal", "idle_threshold", "manual"}:
            raise ValueError(f"Unsupported trigger_type: {self.trigger_type}")
        if not str(self.event_id).startswith("trace:"):
            raise ValueError(f"TraceEvent id must start with 'trace:': {self.event_id}")
        self.timestamp = _to_datetime(self.timestamp)
        self.input_refs = [str(value) for value in self.input_refs]
        self.diff = deep_copy_jsonable(self.diff)
        self.rationale = str(self.rationale)

    def to_dict(self) -> dict[str, Any]:
        return {
            "event_id": self.event_id,
            "timestamp": _datetime_to_iso(self.timestamp),
            "transition_type": self.transition_type,
            "trigger_type": self.trigger_type,
            "operator": self.operator,
            "pre_state_ref": self.pre_state_ref,
            "post_state_ref": self.post_state_ref,
            "input_refs": list(self.input_refs),
            "diff": deep_copy_jsonable(self.diff),
            "rationale": self.rationale,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TraceEvent":
        return cls(
            event_id=data["event_id"],
            timestamp=data["timestamp"],
            transition_type=data["transition_type"],
            trigger_type=data["trigger_type"],
            operator=data["operator"],
            pre_state_ref=data["pre_state_ref"],
            post_state_ref=data["post_state_ref"],
            input_refs=list(data.get("input_refs", [])),
            diff=deep_copy_jsonable(data.get("diff", {})),
            rationale=data.get("rationale", ""),
        )


@dataclass(eq=True)
class ConsciousState:
    """Serializable consciousness state wrapper."""

    world_ref: str
    self_model_ref: SelfModelGraph
    goal_state: list[str] = field(default_factory=list)
    attention_state: dict[str, float] = field(default_factory=dict)
    trace_state: list[TraceEvent] = field(default_factory=list)
    runtime_metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "world_ref": self.world_ref,
            "self_model": self.self_model_ref.snapshot(),
            "goal_state": list(self.goal_state),
            "attention_state": {str(key): float(value) for key, value in self.attention_state.items()},
            "trace_state": [event.to_dict() for event in self.trace_state],
            "runtime_metadata": deep_copy_jsonable(self.runtime_metadata),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any], kg: KnowledgeGraph) -> "ConsciousState":
        self_model = SelfModelGraph(kg)
        self_model.load_snapshot(deep_copy_jsonable(data.get("self_model", {})))
        return cls(
            world_ref=str(data["world_ref"]),
            self_model_ref=self_model,
            goal_state=[str(value) for value in data.get("goal_state", [])],
            attention_state={str(key): float(value) for key, value in data.get("attention_state", {}).items()},
            trace_state=[TraceEvent.from_dict(item) for item in data.get("trace_state", [])],
            runtime_metadata=deep_copy_jsonable(data.get("runtime_metadata", {})),
        )


class ConsciousnessEngine:
    """Deterministic writer for self-model updates and trace events."""

    def __init__(self, state: ConsciousState, config: dict[str, Any] | None) -> None:
        self.state = state
        self.config = self._merge_config(DEFAULT_CONSCIOUSNESS_CONFIG, config or {})

    def post_pipeline_update(self, pipeline_result: Any, kg: KnowledgeGraph) -> ConsciousState:
        """Run deterministic post-pipeline self-model updates."""

        self.state.self_model_ref = SelfModelGraph(kg)
        self.state.world_ref = f"world:{pipeline_result.world.domain_id}:{pipeline_result.world.t}"
        self.state.runtime_metadata["last_domain_id"] = str(pipeline_result.world.domain_id)
        self.state.runtime_metadata["last_world_step"] = int(pipeline_result.world.t)

        for operator in (
            self._capability_review,
            self._attention_rebalance,
            self._trait_consolidation,
        ):
            raw_diff = operator()
            rationale = str(raw_diff.get("rationale", "no change"))
            clean_diff = {
                key: value
                for key, value in raw_diff.items()
                if key != "rationale" and value not in ({}, [], None)
            }
            self._apply_diff(clean_diff)
            self._write_trace(
                operator=operator.__name__.lstrip("_"),
                diff=clean_diff,
                trigger_type="manual",
                transition_type="external",
                rationale=rationale if clean_diff else "no change",
            )
        if self.state.trace_state:
            self.state.runtime_metadata["last_update_at"] = self.state.trace_state[-1].timestamp.isoformat()
        return self.state

    def maybe_deliberate(self, now: datetime) -> ConsciousState | None:
        """Run one internal deliberation act if the idle threshold is exceeded."""

        scheduler = IdleScheduler(self.config)
        if scheduler.should_deliberate(self.state, now):
            act = self._select_deliberation_act()
            raw_diff = act()
            rationale = str(raw_diff.get("rationale", "no change"))
            clean_diff = {
                key: value
                for key, value in raw_diff.items()
                if key != "rationale" and value not in ({}, [], None)
            }
            self._apply_diff(clean_diff)
            self._write_trace(
                operator=act.__name__.lstrip("_"),
                diff=clean_diff,
                trigger_type="idle_threshold",
                transition_type="internal",
                rationale=rationale if clean_diff else "no change",
                timestamp=now,
            )
            self.state.runtime_metadata["last_update_at"] = _to_datetime(now).isoformat()
            return self.state
        return None

    def _capability_review(self) -> dict[str, Any]:
        cfg = self.config["operators"]["capability_review"]
        alpha = float(cfg.get("alpha", 2.0))
        beta = float(cfg.get("beta", 4.0))
        per_domain: dict[str, dict[str, float]] = {}
        for node in self.state.self_model_ref.knowledge_graph.query(node_type="self_observation"):
            domain_id = str(node.metadata.get("domain_id", "")).strip()
            if not domain_id:
                continue
            mean_abs_error = float(node.metadata.get("mean_abs_error", 0.0))
            n_forecasts = max(float(node.metadata.get("n_forecasts", 0.0)), 0.0)
            if n_forecasts <= 0.0:
                continue
            stats = per_domain.setdefault(domain_id, {"weighted_mae": 0.0, "n": 0.0})
            stats["weighted_mae"] += mean_abs_error * n_forecasts
            stats["n"] += n_forecasts

        nodes: list[dict[str, Any]] = []
        for domain_id, stats in sorted(per_domain.items()):
            mae = stats["weighted_mae"] / max(stats["n"], 1.0)
            capability = 1.0 / (1.0 + math.exp(-(alpha - beta * mae)))
            existing = self._self_model_node("self_capability", domain_id)
            payload = {
                "domain": domain_id,
                "mean_abs_error": mae,
                "capability": capability,
                "n_forecasts": int(stats["n"]),
            }
            node = SelfModelNode(
                node_id=f"sm:self_capability:{domain_id}",
                node_type="self_capability",
                domain=domain_id,
                payload=payload,
                confidence=capability,
                created_at=existing.created_at if existing is not None else datetime.now(timezone.utc).replace(microsecond=0),
                updated_at=datetime.now(timezone.utc).replace(microsecond=0),
                source_trace_id=self.state.trace_state[-1].event_id if self.state.trace_state else None,
            )
            nodes.append(node.to_dict())
        return {"nodes": nodes, "rationale": "updated capability from rolling MAE" if nodes else "no change"}

    def _attention_rebalance(self) -> dict[str, Any]:
        cfg = self.config["operators"]["attention_rebalance"]
        w_g = float(cfg.get("w_g", 0.30))
        w_u = float(cfg.get("w_u", 0.30))
        w_e = float(cfg.get("w_e", 0.25))
        w_s = float(cfg.get("w_s", 0.15))

        goal_urgency: dict[str, float] = {}
        for node_id in self.state.goal_state:
            node = self._node_by_id(node_id)
            if node is None:
                continue
            domain_id = str(node.domain or node.payload.get("domain", "")).strip()
            if not domain_id:
                continue
            goal_urgency[domain_id] = max(goal_urgency.get(domain_id, 0.0), float(node.payload.get("urgency", node.confidence)))

        uncertainty: dict[str, float] = {}
        for node in self.state.self_model_ref.get_nodes_by_type("self_uncertainty"):
            domain_id = str(node.domain or node.payload.get("domain", "")).strip()
            if not domain_id:
                continue
            uncertainty[domain_id] = max(uncertainty.get(domain_id, 0.0), float(node.payload.get("uncertainty", node.confidence)))

        error_pressure: dict[str, float] = {}
        for node in self.state.self_model_ref.knowledge_graph.query(node_type="self_observation"):
            domain_id = str(node.metadata.get("domain_id", "")).strip()
            if not domain_id:
                continue
            error_pressure[domain_id] = max(error_pressure.get(domain_id, 0.0), float(node.metadata.get("mean_abs_error", 0.0)))
        for node in self.state.self_model_ref.get_nodes_by_type("self_capability"):
            domain_id = str(node.domain or node.payload.get("domain", "")).strip()
            if not domain_id:
                continue
            error_pressure[domain_id] = max(error_pressure.get(domain_id, 0.0), float(node.payload.get("mean_abs_error", 0.0)))

        saturation = {str(key): float(value) for key, value in self.state.attention_state.items()}
        domains = sorted(set(goal_urgency) | set(uncertainty) | set(error_pressure) | set(saturation))
        if not domains:
            return {"rationale": "no change"}

        raw_scores = {
            domain_id: max(
                0.0,
                w_g * goal_urgency.get(domain_id, 0.0)
                + w_u * uncertainty.get(domain_id, 0.0)
                + w_e * error_pressure.get(domain_id, 0.0)
                - w_s * saturation.get(domain_id, 0.0),
            )
            for domain_id in domains
        }
        total = sum(raw_scores.values())
        if total <= 0.0:
            normalized = {domain_id: 1.0 / len(domains) for domain_id in domains}
        else:
            normalized = {domain_id: raw_scores[domain_id] / total for domain_id in domains}

        nodes: list[dict[str, Any]] = []
        for domain_id, weight in normalized.items():
            existing = self._self_model_node("attention_focus", domain_id)
            node = SelfModelNode(
                node_id=f"sm:attention_focus:{domain_id}",
                node_type="attention_focus",
                domain=domain_id,
                payload={
                    "domain": domain_id,
                    "weight": weight,
                    "goal_urgency": goal_urgency.get(domain_id, 0.0),
                    "uncertainty": uncertainty.get(domain_id, 0.0),
                    "error_pressure": error_pressure.get(domain_id, 0.0),
                    "saturation": saturation.get(domain_id, 0.0),
                },
                confidence=weight,
                created_at=existing.created_at if existing is not None else datetime.now(timezone.utc).replace(microsecond=0),
                updated_at=datetime.now(timezone.utc).replace(microsecond=0),
                source_trace_id=self.state.trace_state[-1].event_id if self.state.trace_state else None,
            )
            nodes.append(node.to_dict())
        return {
            "nodes": nodes,
            "attention_state": normalized,
            "rationale": "rebalanced attention weights",
        }

    def _trait_consolidation(self) -> dict[str, Any]:
        cfg = self.config["operators"]["trait_consolidation"]
        lambda_k = float(cfg.get("lambda_k", 0.95))
        eta_k = float(cfg.get("eta_k", 0.10))
        beta_k = float(cfg.get("beta_k", 0.20))
        min_delta = float(cfg.get("min_delta", 0.01))

        nodes: list[dict[str, Any]] = []
        for trait in self.state.self_model_ref.get_nodes_by_type("identity_trait"):
            support = float(trait.payload.get("trait_support", trait.confidence))
            pattern_observed = bool(trait.payload.get("pattern_observed", False))
            delta_mae = max(0.0, float(trait.payload.get("delta_mae", 0.0)))
            new_support = lambda_k * support + eta_k * (1.0 if pattern_observed else 0.0) - beta_k * delta_mae
            new_support = min(max(new_support, 0.0), 1.0)
            delta_support = new_support - support
            if abs(delta_support) <= min_delta:
                continue
            updated_payload = deep_copy_jsonable(trait.payload)
            updated_payload["trait_support"] = new_support
            updated_payload["delta_support"] = delta_support
            updated = SelfModelNode(
                node_id=trait.node_id,
                node_type="identity_trait",
                domain=trait.domain,
                payload=updated_payload,
                confidence=new_support,
                created_at=trait.created_at,
                updated_at=datetime.now(timezone.utc).replace(microsecond=0),
                source_trace_id=self.state.trace_state[-1].event_id if self.state.trace_state else None,
            )
            nodes.append(updated.to_dict())
        return {"nodes": nodes, "rationale": "consolidated identity traits" if nodes else "no change"}

    def _hypothesis_aging(self) -> dict[str, Any]:
        tau_h = 50.0
        updated_nodes: list[dict[str, Any]] = []
        added_edges: list[dict[str, Any]] = []

        for hypothesis in self.state.self_model_ref.get_nodes_by_type("active_hypothesis"):
            age_steps = max(float(hypothesis.payload.get("age_steps", 0.0)), 0.0)
            decay = math.exp(-age_steps / tau_h)
            new_confidence = min(max(hypothesis.confidence * decay, 0.0), 1.0)
            if new_confidence < 0.05:
                updated_nodes.append(
                    SelfModelNode(
                        node_id=hypothesis.node_id,
                        node_type="self_uncertainty",
                        domain=hypothesis.domain,
                        payload={
                            **deep_copy_jsonable(hypothesis.payload),
                            "migrated_from": hypothesis.node_id,
                            "uncertainty": 1.0 - new_confidence,
                        },
                        confidence=min(max(1.0 - new_confidence, 0.0), 1.0),
                        created_at=hypothesis.created_at,
                        updated_at=datetime.now(timezone.utc).replace(microsecond=0),
                        source_trace_id=self.state.trace_state[-1].event_id if self.state.trace_state else None,
                    ).to_dict()
                )
                added_edges.append(
                    {
                        "edge_id": f"sm-edge:revises:{hypothesis.node_id}",
                        "edge_type": "revises",
                        "source_id": hypothesis.node_id,
                        "target_id": hypothesis.node_id,
                        "weight": 1.0,
                        "created_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
                        "trace_id": self.state.trace_state[-1].event_id if self.state.trace_state else None,
                    }
                )
                continue

            updated_payload = deep_copy_jsonable(hypothesis.payload)
            updated_payload["age_steps"] = age_steps + 1.0
            updated_payload["decay_factor"] = decay
            updated_nodes.append(
                SelfModelNode(
                    node_id=hypothesis.node_id,
                    node_type="active_hypothesis",
                    domain=hypothesis.domain,
                    payload=updated_payload,
                    confidence=new_confidence,
                    created_at=hypothesis.created_at,
                    updated_at=datetime.now(timezone.utc).replace(microsecond=0),
                    source_trace_id=self.state.trace_state[-1].event_id if self.state.trace_state else None,
                ).to_dict()
            )

        diff: dict[str, Any] = {}
        if updated_nodes:
            diff["nodes"] = updated_nodes
        if added_edges:
            diff["edges"] = added_edges
        diff["rationale"] = "aged active hypotheses" if diff else "no change"
        return diff

    def _consistency_check(self) -> dict[str, Any]:
        hypotheses = {
            node.node_id: node
            for node in self.state.self_model_ref.get_nodes_by_type("active_hypothesis")
            if node.confidence > 0.5
        }
        nodes: list[dict[str, Any]] = []
        edges: list[dict[str, Any]] = []
        for edge in self.state.self_model_ref.get_edges_by_type("contradicts"):
            if edge.source_id not in hypotheses or edge.target_id not in hypotheses:
                continue
            left = hypotheses[edge.source_id]
            right = hypotheses[edge.target_id]
            pair_slug = ":".join(sorted([left.node_id.split(":")[-1], right.node_id.split(":")[-1]]))
            node_id = f"sm:self_uncertainty:{pair_slug}"
            nodes.append(
                SelfModelNode(
                    node_id=node_id,
                    node_type="self_uncertainty",
                    domain=left.domain or right.domain,
                    payload={
                        "uncertainty": 1.0,
                        "reason": "active contradiction",
                        "hypothesis_ids": [left.node_id, right.node_id],
                    },
                    confidence=1.0,
                    created_at=datetime.now(timezone.utc).replace(microsecond=0),
                    updated_at=datetime.now(timezone.utc).replace(microsecond=0),
                    source_trace_id=self.state.trace_state[-1].event_id if self.state.trace_state else None,
                ).to_dict()
            )
            edges.extend(
                [
                    {
                        "edge_id": f"sm-edge:contradicts:{node_id}:{left.node_id}",
                        "edge_type": "contradicts",
                        "source_id": node_id,
                        "target_id": left.node_id,
                        "weight": 1.0,
                        "created_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
                        "trace_id": self.state.trace_state[-1].event_id if self.state.trace_state else None,
                    },
                    {
                        "edge_id": f"sm-edge:contradicts:{node_id}:{right.node_id}",
                        "edge_type": "contradicts",
                        "source_id": node_id,
                        "target_id": right.node_id,
                        "weight": 1.0,
                        "created_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
                        "trace_id": self.state.trace_state[-1].event_id if self.state.trace_state else None,
                    },
                ]
            )
        diff: dict[str, Any] = {}
        if nodes:
            diff["nodes"] = nodes
        if edges:
            diff["edges"] = edges
        diff["rationale"] = "emitted contradiction uncertainty nodes" if nodes else "no change"
        return diff

    def _apply_diff(self, diff: dict[str, Any]) -> None:
        for node_id in diff.get("remove_node_ids", []):
            if node_id in self.state.self_model_ref.knowledge_graph.graph:
                self.state.self_model_ref.knowledge_graph.graph.remove_node(node_id)
        for node_payload in diff.get("nodes", []):
            self.state.self_model_ref.write(SelfModelNode.from_dict(node_payload), caller_token=ENGINE_TOKEN)
        for edge_payload in diff.get("edges", []):
            from freeman.memory.selfmodel import SelfModelEdge

            self.state.self_model_ref.write(SelfModelEdge.from_dict(edge_payload), caller_token=ENGINE_TOKEN)
        if "attention_state" in diff:
            self.state.attention_state = {
                str(domain_id): float(weight)
                for domain_id, weight in diff["attention_state"].items()
            }

    def _write_trace(
        self,
        *,
        operator: str,
        diff: dict[str, Any],
        trigger_type: str,
        transition_type: str,
        rationale: str,
        timestamp: datetime | None = None,
    ) -> None:
        index = len(self.state.trace_state)
        event = TraceEvent(
            event_id=f"trace:{uuid4()}",
            timestamp=timestamp or datetime.now(timezone.utc).replace(microsecond=0),
            transition_type=transition_type,
            trigger_type=trigger_type,
            operator=operator,
            pre_state_ref=f"state:{index}",
            post_state_ref=f"state:{index + 1}",
            input_refs=self._trace_input_refs(diff),
            diff=deep_copy_jsonable(diff),
            rationale=rationale,
        )
        self.state.trace_state.append(event)

    def _self_model_node(self, node_type: str, domain_id: str) -> SelfModelNode | None:
        for node in self.state.self_model_ref.get_nodes_by_type(node_type):
            if str(node.domain or "") == str(domain_id):
                return node
        return None

    def _node_by_id(self, node_id: str) -> SelfModelNode | None:
        node = self.state.self_model_ref.knowledge_graph.get_node(node_id, lazy_embed=False)
        if node is None:
            return None
        return SelfModelNode.from_kg_node(node)

    def _trace_input_refs(self, diff: dict[str, Any]) -> list[str]:
        refs = [node["node_id"] for node in diff.get("nodes", []) if "node_id" in node]
        refs.extend(edge["edge_id"] for edge in diff.get("edges", []) if "edge_id" in edge)
        refs.extend(str(node_id) for node_id in diff.get("remove_node_ids", []))
        refs.extend(f"attention:{domain_id}" for domain_id in diff.get("attention_state", {}))
        return refs

    def _select_deliberation_act(self):
        tau_h = 50.0
        hypotheses = self.state.self_model_ref.get_nodes_by_type("active_hypothesis")
        mean_age = (
            sum(max(float(node.payload.get("age_steps", 0.0)), 0.0) for node in hypotheses) / len(hypotheses)
            if hypotheses
            else 0.0
        )
        aging_urgency = mean_age / tau_h if tau_h > 0.0 else 0.0
        contradiction_pairs = sum(
            1
            for edge in self.state.self_model_ref.get_edges_by_type("contradicts")
            if self._node_by_id(edge.source_id) is not None and self._node_by_id(edge.target_id) is not None
        )
        consistency_urgency = float(contradiction_pairs)
        return self._consistency_check if consistency_urgency > aging_urgency else self._hypothesis_aging

    def _merge_config(self, base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
        merged = deep_copy_jsonable(base)
        for key, value in override.items():
            if isinstance(value, dict) and isinstance(merged.get(key), dict):
                merged[key] = self._merge_config(merged[key], value)
            else:
                merged[key] = deep_copy_jsonable(value)
        return merged


class IdleScheduler:
    """Deterministic idle-deliberation scorer."""

    def __init__(self, config: dict[str, Any]) -> None:
        idle_cfg = deep_copy_jsonable(config.get("idle_scheduler", {}))
        self.threshold = float(idle_cfg.get("threshold", 0.60))
        weights = idle_cfg.get("weights", {})
        self.weights = {
            "time_since_last_update": float(weights.get("time_since_last_update", 0.25)),
            "confidence_gap": float(weights.get("confidence_gap", 0.35)),
            "hypothesis_age": float(weights.get("hypothesis_age", 0.20)),
            "attention_deficit": float(weights.get("attention_deficit", 0.20)),
        }

    def score(self, state: ConsciousState, now: datetime) -> float:
        now_dt = _to_datetime(now)
        raw_time = self._time_since_last_update(state, now_dt)
        raw_gap = float(state.runtime_metadata.get("confidence_gap", 0.0))
        raw_age = self._mean_hypothesis_age(state)
        raw_attn = float(state.runtime_metadata.get("attention_deficit", self._attention_deficit(state)))
        components = {
            "time_since_last_update": self._normalize(state, "time_since_last_update", raw_time, default_max=86400.0),
            "confidence_gap": self._normalize(state, "confidence_gap", raw_gap, default_max=1.0),
            "hypothesis_age": self._normalize(state, "hypothesis_age", raw_age, default_max=100.0),
            "attention_deficit": self._normalize(state, "attention_deficit", raw_attn, default_max=1.0),
        }
        return sum(self.weights[name] * components[name] for name in self.weights)

    def should_deliberate(self, state: ConsciousState, now: datetime) -> bool:
        return self.score(state, now) > self.threshold

    def _normalize(self, state: ConsciousState, name: str, value: float, *, default_max: float) -> float:
        stats = deep_copy_jsonable(state.runtime_metadata.get("idle_scheduler_stats", {})).get(name, {})
        min_value = float(stats.get("min", 0.0))
        max_value = float(stats.get("max", default_max))
        if max_value <= min_value:
            return 0.0
        return min(max((float(value) - min_value) / (max_value - min_value), 0.0), 1.0)

    def _time_since_last_update(self, state: ConsciousState, now: datetime) -> float:
        last = state.runtime_metadata.get("last_update_at")
        if last is None and state.trace_state:
            last = state.trace_state[-1].timestamp
        if last is None:
            return 0.0
        return max((now - _to_datetime(last)).total_seconds(), 0.0)

    def _mean_hypothesis_age(self, state: ConsciousState) -> float:
        hypotheses = state.self_model_ref.get_nodes_by_type("active_hypothesis")
        if not hypotheses:
            return 0.0
        return sum(max(float(node.payload.get("age_steps", 0.0)), 0.0) for node in hypotheses) / len(hypotheses)

    def _attention_deficit(self, state: ConsciousState) -> float:
        if not state.attention_state:
            return 0.0
        return max(0.0, 1.0 - max(float(value) for value in state.attention_state.values()))


__all__ = [
    "ConsciousState",
    "ConsciousnessEngine",
    "DEFAULT_CONSCIOUSNESS_CONFIG",
    "ENGINE_TOKEN",
    "IdleScheduler",
    "TraceEvent",
]
