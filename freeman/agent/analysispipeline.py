"""Compile -> simulate -> verify -> score -> update-KG pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
import logging
from pathlib import Path
from typing import Any, Dict, Iterable, List

import yaml
from freeman.agent.forecastregistry import Forecast, ForecastRegistry
from freeman.agent.epistemic import (
    build_belief_conflict_node,
    build_disagreement_node,
    compute_confidence_weighted_disagreement,
    detect_belief_conflict,
    extract_reference_outcome_probs,
)
from freeman.agent.proactiveemitter import ProactiveEmitter, ProactiveEvent

from freeman.core.compilevalidator import CompileValidator, OperatorFitReport
from freeman.core.scorer import raw_outcome_scores, score_outcomes
from freeman.core.types import ParameterVector, Policy
from freeman.core.world import WorldState
from freeman.domain.compiler import DomainCompiler
from freeman.game.runner import GameRunner, SimConfig
from freeman.memory.beliefconflictlog import BeliefConflictLog
from freeman.memory.epistemiclog import EpistemicLog, infer_world_tags
from freeman.memory.knowledgegraph import KGNode, KnowledgeGraph
from freeman.memory.reconciler import Reconciler, ReconciliationResult
from freeman.memory.sessionlog import KGDelta, SessionLog
from freeman.verifier.verifier import Verifier, VerifierConfig

LOGGER = logging.getLogger(__name__)


@dataclass
class AnalysisPipelineResult:
    """Structured output of one analysis pipeline run."""

    world: WorldState
    simulation: Dict[str, Any]
    verification: Dict[str, Any]
    raw_scores: Dict[str, float]
    dominant_outcome: str | None
    knowledge_graph_path: str
    reconciliation: ReconciliationResult | None = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    proactive_events: List[ProactiveEvent] = field(default_factory=list)


@dataclass
class AnalysisPipelineConfig:
    """Retrieval limits used by the analysis pipeline."""

    retrieval_top_k: int = 15
    max_context_nodes: int = 30

    @classmethod
    def from_config(cls, config_path: str | Path | None = None) -> "AnalysisPipelineConfig":
        config_file = Path(config_path).resolve() if config_path is not None else Path(__file__).resolve().parents[2] / "config.yaml"
        if not config_file.exists():
            return cls()
        payload = yaml.safe_load(config_file.read_text(encoding="utf-8")) or {}
        memory_cfg = payload.get("memory", {})
        return cls(
            retrieval_top_k=int(memory_cfg.get("retrieval_top_k", 15)),
            max_context_nodes=int(memory_cfg.get("max_context_nodes", 30)),
        )


class AnalysisPipeline:
    """Orchestrate the main v0.1 agent analysis path."""

    def __init__(
        self,
        *,
        compiler: DomainCompiler | None = None,
        sim_config: SimConfig | None = None,
        verifier_config: VerifierConfig | None = None,
        knowledge_graph: KnowledgeGraph | None = None,
        reconciler: Reconciler | None = None,
        forecast_registry: ForecastRegistry | None = None,
        emitter: ProactiveEmitter | None = None,
        config: AnalysisPipelineConfig | None = None,
        config_path: str | Path | None = None,
    ) -> None:
        self.compiler = compiler or DomainCompiler()
        self.sim_config = sim_config or SimConfig()
        self.verifier = Verifier(verifier_config or self.sim_config)
        self.runner = GameRunner(self.sim_config)
        self.knowledge_graph = knowledge_graph or KnowledgeGraph()
        self.reconciler = reconciler or Reconciler()
        self.forecast_registry = forecast_registry
        self.emitter = emitter
        self.config = config or AnalysisPipelineConfig.from_config(config_path)
        self.epistemic_log = EpistemicLog(self.knowledge_graph)
        self.belief_conflict_log = BeliefConflictLog(self.knowledge_graph)
        self._previous_outcome_probs: Dict[str, Dict[str, float]] = {}
        self._outcome_history: Dict[str, List[Dict[str, float]]] = {}

    def run(
        self,
        schema: Dict[str, Any] | WorldState,
        *,
        policies: Iterable[Policy | Dict[str, Any]] = (),
        session_log: SessionLog | None = None,
    ) -> AnalysisPipelineResult:
        """Run the full compile/simulate/verify/score/update workflow."""

        extra_summary_metadata: Dict[str, Any] = {}
        if isinstance(schema, WorldState):
            world = schema.clone()
        else:
            world = self.compiler.compile(schema)
            historical_data = self._extract_historical_data(schema)
            if historical_data:
                validation_report = CompileValidator(
                    compiler=self.compiler,
                    sim_config=self.sim_config,
                ).validate(schema, historical_data=historical_data)
                warnings = self._operator_fit_warning_messages(validation_report.operator_fit_reports)
                if validation_report.operator_fit_reports:
                    extra_summary_metadata["operator_fit_reports"] = [
                        {
                            "resource_id": report.resource_id,
                            "chosen_operator": report.chosen_operator,
                            "scores": dict(report.scores),
                            "best_operator": report.best_operator,
                            "gap": float(report.gap),
                            "warn": bool(report.warn),
                        }
                        for report in validation_report.operator_fit_reports
                    ]
                if warnings:
                    extra_summary_metadata["warnings"] = warnings
                    for message in warnings:
                        LOGGER.warning(message)
        return self._run_world(
            world,
            policies=policies,
            session_log=session_log,
            extra_summary_metadata=extra_summary_metadata or None,
        )

    def update(
        self,
        previous_world: WorldState,
        parameter_vector: ParameterVector,
        *,
        policies: Iterable[Policy | Dict[str, Any]] = (),
        signal_text: str | None = None,
        session_log: SessionLog | None = None,
    ) -> AnalysisPipelineResult:
        """Apply a new dynamic parameter layer to an existing world and re-run simulation."""

        LOGGER.info(
            "analysis_update domain_id=%s shock_decay=%.3f outcome_modifiers=%s",
            previous_world.domain_id,
            float(parameter_vector.shock_decay),
            dict(parameter_vector.outcome_modifiers),
        )
        world = previous_world.clone()
        world.parameter_vector = parameter_vector
        return self._run_world(
            world,
            policies=policies,
            session_log=session_log,
            prior_outcome_probs=score_outcomes(previous_world),
            update_signal_text=signal_text,
            extra_summary_metadata={"parameter_vector": parameter_vector.snapshot()},
        )

    def _run_world(
        self,
        world: WorldState,
        *,
        policies: Iterable[Policy | Dict[str, Any]] = (),
        session_log: SessionLog | None = None,
        prior_outcome_probs: Dict[str, float] | None = None,
        update_signal_text: str | None = None,
        extra_summary_metadata: Dict[str, Any] | None = None,
    ) -> AnalysisPipelineResult:
        """Execute the compile/simulate/verify/score/update workflow from an initialized world."""

        verification = self.verifier.run(world, levels=(1, 2)).snapshot()
        policy_objects = [
            policy if isinstance(policy, Policy) else Policy.from_snapshot(policy)
            for policy in policies
        ]
        sim_result = self.runner.run(world.clone(), policy_objects)
        final_world = WorldState.from_snapshot(sim_result.trajectory[-1])
        raw_scores = raw_outcome_scores(final_world)
        if session_log is None:
            session_log = SessionLog(session_id=f"analysis:{final_world.domain_id}:{final_world.t}")
        dominant_outcome = None
        if sim_result.final_outcome_probs:
            dominant_outcome = max(sim_result.final_outcome_probs, key=sim_result.final_outcome_probs.get)
        analysis_node_id = f"analysis:{final_world.domain_id}:{final_world.t}"
        parameter_effect_trace: List[Dict[str, Any]] = []
        parameter_effect_mismatches: List[Dict[str, Any]] = []
        if prior_outcome_probs is not None:
            parameter_effect_trace = self._parameter_effect_trace(
                prior_outcome_probs,
                sim_result.final_outcome_probs,
                final_world.parameter_vector,
            )
            parameter_effect_mismatches = [
                entry for entry in parameter_effect_trace if not bool(entry.get("sign_match", True))
            ]
            if parameter_effect_mismatches:
                final_world.parameter_vector.conflict_flag = True
        reference_outcome_probs = extract_reference_outcome_probs(final_world)
        disagreement_snapshot = compute_confidence_weighted_disagreement(
            sim_result.final_outcome_probs,
            reference_outcome_probs,
            belief_confidence=float(sim_result.confidence),
        )
        recorded_forecasts = self._record_forecasts(
            final_world,
            sim_result.final_outcome_probs,
            session_log,
            belief_confidence=float(sim_result.confidence),
            reference_outcome_probs=reference_outcome_probs,
            analysis_node_id=analysis_node_id,
        )
        signal_text = self._signal_text(final_world, session_log)
        context_nodes = self._get_context_nodes(signal_text)
        epistemic_node_ids: List[str] = []
        belief_conflicts: List[Dict[str, Any]] = []

        summary_node = KGNode(
            id=analysis_node_id,
            label=f"Analysis {final_world.domain_id}",
            node_type="analysis_run",
            content=(
                f"Dominant outcome: {dominant_outcome or 'n/a'}; "
                f"confidence={sim_result.confidence:.4f}; "
                f"violations={len(sim_result.violations)}"
            ),
            confidence=max(sim_result.confidence, 0.15),
            metadata={
                "domain_id": final_world.domain_id,
                "dominant_outcome": dominant_outcome,
                "final_outcome_probs": sim_result.final_outcome_probs,
                "context_node_ids": [node.id for node in context_nodes],
                "context_node_count": len(context_nodes),
                "forecast_ids": [forecast.forecast_id for forecast in recorded_forecasts],
                "verification": verification,
                "parameter_vector": final_world.parameter_vector.snapshot(),
                "reference_outcome_probs": reference_outcome_probs,
                "disagreement_snapshot": disagreement_snapshot,
                "parameter_effect_trace": parameter_effect_trace,
                "parameter_effect_mismatches": parameter_effect_mismatches,
                **(extra_summary_metadata or {}),
            },
        )
        self.knowledge_graph.add_node(summary_node)

        session_log.add_kg_delta(
            KGDelta(
                operation="add_node",
                target_id=summary_node.id,
                payload={"node": summary_node.snapshot()},
                support=max(1, len(sim_result.trajectory) // 5),
                contradiction=sum(1 for violation in sim_result.violations if violation.severity == "hard"),
            )
        )
        disagreement_node = build_disagreement_node(
            domain_id=final_world.domain_id,
            step=final_world.t,
            disagreement_snapshot=disagreement_snapshot,
        )
        if disagreement_node is not None:
            self.knowledge_graph.add_node(disagreement_node)
            epistemic_node_ids.append(disagreement_node.id)
            session_log.add_kg_delta(
                KGDelta(
                    operation="add_node",
                    target_id=disagreement_node.id,
                    payload={"node": disagreement_node.snapshot()},
                    support=1,
                    contradiction=0,
                    metadata={"epistemic_event_type": "belief_disagreement"},
                )
            )
        if parameter_effect_mismatches:
            effect_conflict_node = self._build_parameter_effect_conflict_node(
                domain_id=final_world.domain_id,
                step=final_world.t,
                mismatches=parameter_effect_mismatches,
                rationale=final_world.parameter_vector.rationale,
                signal_text=update_signal_text or "",
            )
            self.knowledge_graph.add_node(effect_conflict_node)
            epistemic_node_ids.append(effect_conflict_node.id)
            session_log.add_kg_delta(
                KGDelta(
                    operation="add_node",
                    target_id=effect_conflict_node.id,
                    payload={"node": effect_conflict_node.snapshot()},
                    support=1,
                    contradiction=1,
                    metadata={"epistemic_event_type": "parameter_effect_conflict"},
                )
            )
        if prior_outcome_probs is not None:
            momentum_reference_outcome_probs = self._momentum_reference(
                final_world.domain_id,
                prior_outcome_probs,
            )
            conflict_snapshot = detect_belief_conflict(
                prior_outcome_probs,
                sim_result.final_outcome_probs,
                momentum_reference_outcome_probs=momentum_reference_outcome_probs,
                signal_source=str(final_world.metadata.get("signal_source", "update_signal")),
                signal_text=update_signal_text or "",
                rationale=final_world.parameter_vector.rationale,
                parameter_conflict_flag=bool(final_world.parameter_vector.conflict_flag),
            )
            if conflict_snapshot is not None:
                conflict_node = build_belief_conflict_node(
                    domain_id=final_world.domain_id,
                    step=final_world.t,
                    conflict_snapshot=conflict_snapshot,
                    rationale=final_world.parameter_vector.rationale,
                    signal_text=update_signal_text or "",
                )
                self.belief_conflict_log.record(conflict_node)
                epistemic_node_ids.append(conflict_node.id)
                belief_conflicts.append(dict(conflict_node.metadata))
                session_log.add_kg_delta(
                    KGDelta(
                        operation="add_node",
                        target_id=conflict_node.id,
                        payload={"node": conflict_node.snapshot()},
                        support=1,
                        contradiction=1,
                        metadata={"epistemic_event_type": "belief_conflict"},
                    )
                )
        reconciliation = self.reconciler.reconcile(self.knowledge_graph, session_log)

        result = AnalysisPipelineResult(
            world=final_world,
            simulation=sim_result.snapshot(),
            verification=verification,
            raw_scores=raw_scores,
            dominant_outcome=dominant_outcome,
            knowledge_graph_path=str(self.knowledge_graph.json_path),
            reconciliation=reconciliation,
            metadata={
                "context_node_ids": [node.id for node in context_nodes],
                "context_node_count": len(context_nodes),
                "forecast_ids": [forecast.forecast_id for forecast in recorded_forecasts],
                "forecast_count": len(recorded_forecasts),
                "epistemic_event_ids": epistemic_node_ids,
                "steps_run": sim_result.steps_run,
                "fixed_point_iters": sim_result.metadata.get("fixed_point_iters"),
                "parameter_vector": final_world.parameter_vector.snapshot(),
                "reference_outcome_probs": reference_outcome_probs,
                "disagreement_snapshot": disagreement_snapshot,
                "belief_conflicts": belief_conflicts,
                "parameter_effect_trace": parameter_effect_trace,
                "parameter_effect_mismatches": parameter_effect_mismatches,
                **(extra_summary_metadata or {}),
            },
        )
        if self.emitter is not None:
            prev_probs = self._previous_outcome_probs.get(final_world.domain_id)
            result.proactive_events = self.emitter.evaluate(result, prev_probs=prev_probs)
        if sim_result.final_outcome_probs:
            self._previous_outcome_probs[final_world.domain_id] = {
                outcome_id: float(prob)
                for outcome_id, prob in sim_result.final_outcome_probs.items()
            }
            self._append_outcome_history(final_world.domain_id, sim_result.final_outcome_probs)
        return result

    def _extract_historical_data(self, schema: Dict[str, Any]) -> Dict[str, List[float]] | None:
        """Return optional resource histories bundled with a raw schema payload."""

        payload = schema.get("historical_data")
        if payload is None:
            payload = schema.get("metadata", {}).get("historical_data")
        if not isinstance(payload, dict):
            return None
        histories: Dict[str, List[float]] = {}
        for resource_id, series in payload.items():
            if not isinstance(series, list):
                continue
            histories[str(resource_id)] = [float(value) for value in series]
        return histories or None

    def _operator_fit_warning_messages(self, reports: List[OperatorFitReport]) -> List[str]:
        """Format operator-selection warnings for interface layers."""

        messages: List[str] = []
        for report in reports:
            if not report.warn:
                continue
            messages.append(
                f"[WARN] Resource '{report.resource_id}': chosen operator '{report.chosen_operator}' "
                f"has RMSE gap +{report.gap * 100.0:.0f}% vs best '{report.best_operator}'. "
                "Consider switching operator in schema."
            )
        return messages

    def _signal_text(self, world: WorldState, session_log: SessionLog | None) -> str:
        """Build a compact retrieval query from the incoming signal context."""

        parts = [world.domain_id]
        if session_log is not None:
            for delta in session_log.kg_deltas:
                if delta.operation not in {"add_node", "update_node"}:
                    continue
                payload = delta.payload.get("node", delta.payload)
                node = payload if isinstance(payload, KGNode) else KGNode.from_snapshot(payload)
                parts.extend([node.label, node.content])
        return " ".join(part for part in parts if part).strip()

    def _get_context_nodes(self, signal_text: str) -> List[KGNode]:
        """Select retrieval context without exposing the full KG to downstream LLMs."""

        if self.knowledge_graph.vectorstore is not None:
            nodes = self.knowledge_graph.semantic_query(signal_text, top_k=self.config.retrieval_top_k)
        else:
            nodes = [node for node in self.knowledge_graph.nodes() if node.status != "archived"]
        return nodes[: self.config.max_context_nodes]

    def _record_forecasts(
        self,
        final_world: WorldState,
        final_outcome_probs: Dict[str, float],
        session_log: SessionLog,
        *,
        belief_confidence: float,
        reference_outcome_probs: Dict[str, float],
        analysis_node_id: str,
    ) -> List[Forecast]:
        """Record probabilistic outcome forecasts for later verification."""

        if self.forecast_registry is None or not final_outcome_probs:
            return []
        recorded: List[Forecast] = []
        for outcome_id, prob in final_outcome_probs.items():
            tags = infer_world_tags(final_world)
            reference_prob = reference_outcome_probs.get(outcome_id)
            reference_gap = float(prob - reference_prob) if reference_prob is not None else None
            confidence_weighted_gap = (
                float(reference_gap * belief_confidence) if reference_gap is not None else None
            )
            forecast = Forecast(
                forecast_id=f"{final_world.domain_id}:{final_world.t}:{outcome_id}",
                domain_id=final_world.domain_id,
                outcome_id=outcome_id,
                predicted_prob=prob,
                session_id=session_log.session_id,
                horizon_steps=self.sim_config.max_steps,
                created_at=datetime.now(timezone.utc).replace(microsecond=0),
                created_step=int(final_world.t),
                metadata={
                    "analysis_node_id": analysis_node_id,
                    "belief_confidence": float(belief_confidence),
                    "rationale_at_time": final_world.parameter_vector.rationale,
                    "parameter_vector": final_world.parameter_vector.snapshot(),
                    "reference_prob": reference_prob,
                    "reference_gap": reference_gap,
                    "confidence_weighted_gap": confidence_weighted_gap,
                    "domain_family": tags["domain_family"],
                    "causal_chain": tags["causal_chain"],
                },
            )
            self.forecast_registry.record(forecast)
            recorded.append(forecast)
        return recorded

    def verify_forecast(
        self,
        forecast_id: str,
        *,
        actual_prob: float,
        verified_at: datetime,
        session_log: SessionLog | None = None,
    ) -> Forecast:
        """Verify a forecast and persist an epistemic error trace into the KG."""

        if self.forecast_registry is None:
            raise ValueError("verify_forecast requires an attached ForecastRegistry.")
        if session_log is None:
            session_log = SessionLog(session_id=f"verify:{forecast_id}")
        forecast = self.forecast_registry.verify(
            forecast_id,
            actual_prob=actual_prob,
            verified_at=verified_at,
        )
        epistemic_node = self.epistemic_log.record(forecast)
        session_log.add_kg_delta(
            KGDelta(
                operation="add_node",
                target_id=epistemic_node.id,
                payload={"node": epistemic_node.snapshot()},
                support=1,
                contradiction=int(forecast.error > 0.0),
                metadata={"epistemic_event_type": "forecast_verification"},
            )
        )
        self_model_node = self.reconciler.update_self_model(self.knowledge_graph, forecast)
        session_log.add_kg_delta(
            KGDelta(
                operation="add_node",
                target_id=self_model_node.id,
                payload={"node": self_model_node.snapshot()},
                support=1,
                contradiction=0,
                metadata={"epistemic_event_type": "self_model_update"},
            )
        )
        self.reconciler.reconcile(self.knowledge_graph, session_log)
        return forecast

    def _momentum_reference(
        self,
        domain_id: str,
        prior_outcome_probs: Dict[str, float],
    ) -> Dict[str, float] | None:
        """Return the pre-prior belief surface needed to estimate current momentum."""

        history = self._outcome_history.get(domain_id, [])
        if not history:
            return None
        if len(history) == 1:
            return history[0]
        if self._same_probability_surface(history[-1], prior_outcome_probs):
            return history[-2]
        return history[-1]

    def _append_outcome_history(self, domain_id: str, outcome_probs: Dict[str, float]) -> None:
        """Keep a short rolling history of belief surfaces per domain."""

        history = self._outcome_history.setdefault(domain_id, [])
        history.append({outcome_id: float(prob) for outcome_id, prob in outcome_probs.items()})
        if len(history) > 5:
            del history[:-5]

    def _same_probability_surface(
        self,
        left: Dict[str, float],
        right: Dict[str, float],
        *,
        tolerance: float = 1.0e-9,
    ) -> bool:
        """Return whether two outcome distributions are effectively identical."""

        keys = set(left) | set(right)
        return all(abs(float(left.get(key, 0.0)) - float(right.get(key, 0.0))) <= tolerance for key in keys)

    def _parameter_effect_trace(
        self,
        prior_outcome_probs: Dict[str, float],
        posterior_outcome_probs: Dict[str, float],
        parameter_vector: ParameterVector,
        *,
        tolerance: float = 1.0e-6,
    ) -> List[Dict[str, Any]]:
        """Compare intended modifier directions with realized posterior probability shifts."""

        trace: List[Dict[str, Any]] = []
        for conflict in parameter_vector.repair_conflicts:
            trace.append(
                {
                    "mismatch_type": "unknown_outcome_id",
                    "outcome_id": conflict.get("corrected_to") or conflict.get("hallucinated_outcome_id"),
                    "hallucinated_outcome_id": conflict.get("hallucinated_outcome_id"),
                    "corrected_to": conflict.get("corrected_to"),
                    "dropped": bool(conflict.get("dropped", False)),
                    "modifier": conflict.get("modifier"),
                    "intended_direction": "unknown",
                    "actual_direction": "corrected" if conflict.get("corrected_to") else "dropped",
                    "sign_match": False,
                    "prior_probability": None,
                    "posterior_probability": None,
                    "delta_probability": 0.0,
                    "fuzzy_ratio": conflict.get("fuzzy_ratio"),
                }
            )
        for outcome_id, modifier in parameter_vector.outcome_modifiers.items():
            modifier_value = float(modifier)
            if abs(modifier_value - 1.0) <= tolerance:
                continue
            prior_prob = float(prior_outcome_probs.get(outcome_id, 0.0))
            posterior_prob = float(posterior_outcome_probs.get(outcome_id, prior_prob))
            delta_probability = float(posterior_prob - prior_prob)
            intended_direction = "up" if modifier_value > 1.0 else "down"
            if delta_probability > tolerance:
                actual_direction = "up"
            elif delta_probability < -tolerance:
                actual_direction = "down"
            else:
                actual_direction = "flat"
            trace.append(
                {
                    "mismatch_type": "direction_mismatch" if actual_direction != intended_direction else "direction_trace",
                    "outcome_id": outcome_id,
                    "modifier": modifier_value,
                    "intended_direction": intended_direction,
                    "actual_direction": actual_direction,
                    "sign_match": actual_direction == intended_direction,
                    "prior_probability": prior_prob,
                    "posterior_probability": posterior_prob,
                    "delta_probability": delta_probability,
                }
            )
        return trace

    def _build_parameter_effect_conflict_node(
        self,
        *,
        domain_id: str,
        step: int,
        mismatches: List[Dict[str, Any]],
        rationale: str,
        signal_text: str,
    ) -> KGNode:
        """Persist simulator sign mismatches between intended and realized outcome motion."""

        strongest = max(mismatches, key=lambda entry: abs(float(entry.get("delta_probability", 0.0))))
        confidence = min(1.0, 0.35 + abs(float(strongest.get("delta_probability", 0.0))))
        if strongest.get("mismatch_type") == "unknown_outcome_id":
            target = strongest.get("corrected_to")
            if target:
                content = (
                    f"LLM emitted unknown outcome id {strongest.get('hallucinated_outcome_id')}; "
                    f"repaired to {target} before simulator update."
                )
            else:
                content = (
                    f"LLM emitted unknown outcome id {strongest.get('hallucinated_outcome_id')}; "
                    "modifier was dropped before simulator update."
                )
        else:
            content = (
                f"Parameter intent diverged from realized posterior drift for {len(mismatches)} outcomes; "
                f"strongest mismatch={strongest['outcome_id']} intended {strongest['intended_direction']} "
                f"but moved {strongest['actual_direction']}."
            )
        return KGNode(
            id=f"parameter_effect_conflict:{domain_id}:{step}",
            label=f"Parameter Effect Conflict {domain_id}",
            node_type="parameter_effect_conflict",
            content=content,
            confidence=confidence,
            metadata={
                "domain_id": domain_id,
                "step": int(step),
                "mismatches": mismatches,
                "rationale": rationale,
                "signal_excerpt": signal_text[:280],
                "mismatch_count": len(mismatches),
                "strongest_mismatch": strongest,
            },
        )


__all__ = ["AnalysisPipeline", "AnalysisPipelineConfig", "AnalysisPipelineResult"]
