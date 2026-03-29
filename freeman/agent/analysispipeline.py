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
    build_epistemic_log_node,
    compute_confidence_weighted_disagreement,
    detect_belief_conflict,
    extract_reference_outcome_probs,
)
from freeman.agent.proactiveemitter import ProactiveEmitter, ProactiveEvent

from freeman.core.scorer import raw_outcome_scores, score_outcomes
from freeman.core.types import ParameterVector, Policy
from freeman.core.world import WorldState
from freeman.domain.compiler import DomainCompiler
from freeman.game.runner import GameRunner, SimConfig
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
        self._previous_outcome_probs: Dict[str, Dict[str, float]] = {}

    def run(
        self,
        schema: Dict[str, Any] | WorldState,
        *,
        policies: Iterable[Policy | Dict[str, Any]] = (),
        session_log: SessionLog | None = None,
    ) -> AnalysisPipelineResult:
        """Run the full compile/simulate/verify/score/update workflow."""

        world = schema.clone() if isinstance(schema, WorldState) else self.compiler.compile(schema)
        return self._run_world(world, policies=policies, session_log=session_log)

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
        if prior_outcome_probs is not None:
            conflict_snapshot = detect_belief_conflict(
                prior_outcome_probs,
                sim_result.final_outcome_probs,
            )
            if conflict_snapshot is not None:
                conflict_node = build_belief_conflict_node(
                    domain_id=final_world.domain_id,
                    step=final_world.t,
                    conflict_snapshot=conflict_snapshot,
                    rationale=final_world.parameter_vector.rationale,
                    signal_text=update_signal_text or "",
                )
                self.knowledge_graph.add_node(conflict_node)
                epistemic_node_ids.append(conflict_node.id)
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
        return result

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
        epistemic_node = build_epistemic_log_node(forecast)
        self.knowledge_graph.add_node(epistemic_node)
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


__all__ = ["AnalysisPipeline", "AnalysisPipelineConfig", "AnalysisPipelineResult"]
