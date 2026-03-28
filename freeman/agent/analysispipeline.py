"""Compile -> simulate -> verify -> score -> update-KG pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
import logging
from pathlib import Path
from typing import Any, Dict, Iterable, List

import yaml
from freeman.agent.forecastregistry import Forecast, ForecastRegistry
from freeman.agent.proactiveemitter import ProactiveEmitter, ProactiveEvent

from freeman.core.scorer import raw_outcome_scores
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
            extra_summary_metadata={"parameter_vector": parameter_vector.snapshot()},
        )

    def _run_world(
        self,
        world: WorldState,
        *,
        policies: Iterable[Policy | Dict[str, Any]] = (),
        session_log: SessionLog | None = None,
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
        recorded_forecasts = self._record_forecasts(final_world, sim_result.final_outcome_probs, session_log)
        signal_text = self._signal_text(final_world, session_log)
        context_nodes = self._get_context_nodes(signal_text)

        summary_node = KGNode(
            id=f"analysis:{final_world.domain_id}:{final_world.t}",
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
                "steps_run": sim_result.steps_run,
                "fixed_point_iters": sim_result.metadata.get("fixed_point_iters"),
                "parameter_vector": final_world.parameter_vector.snapshot(),
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
    ) -> List[Forecast]:
        """Record probabilistic outcome forecasts for later verification."""

        if self.forecast_registry is None or not final_outcome_probs:
            return []
        recorded: List[Forecast] = []
        for outcome_id, prob in final_outcome_probs.items():
            forecast = Forecast(
                forecast_id=f"{final_world.domain_id}:{final_world.t}:{outcome_id}",
                domain_id=final_world.domain_id,
                outcome_id=outcome_id,
                predicted_prob=prob,
                session_id=session_log.session_id,
                horizon_steps=self.sim_config.max_steps,
                created_at=datetime.now(timezone.utc).replace(microsecond=0),
                created_step=int(final_world.t),
            )
            self.forecast_registry.record(forecast)
            recorded.append(forecast)
        return recorded


__all__ = ["AnalysisPipeline", "AnalysisPipelineConfig", "AnalysisPipelineResult"]
