"""Minimal compile -> verify -> simulate -> score -> reconcile pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List

import yaml

from freeman.agent.forecastregistry import Forecast, ForecastRegistry
from freeman.core.scorer import raw_outcome_scores, score_outcomes
from freeman.core.transition import step_world
from freeman.core.types import ParameterVector, Policy
from freeman.core.world import WorldState
from freeman.domain.compiler import DomainCompiler
from freeman.exceptions import HardStopException
from freeman.game.runner import GameRunner, SimConfig
from freeman.memory.knowledgegraph import KGEdge, KGNode, KnowledgeGraph
from freeman.verifier.verifier import Verifier


def _now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


@dataclass
class AnalysisPipelineConfig:
    """Lite pipeline thresholds."""

    probability_conflict_threshold: float = 0.25
    forecast_horizon_steps: int = 50

    @classmethod
    def from_config(cls, config_path: str | Path | None = None) -> "AnalysisPipelineConfig":
        if config_path is None:
            return cls()
        config_file = Path(config_path).resolve()
        if not config_file.exists():
            return cls()
        payload = yaml.safe_load(config_file.read_text(encoding="utf-8")) or {}
        signals = payload.get("signals", {})
        return cls(
            probability_conflict_threshold=float(signals.get("conflict_threshold", 0.25)),
            forecast_horizon_steps=max(int(payload.get("sim", {}).get("max_steps", 50)), 1),
        )


@dataclass
class AnalysisPipelineResult:
    """Structured output of one lite pipeline run."""

    world: WorldState
    simulation: Dict[str, Any]
    verification: Dict[str, Any]
    raw_scores: Dict[str, float]
    dominant_outcome: str | None
    knowledge_graph_path: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)

    def snapshot(self) -> dict[str, Any]:
        return {
            "world": self.world.snapshot(),
            "simulation": self.simulation,
            "verification": self.verification,
            "raw_scores": self.raw_scores,
            "dominant_outcome": self.dominant_outcome,
            "knowledge_graph_path": self.knowledge_graph_path,
            "metadata": self.metadata,
            "warnings": list(self.warnings),
        }


class AnalysisPipeline:
    """Orchestrate the Freeman lite analysis path."""

    def __init__(
        self,
        *,
        compiler: DomainCompiler | None = None,
        sim_config: SimConfig | None = None,
        knowledge_graph: KnowledgeGraph | None = None,
        forecast_registry: ForecastRegistry | None = None,
        config: AnalysisPipelineConfig | None = None,
        config_path: str | Path | None = None,
    ) -> None:
        self.compiler = compiler or DomainCompiler()
        self.sim_config = sim_config or SimConfig()
        self.verifier = Verifier(self.sim_config)
        self.runner = GameRunner(self.sim_config)
        self.knowledge_graph = knowledge_graph or KnowledgeGraph(auto_save=False)
        self.forecast_registry = forecast_registry
        self.config = config or AnalysisPipelineConfig(
            probability_conflict_threshold=AnalysisPipelineConfig.from_config(config_path).probability_conflict_threshold,
            forecast_horizon_steps=max(int(self.sim_config.max_steps), 1),
        )

    def run(
        self,
        schema: Dict[str, Any] | WorldState,
        *,
        policies: Iterable[Policy | Dict[str, Any]] = (),
        verify_level2: bool = False,
        source_text: str | None = None,
        source_id: str | None = None,
        assumptions: Iterable[str] = (),
    ) -> AnalysisPipelineResult:
        world = schema.clone() if isinstance(schema, WorldState) else self.compiler.compile(schema)
        return self._execute(
            world,
            policies=policies,
            verify_level2=verify_level2,
            source_text=source_text,
            source_id=source_id,
            assumptions=list(assumptions),
        )

    def update(
        self,
        previous_world: WorldState,
        parameter_vector: ParameterVector,
        *,
        policies: Iterable[Policy | Dict[str, Any]] = (),
        signal_text: str | None = None,
        signal_id: str | None = None,
        verify_level2: bool = False,
    ) -> AnalysisPipelineResult:
        world = previous_world.clone()
        world.parameter_vector = parameter_vector
        return self._execute(
            world,
            policies=policies,
            verify_level2=verify_level2,
            prior_outcome_probs=score_outcomes(previous_world),
            source_text=signal_text,
            source_id=signal_id,
            assumptions=[],
        )

    def _execute(
        self,
        world: WorldState,
        *,
        policies: Iterable[Policy | Dict[str, Any]],
        verify_level2: bool,
        prior_outcome_probs: Dict[str, float] | None = None,
        source_text: str | None = None,
        source_id: str | None = None,
        assumptions: list[str] | None = None,
    ) -> AnalysisPipelineResult:
        verification = self._verify_world(world, verify_level2=verify_level2)
        policy_objects = [
            policy if isinstance(policy, Policy) else Policy.from_snapshot(policy)
            for policy in policies
        ]
        sim_result = self.runner.run(world.clone(), policy_objects)
        final_world = WorldState.from_snapshot(sim_result.trajectory[-1])
        raw_scores = raw_outcome_scores(final_world)
        final_probs = dict(sim_result.final_outcome_probs)
        dominant_outcome = max(final_probs, key=final_probs.get) if final_probs else None
        warnings = [violation["description"] for violation in verification.get("violations", [])]

        analysis_node_id = f"analysis:{final_world.domain_id}:{int(final_world.t)}"
        verified_forecasts = self._verify_due_forecasts(final_world, final_probs)
        forecast_ids = self._record_forecasts(final_world, final_probs, analysis_node_id)
        probability_shifts = self._probability_shifts(prior_outcome_probs, final_probs)
        conflict_outcomes = [
            outcome_id
            for outcome_id, delta in probability_shifts.items()
            if abs(float(delta)) >= self.config.probability_conflict_threshold
        ]

        summary_node = KGNode(
            id=analysis_node_id,
            label=f"Analysis {final_world.domain_id}",
            node_type="analysis_run",
            content=(
                f"Dominant outcome: {dominant_outcome or 'n/a'}; "
                f"confidence={sim_result.confidence:.4f}; "
                f"violations={len(sim_result.violations)}"
            ),
            confidence=max(float(sim_result.confidence), 0.15),
            metadata={
                "domain_id": final_world.domain_id,
                "dominant_outcome": dominant_outcome,
                "final_outcome_probs": final_probs,
                "raw_scores": raw_scores,
                "verification": verification,
                "parameter_vector": final_world.parameter_vector.snapshot(),
                "probability_shifts": probability_shifts,
                "conflict_outcomes": conflict_outcomes,
                "assumptions": list(assumptions or []),
                "forecast_ids": forecast_ids,
                "verified_forecast_ids": verified_forecasts,
                "source_id": source_id,
                "source_text": source_text,
                "executed_at": _now_iso(),
            },
        )
        self._upsert_node(summary_node)
        if source_text:
            self._upsert_signal_node(source_id or analysis_node_id, source_text, analysis_node_id, summary_node.confidence)
        self._export_parameter_nodes(final_world, analysis_node_id, source_id, summary_node.confidence)
        self._export_causal_edges(final_world, analysis_node_id, summary_node.confidence)
        self.knowledge_graph.save()
        if self.forecast_registry is not None:
            self.forecast_registry.save()

        result = AnalysisPipelineResult(
            world=final_world,
            simulation=sim_result.snapshot(),
            verification=verification,
            raw_scores=raw_scores,
            dominant_outcome=dominant_outcome,
            knowledge_graph_path=str(self.knowledge_graph.json_path),
            metadata={
                "steps_run": sim_result.steps_run,
                "forecast_ids": forecast_ids,
                "verified_forecast_ids": verified_forecasts,
                "probability_shifts": probability_shifts,
                "conflict_outcomes": conflict_outcomes,
                "source_id": source_id,
            },
            warnings=warnings,
        )
        return result

    def _verify_world(self, world: WorldState, *, verify_level2: bool) -> dict[str, Any]:
        baseline = world.clone()
        try:
            next_world, _l0 = step_world(baseline.clone(), [], dt=float(self.sim_config.dt))
        except HardStopException as exc:
            next_world = baseline.clone()
            report = self.verifier.run(
                baseline.clone(),
                levels=(0, 1, 2) if verify_level2 else (0, 1),
                prev_world=baseline.clone(),
                next_world=next_world,
            )
            payload = report.snapshot()
            payload["violations"] = [violation.snapshot() for violation in exc.violations] + payload["violations"]
            return payload
        report = self.verifier.run(
            baseline.clone(),
            levels=(0, 1, 2) if verify_level2 else (0, 1),
            prev_world=baseline.clone(),
            next_world=next_world,
        )
        return report.snapshot()

    def _upsert_node(self, node: KGNode) -> None:
        existing = None
        if node.id in self.knowledge_graph.graph:
            attrs = dict(self.knowledge_graph.graph.nodes[node.id])
            if not attrs or "id" not in attrs:
                self.knowledge_graph.graph.remove_node(node.id)
            else:
                existing = KGNode.from_snapshot(attrs)
        if existing is not None:
            merged_metadata = dict(existing.metadata)
            merged_metadata.update(node.metadata)
            node.metadata = merged_metadata
            node.evidence = sorted(set(existing.evidence) | set(node.evidence))
            node.sources = sorted(set(existing.sources) | set(node.sources))
            node.confidence = max(float(existing.confidence), float(node.confidence))
        self.knowledge_graph.add_node(node)

    def _upsert_signal_node(self, source_id: str, source_text: str, analysis_node_id: str, confidence: float) -> None:
        signal_node = KGNode(
            id=f"signal:{source_id}",
            label=f"Signal {source_id}",
            node_type="signal",
            content=source_text,
            confidence=max(confidence, 0.10),
            metadata={"source_id": source_id, "recorded_at": _now_iso()},
        )
        self._upsert_node(signal_node)
        self.knowledge_graph.add_edge(
            KGEdge(
                id=f"edge:signal:{source_id}:analysis:{analysis_node_id}",
                source=signal_node.id,
                target=analysis_node_id,
                relation_type="triggered",
                confidence=confidence,
            )
        )

    def _export_parameter_nodes(
        self,
        world: WorldState,
        analysis_node_id: str,
        source_id: str | None,
        confidence: float,
    ) -> None:
        signal_node_id = f"signal:{source_id}" if source_id else None
        for outcome_id, modifier in sorted(world.parameter_vector.outcome_modifiers.items()):
            node = KGNode(
                id=f"parameter:{analysis_node_id}:outcome:{outcome_id}",
                label=f"Outcome modifier {outcome_id}",
                node_type="parameter_delta",
                content=f"{outcome_id} modifier {float(modifier):+.4f}",
                confidence=max(confidence, 0.10),
                metadata={"outcome_id": outcome_id, "modifier": float(modifier)},
            )
            self._upsert_node(node)
            self.knowledge_graph.add_edge(
                KGEdge(
                    id=f"edge:{analysis_node_id}:parameter:{outcome_id}",
                    source=node.id,
                    target=analysis_node_id,
                    relation_type="influences",
                    confidence=confidence,
                )
            )
            if signal_node_id:
                self.knowledge_graph.add_edge(
                    KGEdge(
                        id=f"edge:{signal_node_id}:parameter:{outcome_id}",
                        source=signal_node_id,
                        target=node.id,
                        relation_type="causes",
                        confidence=confidence,
                    )
                )
        shock_delta = float(world.parameter_vector.shock_decay - 1.0)
        if abs(shock_delta) > 1.0e-9:
            node = KGNode(
                id=f"parameter:{analysis_node_id}:shock_decay",
                label="Shock decay delta",
                node_type="parameter_delta",
                content=f"shock_decay {shock_delta:+.4f}",
                confidence=max(confidence, 0.10),
                metadata={"shock_decay": float(world.parameter_vector.shock_decay)},
            )
            self._upsert_node(node)
            self.knowledge_graph.add_edge(
                KGEdge(
                    id=f"edge:{node.id}:analysis:{analysis_node_id}",
                    source=node.id,
                    target=analysis_node_id,
                    relation_type="influences",
                    confidence=confidence,
                )
            )

    def _export_causal_edges(self, world: WorldState, analysis_node_id: str, confidence: float) -> None:
        for edge in world.causal_dag:
            edge_node = KGNode(
                id=f"causal:{world.domain_id}:{edge.source}:{edge.target}",
                label=f"{edge.source} -> {edge.target}",
                node_type="causal_edge",
                content=f"{edge.source} -> {edge.target} ({edge.expected_sign})",
                confidence=max(confidence, 0.10),
                metadata={
                    "domain_id": world.domain_id,
                    "source": edge.source,
                    "target": edge.target,
                    "expected_sign": edge.expected_sign,
                    "strength": edge.strength,
                },
            )
            self._upsert_node(edge_node)
            self.knowledge_graph.add_edge(
                KGEdge(
                    id=f"edge:analysis:{analysis_node_id}:causal:{edge.source}:{edge.target}",
                    source=analysis_node_id,
                    target=edge_node.id,
                    relation_type="exports",
                    confidence=confidence,
                )
            )

    def _record_forecasts(
        self,
        world: WorldState,
        final_probs: Dict[str, float],
        analysis_node_id: str,
    ) -> list[str]:
        if self.forecast_registry is None or not final_probs:
            return []
        forecast_ids: list[str] = []
        for outcome_id, prob in final_probs.items():
            forecast = Forecast(
                forecast_id=f"{world.domain_id}:{int(world.t)}:{outcome_id}",
                domain_id=world.domain_id,
                outcome_id=outcome_id,
                predicted_prob=prob,
                session_id=analysis_node_id,
                horizon_steps=self.config.forecast_horizon_steps,
                created_at=datetime.now(timezone.utc).replace(microsecond=0),
                created_step=int(world.t),
                metadata={
                    "analysis_node_id": analysis_node_id,
                    "parameter_vector": world.parameter_vector.snapshot(),
                    "created_at": _now_iso(),
                },
            )
            self.forecast_registry.record(forecast)
            forecast_ids.append(forecast.forecast_id)
            self._upsert_node(
                KGNode(
                    id=f"forecast:{forecast.forecast_id}",
                    label=f"Forecast {outcome_id}",
                    node_type="forecast",
                    content=f"Predicted P({outcome_id})={float(prob):.4f}",
                    confidence=max(float(prob), 0.10),
                    metadata=forecast.snapshot(),
                )
            )
            self.knowledge_graph.add_edge(
                KGEdge(
                    id=f"edge:{analysis_node_id}:forecast:{forecast.forecast_id}",
                    source=analysis_node_id,
                    target=f"forecast:{forecast.forecast_id}",
                    relation_type="records",
                    confidence=max(float(prob), 0.10),
                )
            )
        return forecast_ids

    def _verify_due_forecasts(self, world: WorldState, current_probs: Dict[str, float]) -> list[str]:
        if self.forecast_registry is None:
            return []
        verified_ids: list[str] = []
        for forecast in self.forecast_registry.due(int(world.t)):
            actual_prob = float(current_probs.get(forecast.outcome_id, 0.0))
            verified = self.forecast_registry.verify(
                forecast.forecast_id,
                actual_prob,
                datetime.now(timezone.utc).replace(microsecond=0),
            )
            verified_ids.append(verified.forecast_id)
            verification_node = KGNode(
                id=f"forecast_verification:{verified.forecast_id}:{int(world.t)}",
                label=f"Forecast verification {verified.outcome_id}",
                node_type="forecast_verification",
                content=(
                    f"Predicted {verified.predicted_prob:.4f}, "
                    f"actual {float(verified.actual_prob or 0.0):.4f}, "
                    f"error {float(verified.error or 0.0):.4f}"
                ),
                confidence=max(1.0 - float(verified.error or 0.0), 0.10),
                metadata=verified.snapshot(),
            )
            self._upsert_node(verification_node)
            self.knowledge_graph.add_edge(
                KGEdge(
                    id=f"edge:forecast:{verified.forecast_id}:verification:{int(world.t)}",
                    source=f"forecast:{verified.forecast_id}",
                    target=verification_node.id,
                    relation_type="verified_by",
                    confidence=max(1.0 - float(verified.error or 0.0), 0.10),
                )
            )
        return verified_ids

    def _probability_shifts(
        self,
        prior_outcome_probs: Dict[str, float] | None,
        final_outcome_probs: Dict[str, float],
    ) -> Dict[str, float]:
        if prior_outcome_probs is None:
            return {}
        outcome_ids = set(prior_outcome_probs) | set(final_outcome_probs)
        return {
            outcome_id: float(final_outcome_probs.get(outcome_id, 0.0) - prior_outcome_probs.get(outcome_id, 0.0))
            for outcome_id in sorted(outcome_ids)
        }


__all__ = ["AnalysisPipeline", "AnalysisPipelineConfig", "AnalysisPipelineResult"]
