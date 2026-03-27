"""Compile -> simulate -> verify -> score -> update-KG pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List

from freeman.core.scorer import raw_outcome_scores
from freeman.core.types import Policy
from freeman.core.world import WorldState
from freeman.domain.compiler import DomainCompiler
from freeman.game.runner import GameRunner, SimConfig
from freeman.memory.knowledgegraph import KGNode, KnowledgeGraph
from freeman.memory.reconciler import Reconciler, ReconciliationResult
from freeman.memory.sessionlog import KGDelta, SessionLog
from freeman.verifier.verifier import Verifier, VerifierConfig


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
    ) -> None:
        self.compiler = compiler or DomainCompiler()
        self.sim_config = sim_config or SimConfig()
        self.verifier = Verifier(verifier_config or self.sim_config)
        self.runner = GameRunner(self.sim_config)
        self.knowledge_graph = knowledge_graph or KnowledgeGraph()
        self.reconciler = reconciler or Reconciler()

    def run(
        self,
        schema: Dict[str, Any] | WorldState,
        *,
        policies: Iterable[Policy | Dict[str, Any]] = (),
        session_log: SessionLog | None = None,
    ) -> AnalysisPipelineResult:
        """Run the full compile/simulate/verify/score/update workflow."""

        world = schema.clone() if isinstance(schema, WorldState) else self.compiler.compile(schema)
        verification = self.verifier.run(world, levels=(1, 2)).snapshot()
        policy_objects = [
            policy if isinstance(policy, Policy) else Policy.from_snapshot(policy)
            for policy in policies
        ]
        sim_result = self.runner.run(world.clone(), policy_objects)
        final_world = WorldState.from_snapshot(sim_result.trajectory[-1])
        raw_scores = raw_outcome_scores(final_world)
        dominant_outcome = None
        if sim_result.final_outcome_probs:
            dominant_outcome = max(sim_result.final_outcome_probs, key=sim_result.final_outcome_probs.get)

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
                "verification": verification,
            },
        )
        self.knowledge_graph.add_node(summary_node)

        if session_log is None:
            session_log = SessionLog(session_id=f"analysis:{final_world.domain_id}:{final_world.t}")
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

        return AnalysisPipelineResult(
            world=final_world,
            simulation=sim_result.snapshot(),
            verification=verification,
            raw_scores=raw_scores,
            dominant_outcome=dominant_outcome,
            knowledge_graph_path=str(self.knowledge_graph.json_path),
            reconciliation=reconciliation,
            metadata={
                "steps_run": sim_result.steps_run,
                "fixed_point_iters": sim_result.metadata.get("fixed_point_iters"),
            },
        )


__all__ = ["AnalysisPipeline", "AnalysisPipelineResult"]
