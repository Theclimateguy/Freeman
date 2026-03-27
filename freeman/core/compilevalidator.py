"""Compile validation, backtesting, and ensemble sign consensus."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Sequence

import numpy as np

from freeman.domain.compiler import DomainCompiler
from freeman.game.runner import GameRunner, SimConfig
from freeman.memory.knowledgegraph import KnowledgeGraph


@dataclass
class HistoricalFitScore:
    """Backtest quality summary for one compile candidate."""

    metric_name: str
    score: float
    horizon: int
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CompileCandidate:
    """One machine-generated domain compile candidate."""

    candidate_id: str
    schema: Dict[str, Any]
    policies: List[Dict[str, Any]] = field(default_factory=list)
    assumptions: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    review_required: bool = False

    @property
    def reviewRequired(self) -> bool:  # noqa: N802
        return self.review_required


@dataclass
class CompileValidationReport:
    """Aggregate compile-validation result."""

    best_candidate_id: str | None
    candidates: List[CompileCandidate]
    fit_scores: Dict[str, HistoricalFitScore]
    sign_consensus: Dict[str, str]
    passed: bool
    review_required: bool
    rejected_candidate_ids: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def reviewRequired(self) -> bool:  # noqa: N802
        return self.review_required


class CompileValidator:
    """Validate compiled domains by historical fit and ensemble sign consensus."""

    def __init__(
        self,
        *,
        compiler: DomainCompiler | None = None,
        sim_config: SimConfig | None = None,
        backtest_horizon: int = 5,
        historical_fit_threshold: float = 0.5,
        sign_conflict_action: str = "review",
    ) -> None:
        self.compiler = compiler or DomainCompiler()
        self.sim_config = sim_config or SimConfig(max_steps=5)
        self.backtest_horizon = int(backtest_horizon)
        self.historical_fit_threshold = float(historical_fit_threshold)
        self.sign_conflict_action = sign_conflict_action

    def backtest(
        self,
        candidate: CompileCandidate,
        historical_data: Dict[str, Sequence[float]],
        *,
        horizon: int | None = None,
    ) -> HistoricalFitScore:
        """Compare simulated trajectories to observed historical paths."""

        horizon = min(int(horizon or self.backtest_horizon), *(len(series) - 1 for series in historical_data.values()))
        world = self.compiler.compile(candidate.schema)
        simulation = GameRunner(SimConfig(max_steps=horizon, seed=self.sim_config.seed)).run(world, [])

        resource_scores: Dict[str, float] = {}
        for resource_id, observed in historical_data.items():
            predicted = [snapshot["resources"][resource_id]["value"] for snapshot in simulation.trajectory[: horizon + 1]]
            observed_array = np.array(observed[: horizon + 1], dtype=np.float64)
            predicted_array = np.array(predicted, dtype=np.float64)
            rmse = float(np.sqrt(np.mean((predicted_array - observed_array) ** 2)))
            scale = float(max(np.mean(np.abs(observed_array)), 1.0))
            resource_scores[resource_id] = float(1.0 / (1.0 + rmse / scale))

        score = float(np.mean(list(resource_scores.values()), dtype=np.float64)) if resource_scores else 0.0
        return HistoricalFitScore(
            metric_name="normalized_rmse_fit",
            score=score,
            horizon=horizon,
            details={"resource_scores": resource_scores},
        )

    def sign_voting_consensus(self, candidates: Iterable[CompileCandidate]) -> Dict[str, str]:
        """Vote on edge signs across the ensemble."""

        votes: Dict[str, Counter[str]] = {}
        for candidate in candidates:
            for edge in candidate.schema.get("causal_dag", []):
                edge_id = f"{edge['source']}->{edge['target']}"
                votes.setdefault(edge_id, Counter())
                votes[edge_id][edge["expected_sign"]] += 1

        consensus: Dict[str, str] = {}
        for edge_id, edge_votes in votes.items():
            if len(edge_votes) > 1 and len(set(edge_votes.values())) == 1:
                consensus[edge_id] = "conflict"
                continue
            consensus[edge_id] = edge_votes.most_common(1)[0][0]
        return consensus

    def validate_candidates(
        self,
        candidates: List[CompileCandidate],
        *,
        historical_data: Dict[str, Sequence[float]] | None = None,
        knowledge_graph: KnowledgeGraph | None = None,
    ) -> CompileValidationReport:
        """Validate an ensemble of compile candidates."""

        fit_scores: Dict[str, HistoricalFitScore] = {}
        rejected: List[str] = []
        for candidate in candidates:
            if historical_data:
                fit_scores[candidate.candidate_id] = self.backtest(candidate, historical_data)
            else:
                fit_scores[candidate.candidate_id] = HistoricalFitScore("not_available", 1.0, 0, {})
            if fit_scores[candidate.candidate_id].score < self.historical_fit_threshold:
                rejected.append(candidate.candidate_id)

        sign_consensus = self.sign_voting_consensus(candidates)
        sign_conflicts = [edge_id for edge_id, sign in sign_consensus.items() if sign == "conflict"]
        review_required = bool(sign_conflicts)

        kg_conflicts: List[str] = []
        if knowledge_graph is not None:
            for node in knowledge_graph.query(node_type="causal_edge", status="active"):
                edge_id = str(node.metadata.get("edge_id", ""))
                expected_sign = str(node.metadata.get("expected_sign", ""))
                if edge_id in sign_consensus and sign_consensus[edge_id] not in {expected_sign, "conflict"}:
                    kg_conflicts.append(edge_id)
            if kg_conflicts:
                review_required = True

        passed_candidates = [candidate for candidate in candidates if candidate.candidate_id not in rejected]
        best_candidate_id = None
        if passed_candidates:
            best_candidate_id = max(
                passed_candidates,
                key=lambda item: fit_scores[item.candidate_id].score,
            ).candidate_id
        elif candidates:
            best_candidate_id = max(
                candidates,
                key=lambda item: fit_scores[item.candidate_id].score,
            ).candidate_id

        passed = best_candidate_id is not None
        if review_required and self.sign_conflict_action == "block":
            passed = False

        return CompileValidationReport(
            best_candidate_id=best_candidate_id,
            candidates=candidates,
            fit_scores=fit_scores,
            sign_consensus=sign_consensus,
            passed=passed,
            review_required=review_required,
            rejected_candidate_ids=rejected,
            metadata={
                "sign_conflicts": sign_conflicts,
                "kg_conflicts": kg_conflicts,
                "sign_conflict_action": self.sign_conflict_action,
            },
        )

    def ensemble_compile(
        self,
        compile_fn: Callable[[int], Dict[str, Any] | CompileCandidate],
        *,
        ensemble_size: int = 3,
        historical_data: Dict[str, Sequence[float]] | None = None,
        knowledge_graph: KnowledgeGraph | None = None,
    ) -> CompileValidationReport:
        """Call the supplied compiler function 2-3 times and validate the ensemble."""

        candidates: List[CompileCandidate] = []
        for index in range(int(ensemble_size)):
            raw_candidate = compile_fn(index)
            if isinstance(raw_candidate, CompileCandidate):
                candidate = raw_candidate
            else:
                candidate = CompileCandidate(
                    candidate_id=f"candidate_{index + 1}",
                    schema=raw_candidate["schema"],
                    policies=raw_candidate.get("policies", []),
                    assumptions=raw_candidate.get("assumptions", []),
                    metadata=raw_candidate.get("metadata", {}),
                )
            candidates.append(candidate)
        return self.validate_candidates(
            candidates,
            historical_data=historical_data,
            knowledge_graph=knowledge_graph,
        )


__all__ = [
    "CompileCandidate",
    "CompileValidationReport",
    "CompileValidator",
    "HistoricalFitScore",
]
