"""Compile validation, backtesting, and ensemble sign consensus."""

from __future__ import annotations

from collections import Counter
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Sequence

import numpy as np

from freeman_librarian.core.evolution import DEFAULT_EVOLUTION_REGISTRY, get_operator
from freeman_librarian.core.types import Resource
from freeman_librarian.core.world import WorldState
from freeman_librarian.domain.compiler import DomainCompiler
from freeman_librarian.game.runner import GameRunner, SimConfig
from freeman_librarian.memory.knowledgegraph import KnowledgeGraph


@dataclass
class HistoricalFitScore:
    """Backtest quality summary for one compile candidate."""

    metric_name: str
    score: float
    horizon: int
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class OperatorFitReport:
    """RMSE comparison between the chosen operator and alternatives."""

    resource_id: str
    chosen_operator: str
    scores: Dict[str, float]
    best_operator: str
    gap: float
    warn: bool


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
    operator_fit_reports: List[OperatorFitReport] = field(default_factory=list)
    suggested_outcome_weights: Dict[str, List[float]] | None = None
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

    def validate(
        self,
        candidate: CompileCandidate | Dict[str, Any],
        *,
        historical_data: Dict[str, Sequence[float]] | None = None,
        knowledge_graph: KnowledgeGraph | None = None,
    ) -> CompileValidationReport:
        """Validate a single compile candidate or raw schema."""

        compile_candidate = (
            candidate
            if isinstance(candidate, CompileCandidate)
            else CompileCandidate(candidate_id="candidate_1", schema=deepcopy(candidate))
        )
        return self.validate_candidates(
            [compile_candidate],
            historical_data=historical_data,
            knowledge_graph=knowledge_graph,
        )

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

        operator_fit_reports: List[OperatorFitReport] = []
        if historical_data and best_candidate_id is not None:
            best_candidate = next(
                candidate for candidate in candidates if candidate.candidate_id == best_candidate_id
            )
            operator_fit_reports = self._operator_fit_reports_for_schema(best_candidate.schema, historical_data)

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
            operator_fit_reports=operator_fit_reports,
            metadata={
                "sign_conflicts": sign_conflicts,
                "kg_conflicts": kg_conflicts,
                "operator_warnings": [
                    report.resource_id for report in operator_fit_reports if report.warn
                ],
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

    def compare_operators(
        self,
        resource_id: str,
        historical_series: list[float],
        params: dict,
        warn_threshold: float = 0.10,
    ) -> OperatorFitReport:
        """Compare all evolution operators against one observed resource series."""

        observed = np.array([float(value) for value in historical_series], dtype=np.float64)
        if len(observed) < 2:
            raise ValueError("historical_series must contain at least two observations")

        resource_payload = deepcopy(params)
        chosen_operator = str(resource_payload.get("evolution_type", "stock_flow"))
        scores: Dict[str, float] = {}
        for operator_name in DEFAULT_EVOLUTION_REGISTRY.available():
            predicted = np.array(
                self._simulate_operator_series(resource_id, observed.tolist(), operator_name, resource_payload),
                dtype=np.float64,
            )
            scores[operator_name] = float(np.sqrt(np.mean((predicted - observed) ** 2)))

        best_score = float(min(scores.values()))
        score_scale = float(max(np.mean(np.abs(observed)), 1.0))
        tied_operators = [
            operator_name
            for operator_name, score in scores.items()
            if np.isclose(score, best_score, rtol=1.0e-6, atol=score_scale * 1.0e-6)
        ]
        best_operator = min(tied_operators, key=self._operator_priority) if tied_operators else min(
            scores,
            key=lambda operator_name: (scores[operator_name], self._operator_priority(operator_name)),
        )
        best_score = float(scores[best_operator])
        chosen_score = float(scores.get(chosen_operator, float("inf")))
        if chosen_operator == best_operator:
            gap = 0.0
        else:
            gap = float(max(chosen_score - best_score, 0.0) / max(best_score, 1.0e-12))
        return OperatorFitReport(
            resource_id=resource_id,
            chosen_operator=chosen_operator,
            scores=scores,
            best_operator=best_operator,
            gap=gap,
            warn=bool(gap > float(warn_threshold)),
        )

    def fit_outcome_weights(
        self,
        historical_series: list[dict],
        learning_rate: float = 0.01,
        max_iter: int = 500,
        l2_reg: float = 1.0e-3,
    ) -> Dict[str, List[float]]:
        """Fit outcome scoring weights by softmax cross-entropy on historical state/outcome pairs."""

        if not historical_series:
            return {}

        feature_keys = sorted(
            {
                str(feature_key)
                for item in historical_series
                for feature_key in dict(item.get("state", {})).keys()
            }
        )
        outcome_ids = sorted(
            {
                str(item.get("outcome"))
                for item in historical_series
                if str(item.get("outcome", "")).strip()
            }
        )
        if not feature_keys or not outcome_ids:
            return {}

        feature_index = {feature_key: idx for idx, feature_key in enumerate(feature_keys)}
        outcome_index = {outcome_id: idx for idx, outcome_id in enumerate(outcome_ids)}
        x_matrix = np.zeros((len(historical_series), len(feature_keys)), dtype=np.float64)
        y_vector = np.zeros(len(historical_series), dtype=np.int64)
        for row_idx, item in enumerate(historical_series):
            state = dict(item.get("state", {}))
            for feature_key, value in state.items():
                if feature_key in feature_index:
                    x_matrix[row_idx, feature_index[feature_key]] = float(value)
            y_vector[row_idx] = outcome_index[str(item["outcome"])]

        weights = np.zeros((len(outcome_ids), len(feature_keys)), dtype=np.float64)
        target = np.eye(len(outcome_ids), dtype=np.float64)[y_vector]
        sample_count = max(len(historical_series), 1)
        for _ in range(int(max_iter)):
            logits = x_matrix @ weights.T
            logits -= np.max(logits, axis=1, keepdims=True)
            probs = np.exp(logits)
            probs /= np.sum(probs, axis=1, keepdims=True)
            gradient = ((probs - target).T @ x_matrix) / sample_count + 2.0 * float(l2_reg) * weights
            weights -= float(learning_rate) * gradient

        return {
            outcome_id: [float(value) for value in weights[outcome_index[outcome_id]].tolist()]
            for outcome_id in outcome_ids
        }

    def _operator_fit_reports_for_schema(
        self,
        schema: Dict[str, Any],
        historical_data: Dict[str, Sequence[float]],
    ) -> List[OperatorFitReport]:
        """Return operator-comparison reports for schema resources with history."""

        reports: List[OperatorFitReport] = []
        for resource in schema.get("resources", []):
            resource_id = resource.get("id")
            if resource_id is None or resource_id not in historical_data:
                continue
            series = historical_data[resource_id]
            if len(series) < 2:
                continue
            reports.append(
                self.compare_operators(
                    resource_id=str(resource_id),
                    historical_series=[float(value) for value in series],
                    params=resource,
                )
            )
        return reports

    def _simulate_operator_series(
        self,
        resource_id: str,
        historical_series: list[float],
        operator_name: str,
        params: dict,
    ) -> list[float]:
        """Generate a univariate trajectory for one operator without mutating world state."""

        resource_payload = deepcopy(params)
        min_value = float(resource_payload.get("min_value", min(historical_series)))
        max_value = resource_payload.get("max_value", max(historical_series) * 1.5 + 1.0)
        if max_value == "inf":
            max_value = max(historical_series) * 1.5 + 1.0
        resource = Resource(
            id=resource_id,
            name=str(resource_payload.get("name", resource_id)),
            value=float(historical_series[0]),
            unit=str(resource_payload.get("unit", "unit")),
            owner_id=resource_payload.get("owner_id"),
            min_value=min_value,
            max_value=float(max_value),
            evolution_type=operator_name,
            evolution_params=self._fit_operator_params(operator_name, historical_series),
            conserved=bool(resource_payload.get("conserved", False)),
        )
        world = WorldState(
            domain_id="operator_fit",
            t=0,
            actors={},
            resources={resource_id: resource},
            relations=[],
            outcomes={},
            causal_dag=[],
            metadata={},
            seed=self.sim_config.seed,
        )
        operator = get_operator(operator_name, resource.evolution_params)
        predicted = [float(resource.value)]
        for _ in range(len(historical_series) - 1):
            next_value = float(operator.step(resource, world, None, 1.0))
            resource.value = np.float64(next_value)
            world.resources[resource_id].value = np.float64(next_value)
            predicted.append(float(resource.value))
        return predicted

    def _fit_operator_params(self, operator_name: str, historical_series: list[float]) -> Dict[str, Any]:
        """Estimate simple operator parameters from the observed series itself."""

        if operator_name == "linear":
            return self._fit_linear_params(historical_series)
        if operator_name == "stock_flow":
            return self._fit_stock_flow_params(historical_series)
        if operator_name == "logistic":
            return self._fit_logistic_params(historical_series)
        if operator_name == "threshold":
            return self._fit_threshold_params(historical_series)
        if operator_name == "coupled":
            return self._fit_coupled_params(historical_series)
        raise KeyError(f"Unknown operator for fitting: {operator_name}")

    def _fit_linear_params(self, historical_series: list[float]) -> Dict[str, Any]:
        """Fit ``y_(t+1) = a y_t + c`` by least squares."""

        a, c = self._fit_affine_params(historical_series)
        return {"a": a, "b": 0.0, "c": c, "coupling_weights": {}}

    def _fit_stock_flow_params(self, historical_series: list[float]) -> Dict[str, Any]:
        """Fit a stock-flow approximation from the same affine recurrence."""

        a, c = self._fit_affine_params(historical_series)
        return {
            "delta": float(1.0 - a),
            "phi_params": {"base_inflow": c, "policy_scale": 0.0, "coupling_weights": {}},
        }

    def _fit_logistic_params(self, historical_series: list[float]) -> Dict[str, Any]:
        """Fit logistic growth without external forcing from a univariate series."""

        series = np.array(historical_series, dtype=np.float64)
        current = series[:-1]
        nxt = series[1:]
        valid = current > 1.0e-8
        if np.count_nonzero(valid) < 2:
            carrying_capacity = float(max(np.max(series) * 1.05, 1.0))
            return {"r": 0.1, "K": carrying_capacity, "external": 0.0, "policy_scale": 0.0, "coupling_weights": {}}

        current = current[valid]
        nxt = nxt[valid]
        response = (nxt - current) / current
        design = np.column_stack([np.ones_like(current), current])
        coeffs, *_ = np.linalg.lstsq(design, response, rcond=None)
        r = float(max(coeffs[0], 1.0e-6))
        slope = float(coeffs[1])
        carrying_capacity = float(max(np.max(series) * 1.01, 1.0))
        if slope < -1.0e-9:
            carrying_capacity = float(max(-r / slope, carrying_capacity))
        else:
            carrying_capacity = float(max(np.max(series) * 1.05, carrying_capacity))
        return {
            "r": r,
            "K": carrying_capacity,
            "external": 0.0,
            "policy_scale": 0.0,
            "coupling_weights": {},
        }

    def _fit_threshold_params(self, historical_series: list[float]) -> Dict[str, Any]:
        """Fit separate affine branches above and below the median level."""

        series = np.array(historical_series, dtype=np.float64)
        current = series[:-1]
        threshold = float(np.median(current))
        low_mask = current < threshold
        high_mask = ~low_mask
        global_a, global_c = self._fit_affine_params(historical_series)
        low_a, low_c = self._fit_affine_subset(current, series[1:], low_mask, fallback=(global_a, global_c))
        high_a, high_c = self._fit_affine_subset(current, series[1:], high_mask, fallback=(global_a, global_c))
        return {
            "theta": threshold,
            "low_params": {"mode": "linear", "a": low_a, "b": 0.0, "c": low_c, "coupling_weights": {}},
            "high_params": {"mode": "linear", "a": high_a, "b": 0.0, "c": high_c, "coupling_weights": {}},
        }

    def _fit_coupled_params(self, historical_series: list[float]) -> Dict[str, Any]:
        """Blend linear and logistic candidates into a side-effect-free coupled proxy."""

        return {
            "components": [
                {"evolution_type": "linear", "weight": 0.5, "evolution_params": self._fit_linear_params(historical_series)},
                {"evolution_type": "logistic", "weight": 0.5, "evolution_params": self._fit_logistic_params(historical_series)},
            ]
        }

    def _fit_affine_params(self, historical_series: list[float]) -> tuple[float, float]:
        """Return least-squares ``a`` and ``c`` for ``y_(t+1) = a y_t + c``."""

        series = np.array(historical_series, dtype=np.float64)
        current = series[:-1]
        nxt = series[1:]
        if len(current) == 0:
            return 1.0, 0.0
        design = np.column_stack([current, np.ones_like(current)])
        coeffs, *_ = np.linalg.lstsq(design, nxt, rcond=None)
        return float(coeffs[0]), float(coeffs[1])

    def _fit_affine_subset(
        self,
        current: np.ndarray,
        nxt: np.ndarray,
        mask: np.ndarray,
        *,
        fallback: tuple[float, float],
    ) -> tuple[float, float]:
        """Fit affine dynamics on a masked subset or return the fallback."""

        if int(np.count_nonzero(mask)) < 2:
            return fallback
        design = np.column_stack([current[mask], np.ones(int(np.count_nonzero(mask)), dtype=np.float64)])
        coeffs, *_ = np.linalg.lstsq(design, nxt[mask], rcond=None)
        return float(coeffs[0]), float(coeffs[1])

    def _operator_priority(self, operator_name: str) -> int:
        """Prefer simpler operators when RMSE ties exactly."""

        priority = {
            "linear": 0,
            "stock_flow": 1,
            "logistic": 2,
            "threshold": 3,
            "coupled": 4,
        }
        return priority.get(operator_name, len(priority))



__all__ = [
    "CompileCandidate",
    "CompileValidationReport",
    "CompileValidator",
    "HistoricalFitScore",
    "OperatorFitReport",
]
