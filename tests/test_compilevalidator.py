"""Tests for compile validation and sign consensus."""

from __future__ import annotations

import copy
import math

from freeman.core.compilevalidator import CompileCandidate, CompileValidator
from freeman.core.world import WorldState
from scripts.benchmark_faab.metrics import brier_score


def _single_resource_schema(a: float) -> dict:
    return {
        "domain_id": f"candidate_{a}",
        "actors": [],
        "resources": [
            {
                "id": "stock",
                "name": "Stock",
                "value": 10.0,
                "unit": "u",
                "evolution_type": "linear",
                "evolution_params": {"a": a, "c": 0.0},
            }
        ],
        "relations": [],
        "outcomes": [
            {"id": "good", "label": "Good", "scoring_weights": {"stock": 1.0}},
            {"id": "bad", "label": "Bad", "scoring_weights": {"stock": -1.0}},
        ],
        "causal_dag": [],
    }


def _linear_series(start: float, step: float, length: int) -> list[float]:
    return [start + step * index for index in range(length)]


def _logistic_series(length: int, *, initial: float = 5.0, r: float = 0.45, carrying_capacity: float = 100.0) -> list[float]:
    values = [initial]
    current = initial
    for _ in range(length - 1):
        current = current + r * current * (1.0 - current / carrying_capacity)
        values.append(current)
    return values


def _softmax_probability(weight_map: dict[str, list[float]], state: dict[str, float]) -> dict[str, float]:
    feature_keys = sorted(state)
    logits = {}
    for outcome_id, weights in weight_map.items():
        logits[outcome_id] = sum(float(weight) * float(state[key]) for weight, key in zip(weights, feature_keys, strict=False))
    max_logit = max(logits.values())
    exps = {outcome_id: math.exp(logit - max_logit) for outcome_id, logit in logits.items()}
    total = sum(exps.values())
    return {outcome_id: value / total for outcome_id, value in exps.items()}


def test_compilevalidator_selects_best_candidate_by_historical_fit() -> None:
    validator = CompileValidator(backtest_horizon=3, historical_fit_threshold=0.6)
    candidates = [
        CompileCandidate(candidate_id="stable", schema=_single_resource_schema(1.0)),
        CompileCandidate(candidate_id="decay", schema=_single_resource_schema(0.5)),
    ]

    report = validator.validate_candidates(
        candidates,
        historical_data={"stock": [10.0, 10.0, 10.0, 10.0]},
    )

    assert report.best_candidate_id == "stable"
    assert report.fit_scores["stable"].score > report.fit_scores["decay"].score
    assert report.passed is True


def test_compilevalidator_marks_review_required_on_sign_conflict() -> None:
    validator = CompileValidator(sign_conflict_action="review")
    schema_plus = _single_resource_schema(1.0)
    schema_plus["causal_dag"] = [{"source": "stock", "target": "stock", "expected_sign": "+", "strength": "weak"}]
    schema_minus = _single_resource_schema(1.0)
    schema_minus["causal_dag"] = [{"source": "stock", "target": "stock", "expected_sign": "-", "strength": "weak"}]

    report = validator.validate_candidates(
        [
            CompileCandidate(candidate_id="plus", schema=schema_plus),
            CompileCandidate(candidate_id="minus", schema=schema_minus),
        ]
    )

    assert report.review_required is True
    assert report.reviewRequired is True
    assert report.sign_consensus["stock->stock"] == "conflict"


def test_logistic_beats_linear_on_s_curve() -> None:
    validator = CompileValidator()
    historical = _logistic_series(16)
    resource = {
        "id": "population",
        "name": "Population",
        "value": historical[0],
        "unit": "idx",
        "evolution_type": "linear",
        "evolution_params": {"a": 1.0, "c": 0.0},
    }

    report = validator.compare_operators("population", historical, resource)

    assert report.scores["logistic"] < report.scores["linear"]
    assert report.best_operator == "logistic"


def test_linear_beats_logistic_on_linear_trend() -> None:
    validator = CompileValidator()
    historical = _linear_series(10.0, 7.5, 12)
    resource = {
        "id": "inventory",
        "name": "Inventory",
        "value": historical[0],
        "unit": "idx",
        "evolution_type": "linear",
        "evolution_params": {"a": 1.0, "c": 0.0},
    }

    report = validator.compare_operators("inventory", historical, resource)

    assert report.scores["linear"] <= report.scores["logistic"]
    assert report.best_operator == "linear"


def test_warn_flag_raised() -> None:
    validator = CompileValidator()
    historical = _logistic_series(16)
    resource = {
        "id": "adoption",
        "name": "Adoption",
        "value": historical[0],
        "unit": "idx",
        "evolution_type": "linear",
        "evolution_params": {"a": 1.0, "c": 0.0},
    }

    report = validator.compare_operators("adoption", historical, resource, warn_threshold=0.05)

    assert report.warn is True
    assert report.gap > 0.05


def test_no_historical_data_no_report() -> None:
    validator = CompileValidator()
    schema = _single_resource_schema(1.0)

    report = validator.validate(schema)

    assert report.operator_fit_reports == []


def test_compare_operators_no_mutation() -> None:
    validator = CompileValidator()
    historical = _logistic_series(10)
    schema = _single_resource_schema(1.0)
    world = WorldState.from_snapshot(CompileValidator().compiler.compile(schema).snapshot())
    before_world = world.snapshot()
    resource = copy.deepcopy(schema["resources"][0])
    before_resource = copy.deepcopy(resource)

    validator.compare_operators("stock", historical, resource)

    assert resource == before_resource
    assert world.snapshot() == before_world


def test_fit_outcome_weights_improves_brier() -> None:
    validator = CompileValidator()
    history = [
        {"state": {"signal": -3.0, "trend": -1.0}, "outcome": "down"},
        {"state": {"signal": -2.0, "trend": -0.5}, "outcome": "down"},
        {"state": {"signal": -1.0, "trend": -0.2}, "outcome": "down"},
        {"state": {"signal": 1.0, "trend": 0.2}, "outcome": "up"},
        {"state": {"signal": 2.0, "trend": 0.5}, "outcome": "up"},
        {"state": {"signal": 3.0, "trend": 1.0}, "outcome": "up"},
    ]

    weights = validator.fit_outcome_weights(history, learning_rate=0.05, max_iter=800, l2_reg=1.0e-4)
    train_brier = 0.0
    random_brier = 0.0
    for item in history:
        prediction = _softmax_probability(weights, item["state"])
        train_brier += brier_score(prediction, item["outcome"])
        random_brier += brier_score({"up": 0.5, "down": 0.5}, item["outcome"])
    train_brier /= len(history)
    random_brier /= len(history)

    assert set(weights) == {"down", "up"}
    assert train_brier <= random_brier
