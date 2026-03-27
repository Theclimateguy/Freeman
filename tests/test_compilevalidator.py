"""Tests for compile validation and sign consensus."""

from __future__ import annotations

from freeman.core.compilevalidator import CompileCandidate, CompileValidator


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
