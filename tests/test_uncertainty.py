"""Tests for Monte Carlo uncertainty propagation."""

from __future__ import annotations

from freeman.core.types import Outcome, Resource
from freeman.core.uncertainty import ParameterDistribution, UncertaintyEngine
from freeman.core.world import WorldState
from freeman.game.runner import SimConfig


def _world() -> WorldState:
    return WorldState(
        domain_id="uncertainty",
        t=0,
        actors={},
        resources={"r": Resource(id="r", name="R", value=10.0, unit="u")},
        relations=[],
        outcomes={
            "good": Outcome(id="good", label="Good", scoring_weights={"r": 1.0}),
            "bad": Outcome(id="bad", label="Bad", scoring_weights={"r": -1.0}),
        },
        causal_dag=[],
    )


def test_uncertainty_engine_returns_outcome_distribution() -> None:
    engine = UncertaintyEngine(SimConfig(max_steps=0))
    distribution = engine.monte_carlo(
        _world(),
        [ParameterDistribution(path="resources.r.value", distribution_type="normal", params={"mean": 10.0, "std": 1.0})],
        monte_carlo_samples=20,
        seed=7,
    )

    assert len(distribution.samples) == 20
    assert abs(sum(distribution.mean_probs.values()) - 1.0) < 1.0e-8
    assert set(distribution.intervals["good"]) == {"p05", "p50", "p95"}


def test_confidence_from_variance_declines_with_higher_uncertainty() -> None:
    engine = UncertaintyEngine(SimConfig(max_steps=0))
    low_var = engine.monte_carlo(
        _world(),
        [ParameterDistribution(path="resources.r.value", distribution_type="normal", params={"mean": 10.0, "std": 0.1})],
        monte_carlo_samples=30,
        seed=1,
    )
    high_var = engine.monte_carlo(
        _world(),
        [ParameterDistribution(path="resources.r.value", distribution_type="normal", params={"mean": 10.0, "std": 5.0})],
        monte_carlo_samples=30,
        seed=1,
    )

    low_report = engine.confidence_from_variance(low_var)
    high_report = engine.confidence_from_variance(high_var)

    assert 0.0 <= high_report.confidence <= 1.0
    assert low_report.confidence > high_report.confidence
