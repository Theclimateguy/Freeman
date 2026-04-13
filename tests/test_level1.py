"""Tests for level-1 structural verification."""

from __future__ import annotations

from freeman.core.types import Outcome, Resource
from freeman.core.world import WorldState
from freeman.game.runner import SimConfig
from freeman.verifier.level1 import level1_check


def test_null_action_convergence_passes_on_stable_domain(water_market_world) -> None:
    config = SimConfig(convergence_check_steps=250, convergence_epsilon=2.0e-2)
    violations = level1_check(water_market_world, config)

    assert not any(violation.check_name == "null_action_convergence" for violation in violations)
    assert not any(violation.check_name == "spectral_radius" for violation in violations)


def test_spectral_radius_violation_detected() -> None:
    world = WorldState(
        domain_id="unstable",
        t=0,
        actors={},
        resources={
            "r": Resource(
                id="r",
                name="Unstable",
                value=100.0,
                unit="u",
                evolution_type="linear",
                evolution_params={"a": 1.05, "c": 0.0},
            )
        },
        relations=[],
        outcomes={
            "stable": Outcome(id="stable", label="Stable", scoring_weights={"r": 1.0}),
            "crisis": Outcome(id="crisis", label="Crisis", scoring_weights={"r": -1.0}),
        },
        causal_dag=[],
        metadata={"exogenous_inflow": 10.0},
    )

    violations = level1_check(world, SimConfig(convergence_check_steps=10))

    assert any(violation.check_name == "spectral_radius" for violation in violations)
