"""Tests for the Phase 2 fixed-point API."""

from __future__ import annotations

from freeman.core.types import CausalEdge, Outcome, Resource
from freeman.core.world import WorldState
from freeman.verifier.fixedpoint import iterate_fixed_point


def _cycle_world() -> WorldState:
    return WorldState(
        domain_id="cycle",
        t=0,
        actors={},
        resources={
            "x": Resource(
                id="x",
                name="X",
                value=10.0,
                unit="u",
                evolution_type="linear",
                evolution_params={"a": 0.8, "coupling_weights": {"y": -0.05}},
            ),
            "y": Resource(
                id="y",
                name="Y",
                value=5.0,
                unit="u",
                evolution_type="linear",
                evolution_params={"a": 0.7, "coupling_weights": {"x": 0.1}},
            ),
        },
        relations=[],
        outcomes={
            "good": Outcome(id="good", label="Good", scoring_weights={"x": 0.1, "y": 0.1}),
            "bad": Outcome(id="bad", label="Bad", scoring_weights={"x": -0.1, "y": -0.1}),
        },
        causal_dag=[
            CausalEdge(source="x", target="y", expected_sign="+", strength="strong"),
            CausalEdge(source="y", target="x", expected_sign="-", strength="strong"),
        ],
    )


def test_iterate_fixed_point_returns_history_and_converges() -> None:
    world = _cycle_world()

    result = iterate_fixed_point(world, world.causal_dag, max_iter=10, alpha=0.1)

    assert result.converged is True
    assert result.iterations >= 0
    assert result.history
    assert result.history[-1]["spectral_radius"] < 1.0 + 1.0e-6


def test_iterate_fixed_point_stops_on_spectral_radius_guard() -> None:
    world = WorldState(
        domain_id="unstable_fp",
        t=0,
        actors={},
        resources={
            "r": Resource(
                id="r",
                name="R",
                value=100.0,
                unit="u",
                evolution_type="linear",
                evolution_params={"a": 1.05},
            )
        },
        relations=[],
        outcomes={
            "good": Outcome(id="good", label="Good", scoring_weights={"r": 1.0}),
            "bad": Outcome(id="bad", label="Bad", scoring_weights={"r": -1.0}),
        },
        causal_dag=[],
    )

    result = iterate_fixed_point(world, world.causal_dag, max_iter=5)

    assert result.converged is False
    assert any(violation.check_name == "spectral_radius_guard" for violation in result.violations)
