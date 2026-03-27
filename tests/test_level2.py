"""Tests for sign-consistency and fixed-point correction."""

from __future__ import annotations

from freeman.core.types import CausalEdge, Outcome, Resource
from freeman.core.world import WorldState
from freeman.verifier.fixed_point import find_fixed_point
from freeman.verifier.level2 import level2_check


def test_sign_violation_detected() -> None:
    world = WorldState(
        domain_id="sign_violation",
        t=0,
        actors={},
        resources={
            "x": Resource(id="x", name="X", value=10.0, unit="u", evolution_type="linear", evolution_params={"a": 0.9}),
            "y": Resource(
                id="y",
                name="Y",
                value=5.0,
                unit="u",
                evolution_type="linear",
                evolution_params={"a": 0.8, "coupling_weights": {"x": 0.5}},
            ),
        },
        relations=[],
        outcomes={
            "good": Outcome(id="good", label="Good", scoring_weights={"x": 0.1, "y": 0.1}),
            "bad": Outcome(id="bad", label="Bad", scoring_weights={"x": -0.1, "y": -0.1}),
        },
        causal_dag=[CausalEdge(source="x", target="y", expected_sign="-", strength="strong")],
        metadata={"exogenous_inflow": 5.0},
    )

    violations = level2_check(world, world.causal_dag)

    assert any(violation.check_name == "sign_consistency" for violation in violations)


def test_fixed_point_converges_on_simple_cycle() -> None:
    world = WorldState(
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
        metadata={"exogenous_inflow": 0.0},
    )

    corrected, converged, iterations = find_fixed_point(world, world.causal_dag, max_iter=10, alpha=0.1)

    assert corrected.domain_id == world.domain_id
    assert converged is True
    assert iterations >= 0
