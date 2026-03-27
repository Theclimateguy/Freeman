"""Tests for the aggregate verifier API."""

from __future__ import annotations

from freeman.core.types import CausalEdge, Outcome, Resource
from freeman.core.world import WorldState
from freeman.verifier.verifier import Verifier, VerifierConfig


def _sign_violation_world() -> WorldState:
    return WorldState(
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
    )


def test_verifier_level0_returns_report() -> None:
    verifier = Verifier()
    prev_world = WorldState(
        domain_id="level0_report",
        t=0,
        actors={},
        resources={
            "r": Resource(
                id="r",
                name="R",
                value=10.0,
                unit="u",
                conserved=True,
                evolution_type="linear",
                evolution_params={"a": 1.0, "c": 5.0},
            )
        },
        relations=[],
        outcomes={
            "good": Outcome(id="good", label="Good", scoring_weights={"r": 1.0}),
            "bad": Outcome(id="bad", label="Bad", scoring_weights={"r": -1.0}),
        },
        causal_dag=[],
        metadata={"exogenous_inflow": 0.0},
    )
    next_world = prev_world.clone()
    next_world.resources["r"].value = 20.0

    report = verifier.level0(prev_world, next_world)

    assert report.passed is False
    assert any(violation.check_name == "conservation" for violation in report.violations)


def test_verifier_level1_includes_causal_sign_precheck() -> None:
    report = Verifier().level1(_sign_violation_world())

    assert any(violation.check_name == "sign_consistency" for violation in report.violations)
    assert report.metadata["causal_edges_checked"] == 1


def test_verifier_level2_uses_override_dag_and_correction_budget() -> None:
    world = _sign_violation_world()
    override_dag = [CausalEdge(source="x", target="y", expected_sign="-", strength="strong")]

    report = Verifier(VerifierConfig(fixed_point_max_iter=10)).level2(world, causal_dag=override_dag)

    assert report.metadata["dag_source"] == "override"
    assert report.metadata["initial_sign_violations"] == 1
    assert report.metadata["correction_iterations_budget"] == 2
