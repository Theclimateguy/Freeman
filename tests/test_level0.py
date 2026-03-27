"""Tests for level-0 invariants."""

from __future__ import annotations

import pytest

from freeman.core.transition import step_world
from freeman.core.types import Outcome, Resource
from freeman.core.world import WorldState
from freeman.exceptions import HardStopException
from freeman.verifier.level0 import level0_check


def _base_world(resource: Resource) -> WorldState:
    return WorldState(
        domain_id="level0",
        t=0,
        actors={},
        resources={"r": resource},
        relations=[],
        outcomes={
            "good": Outcome(id="good", label="Good", scoring_weights={"r": 1.0}),
            "bad": Outcome(id="bad", label="Bad", scoring_weights={"r": -1.0}),
        },
        causal_dag=[],
        metadata={"exogenous_inflow": 0.0},
    )


def test_conservation_violation_raises_hard_stop() -> None:
    world = _base_world(
        Resource(
            id="r",
            name="R",
            value=10.0,
            unit="u",
            conserved=True,
            evolution_type="linear",
            evolution_params={"a": 1.0, "c": 5.0},
        )
    )

    with pytest.raises(HardStopException) as exc_info:
        step_world(world, [])

    assert any(violation.check_name == "conservation" for violation in exc_info.value.violations)


def test_conservation_ignores_nonconserved_resources() -> None:
    prev = WorldState(
        domain_id="units",
        t=0,
        actors={},
        resources={
            "water": Resource(id="water", name="Water", value=100.0, unit="km3", conserved=True),
            "gdp": Resource(id="gdp", name="GDP", value=10.0, unit="usd", conserved=False),
        },
        relations=[],
        outcomes={
            "good": Outcome(id="good", label="Good", scoring_weights={"gdp": 1.0}),
            "bad": Outcome(id="bad", label="Bad", scoring_weights={"gdp": -1.0}),
        },
        causal_dag=[],
        metadata={"exogenous_inflows": {"water": 0.0}},
    )
    next_world = prev.clone()
    next_world.resources["gdp"].value = 1000.0

    violations = level0_check(prev, next_world)

    assert not any(violation.check_name == "conservation" for violation in violations)


def test_nonnegativity_violation_raises_hard_stop() -> None:
    world = _base_world(
        Resource(
            id="r",
            name="R",
            value=10.0,
            unit="u",
            min_value=0.0,
            evolution_type="linear",
            evolution_params={"a": 0.5, "c": -20.0},
        )
    )

    with pytest.raises(HardStopException) as exc_info:
        step_world(world, [])

    assert any(violation.check_name == "nonnegativity" for violation in exc_info.value.violations)


def test_probability_simplex_violation_detected() -> None:
    prev = _base_world(Resource(id="r", name="R", value=1.0, unit="u"))
    next_world = prev.clone()
    next_world.outcomes = {}

    violations = level0_check(prev, next_world)

    assert any(violation.check_name == "probability_simplex" for violation in violations)
