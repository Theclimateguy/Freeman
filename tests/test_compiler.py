"""Tests for schema compilation."""

from __future__ import annotations

import pytest

from freeman.core.transition import step_world
from freeman.domain.compiler import DomainCompiler
from freeman.exceptions import ValidationError


def test_water_market_schema_compiles(water_market_schema) -> None:
    world = DomainCompiler().compile(water_market_schema)

    assert world.domain_id == "water_market"
    assert set(world.resources) == {"water_stock", "agriculture_output", "conflict_level"}


def test_invalid_schema_raises_validation_error(water_market_schema) -> None:
    water_market_schema["resources"][0]["evolution_type"] = "not_registered"

    with pytest.raises(ValidationError):
        DomainCompiler().compile(water_market_schema)


def test_empty_outcomes_raise_validation_error(water_market_schema) -> None:
    water_market_schema["outcomes"] = []

    with pytest.raises(ValidationError, match="at least one outcome"):
        DomainCompiler().compile(water_market_schema)


def test_actor_update_rules_are_compiled_and_applied(water_market_schema) -> None:
    water_market_schema["actor_update_rules"] = {
        "country_a": {
            "influence": {
                "decay": 0.9,
                "base": 0.01,
                "weights": {"water_stock": 0.0001},
                "min_value": 0.0,
                "max_value": 1.0,
            }
        }
    }

    world = DomainCompiler().compile(water_market_schema)
    next_world, _ = step_world(world, [])

    assert "country_a" in world.actor_update_rules
    assert next_world.actors["country_a"].state["influence"] != world.actors["country_a"].state["influence"]
