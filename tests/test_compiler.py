"""Tests for schema compilation."""

from __future__ import annotations

import pytest

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
