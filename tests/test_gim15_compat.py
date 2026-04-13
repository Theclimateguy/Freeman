"""Regression-style test for the bundled GIM15 profile."""

from __future__ import annotations

from freeman.domain.compiler import DomainCompiler
from freeman.game.runner import GameRunner, SimConfig


def test_gim15_profile_compiles_and_runs(gim15_schema) -> None:
    world = DomainCompiler().compile(gim15_schema)
    result = GameRunner(SimConfig(max_steps=10, level2_check_every=5)).run(world, [])

    assert world.domain_id == "gim15"
    assert result.steps_run == 10
    assert result.confidence > 0.0
    assert not any(violation.severity == "hard" for violation in result.violations)
