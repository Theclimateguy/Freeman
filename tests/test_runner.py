"""Integration tests for the game runner."""

from __future__ import annotations

import json

from freeman.game.runner import GameRunner, SimConfig


def test_runner_executes_50_steps_and_serializes(water_market_world) -> None:
    runner = GameRunner(SimConfig(max_steps=50, level2_check_every=5))
    result = runner.run(water_market_world, [])

    payload = json.loads(result.to_json())

    assert result.steps_run == 50
    assert len(result.trajectory) == 51
    assert payload["domain_id"] == "water_market"
    assert payload["steps_run"] == 50
