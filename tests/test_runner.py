"""Integration tests for the game runner."""

from __future__ import annotations

import json

from freeman.core.types import CausalEdge, Outcome, Resource
from freeman.core.world import WorldState
from freeman.game.runner import GameRunner, SimConfig


def test_runner_executes_50_steps_and_serializes(water_market_world) -> None:
    runner = GameRunner(SimConfig(max_steps=50, level2_check_every=5))
    result = runner.run(water_market_world, [])

    payload = json.loads(result.to_json())

    assert result.steps_run == 50
    assert len(result.trajectory) == 51
    assert payload["domain_id"] == "water_market"
    assert payload["steps_run"] == 50


def test_runner_stops_on_hard_level2_violation() -> None:
    world = WorldState(
        domain_id="runner_level2_stop",
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

    result = GameRunner(
        SimConfig(max_steps=10, level2_check_every=1, stop_on_hard_level2=True, level2_shock_delta=0.01)
    ).run(world, [])

    assert result.steps_run == 1
    assert result.metadata["stop_reason"] == "hard_level2_violation"
