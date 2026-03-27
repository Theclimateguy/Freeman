"""Phase 1 core API compatibility tests."""

from __future__ import annotations

import math

from freeman.core.evolution import EvolutionRegistry
from freeman.core.scorer import raw_outcome_scores, score_outcomes
from freeman.core.types import Outcome, Resource
from freeman.core.world import OutcomeRegistry, WorldGraph, WorldState


def _base_world() -> WorldGraph:
    return WorldGraph(
        domain_id="phase1",
        t=0,
        actors={},
        resources={
            "x": Resource(id="x", name="X", value=2.0, unit="u"),
            "y": Resource(id="y", name="Y", value=1.0, unit="u"),
        },
        relations=[],
        outcomes={
            "good": Outcome(id="good", label="Good", scoring_weights={"x": 1.0, "y": -0.5}),
            "bad": Outcome(id="bad", label="Bad", scoring_weights={"x": -1.0, "y": 0.5}),
        },
        causal_dag=[],
    )


def test_worldgraph_exposes_live_outcome_registry() -> None:
    world = _base_world()

    assert WorldState is WorldGraph
    assert isinstance(world.outcome_registry, OutcomeRegistry)
    assert world.outcome_registry.get("good") is world.outcomes["good"]

    world.add_outcome(Outcome(id="neutral", label="Neutral", scoring_weights={"x": 0.0}))
    restored = WorldState.from_snapshot(world.snapshot())

    assert "neutral" in world.outcomes
    assert "neutral" in restored.outcomes


def test_evolution_registry_supports_spec_operator_set() -> None:
    registry = EvolutionRegistry()
    operator = registry.create("linear", {"a": 0.8, "b": 0.0, "c": 1.0})
    resource = Resource(id="stock", name="Stock", value=10.0, unit="u")

    assert set(registry.available()) == {"coupled", "linear", "logistic", "stock_flow", "threshold"}
    assert math.isclose(
        operator.step(resource, _base_world(), None),
        float(resource.value) + operator.delta(resource, _base_world(), None),
        rel_tol=0.0,
        abs_tol=1.0e-9,
    )


def test_softmax_scoring_matches_weighted_state() -> None:
    world = _base_world()

    raw_scores = raw_outcome_scores(world)
    probs = score_outcomes(world)

    assert raw_scores["good"] > raw_scores["bad"]
    assert probs["good"] > probs["bad"]
    assert math.isclose(sum(probs.values()), 1.0, rel_tol=0.0, abs_tol=1.0e-9)
