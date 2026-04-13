"""Phase 1 core API compatibility tests."""

from __future__ import annotations

import math

from freeman.core.evolution import EvolutionRegistry
from freeman.core.scorer import raw_outcome_scores, regime_shift_matches, score_outcomes
from freeman.core.types import Outcome, ParameterVector, Resource
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
    world.parameter_vector = ParameterVector(
        outcome_modifiers={"good": 1.5},
        shock_decay=0.7,
        edge_weight_deltas={"x.y": 0.2},
        rationale="test vector",
    )

    assert WorldState is WorldGraph
    assert isinstance(world.outcome_registry, OutcomeRegistry)
    assert world.outcome_registry.get("good") is world.outcomes["good"]

    world.add_outcome(Outcome(id="neutral", label="Neutral", scoring_weights={"x": 0.0}))
    restored = WorldState.from_snapshot(world.snapshot())

    assert "neutral" in world.outcomes
    assert "neutral" in restored.outcomes
    assert restored.parameter_vector.snapshot() == world.parameter_vector.snapshot()


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
    base_scores = raw_outcome_scores(world)
    world.parameter_vector = ParameterVector(outcome_modifiers={"good": 2.0})

    raw_scores = raw_outcome_scores(world)
    probs = score_outcomes(world)

    assert raw_scores["good"] > base_scores["good"]
    assert raw_scores["good"] > raw_scores["bad"]
    assert probs["good"] > probs["bad"]
    assert math.isclose(sum(probs.values()), 1.0, rel_tol=0.0, abs_tol=1.0e-9)


def test_worldgraph_apply_shocks_decays_previous_deviation() -> None:
    world = WorldGraph(
        domain_id="decay",
        t=0,
        actors={},
        resources={
            "x": Resource(id="x", name="X", value=10.0, unit="u"),
        },
        relations=[],
        outcomes={},
        causal_dag=[],
    )

    first = world.apply_shocks({"x": 10.0})
    second = first.apply_shocks({"x": 4.0}, time_decay=0.5)

    assert math.isclose(float(first.resources["x"].value), 20.0, rel_tol=0.0, abs_tol=1.0e-9)
    assert math.isclose(float(second.resources["x"].value), 19.0, rel_tol=0.0, abs_tol=1.0e-9)
    assert math.isclose(float(first.metadata["_shock_state"]["resources"]["x"]), 10.0, rel_tol=0.0, abs_tol=1.0e-9)
    assert math.isclose(float(second.metadata["_shock_state"]["resources"]["x"]), 9.0, rel_tol=0.0, abs_tol=1.0e-9)

    first.parameter_vector = ParameterVector(shock_decay=0.5)
    third = first.apply_shocks({"x": 4.0})
    assert math.isclose(float(third.resources["x"].value), 19.0, rel_tol=0.0, abs_tol=1.0e-9)
    assert math.isclose(float(third.metadata["_shock_state"]["resources"]["x"]), 9.0, rel_tol=0.0, abs_tol=1.0e-9)


def test_regime_shift_multiplier_uses_decayed_state_context() -> None:
    world = WorldGraph(
        domain_id="regime",
        t=0,
        actors={},
        resources={
            "business_demand": Resource(id="business_demand", name="Demand", value=10.0, unit="u"),
            "policy_rate": Resource(id="policy_rate", name="Policy", value=0.0, unit="u"),
        },
        relations=[],
        outcomes={
            "stable": Outcome(id="stable", label="Stable", scoring_weights={"policy_rate": 1.0}),
            "recession": Outcome(
                id="recession",
                label="Recession",
                scoring_weights={"business_demand": 1.0},
                regime_shifts=[{"condition": "business_demand <= -5 AND policyrate >= 5", "multiplier": 3.0}],
            ),
        },
        causal_dag=[],
    ).apply_shocks({"business_demand": -7.0, "policy_rate": 5.0})
    world.parameter_vector = ParameterVector(outcome_modifiers={"recession": 2.0})

    raw_scores = raw_outcome_scores(world)
    probs = score_outcomes(world)

    assert math.isclose(raw_scores["stable"], 5.0, rel_tol=0.0, abs_tol=1.0e-9)
    assert math.isclose(raw_scores["recession"], 18.0, rel_tol=0.0, abs_tol=1.0e-9)
    assert probs["recession"] > probs["stable"]


def test_regime_shift_uses_accumulated_shock_state_after_level_recovery() -> None:
    world = WorldGraph(
        domain_id="film_regime",
        t=0,
        actors={},
        resources={
            "critic_sentiment": Resource(id="critic_sentiment", name="Critics", value=50.0, unit="u"),
            "box_office_legs": Resource(id="box_office_legs", name="Legs", value=30.0, unit="u"),
        },
        relations=[],
        outcomes={},
        causal_dag=[],
    ).apply_shocks({"critic_sentiment": -6.0, "box_office_legs": -8.0})

    world.resources["critic_sentiment"].value = 48.0
    world.resources["box_office_legs"].value = 32.0

    assert regime_shift_matches(world, "criticsentiment <= -5 AND boxofficelegs <= -5")
