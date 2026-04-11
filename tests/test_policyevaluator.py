"""Tests for counterfactual policy evaluation."""

from __future__ import annotations

import pytest

from freeman.agent import AnalysisPipeline, PolicyEvaluator
from freeman.core.types import Policy
from freeman.domain.compiler import DomainCompiler
from freeman.game.runner import SimConfig
from freeman.memory.epistemiclog import EpistemicLog
from freeman.memory.knowledgegraph import KGNode, KnowledgeGraph


def _policy_lab_schema() -> dict:
    return {
        "domain_id": "policy_lab",
        "actors": [
            {"id": "planner", "name": "Planner", "state": {"credibility": 0.6}},
        ],
        "resources": [
            {
                "id": "signal",
                "name": "Signal",
                "value": 5.0,
                "unit": "idx",
                "owner_id": "planner",
                "min_value": 0.0,
                "max_value": 50.0,
                "evolution_type": "linear",
                "evolution_params": {"a": 1.0, "b": 2.0, "c": 0.0},
            },
        ],
        "relations": [],
        "outcomes": [
            {
                "id": "good",
                "label": "Good",
                "scoring_weights": {"signal": 1.0},
            },
            {
                "id": "bad",
                "label": "Bad",
                "scoring_weights": {"signal": -1.0},
            },
        ],
        "causal_dag": [],
        "metadata": {},
    }


def _conserved_policy_schema() -> dict:
    return {
        "domain_id": "conserved_policy",
        "actors": [
            {"id": "planner", "name": "Planner", "state": {"credibility": 0.6}},
        ],
        "resources": [
            {
                "id": "reservoir",
                "name": "Reservoir",
                "value": 10.0,
                "unit": "idx",
                "owner_id": "planner",
                "min_value": 0.0,
                "max_value": 100.0,
                "conserved": True,
                "evolution_type": "stock_flow",
                "evolution_params": {
                    "delta": 0.0,
                    "phi_params": {"base_inflow": 0.0, "policy_scale": 15.0},
                },
            }
        ],
        "relations": [],
        "outcomes": [
            {"id": "stable", "label": "Stable", "scoring_weights": {"reservoir": 1.0}},
            {"id": "overflow", "label": "Overflow", "scoring_weights": {"reservoir": -1.0}},
        ],
        "causal_dag": [],
        "metadata": {"exogenous_inflows": {"reservoir": 0.0}},
    }


def _constant_world_schema() -> dict:
    return {
        "domain_id": "constant_world",
        "actors": [
            {"id": "planner", "name": "Planner", "state": {"credibility": 0.5}},
        ],
        "resources": [
            {
                "id": "signal",
                "name": "Signal",
                "value": 5.0,
                "unit": "idx",
                "owner_id": "planner",
                "min_value": 0.0,
                "max_value": 10.0,
                "evolution_type": "linear",
                "evolution_params": {"a": 1.0, "b": 0.0, "c": 0.0},
            }
        ],
        "relations": [],
        "outcomes": [
            {"id": "yes", "label": "YES", "scoring_weights": {"signal": 1.0}},
            {"id": "no", "label": "NO", "scoring_weights": {"signal": -1.0}},
        ],
        "causal_dag": [],
        "metadata": {},
    }


def _japan_debt_policy_schema() -> dict:
    return {
        "domain_id": "japan_debt",
        "actors": [
            {"id": "japan", "name": "Japan", "state": {"influence": 0.72}},
        ],
        "resources": [
            {
                "id": "japan_debt_ratio",
                "name": "Japan Debt Ratio",
                "value": 2.6,
                "unit": "ratio",
                "owner_id": "japan",
                "min_value": 0.0,
                "max_value": 5.0,
                "evolution_type": "linear",
                "evolution_params": {
                    "a": 0.85,
                    "b": -0.2,
                    "c": 0.05,
                },
            },
            {
                "id": "japan_gdp_growth",
                "name": "Japan GDP Growth",
                "value": 0.25,
                "unit": "rate",
                "min_value": -1.0,
                "max_value": 1.0,
                "evolution_type": "linear",
                "evolution_params": {
                    "a": 0.4,
                    "c": 0.35,
                    "coupling_weights": {"japan_debt_ratio": -0.12},
                },
            },
        ],
        "relations": [],
        "outcomes": [
            {
                "id": "debt_crisis",
                "label": "Debt Crisis",
                "scoring_weights": {
                    "japan_debt_ratio": 3.5,
                    "japan_gdp_growth": -3.0,
                },
            },
            {
                "id": "stable",
                "label": "Stable",
                "scoring_weights": {
                    "japan_debt_ratio": -2.5,
                    "japan_gdp_growth": 4.0,
                },
            },
        ],
        "causal_dag": [],
        "metadata": {"base_year": 2025, "time_unit": "year"},
    }


def test_policy_evaluator_ranks_policies_and_preserves_world(tmp_path) -> None:
    knowledge_graph = KnowledgeGraph(json_path=tmp_path / "kg.json", auto_load=False, auto_save=False)
    knowledge_graph.add_node(
        KGNode(
            id="self:forecast_error:policy_lab:good",
            label="Policy Lab Calibration",
            node_type="self_observation",
            content="MAE tracker.",
            confidence=0.9,
            metadata={"domain_id": "policy_lab", "outcome_id": "good", "mean_abs_error": 0.25, "n_forecasts": 4},
        )
    )
    world = DomainCompiler().compile(_policy_lab_schema())
    before = world.snapshot()
    evaluator = PolicyEvaluator(
        sim_config=SimConfig(max_steps=6, level2_check_every=3, convergence_check_steps=10, fixed_point_max_iter=6, seed=7),
        epistemic_log=EpistemicLog(knowledge_graph),
        planning_horizon=5,
        stability_patience=0,
    )
    grow = Policy(actor_id="planner", actions={"expand_buffer": 1.0})
    neutral = Policy(actor_id="planner", actions={"hold": 0.0})
    shrink = Policy(actor_id="planner", actions={"drain_buffer": -1.0})

    results = evaluator.evaluate(world, [neutral, shrink, grow])

    assert world.snapshot() == before
    assert [result.policy.actions for result in results] == [
        {"expand_buffer": 1.0},
        {"hold": 0.0},
        {"drain_buffer": -1.0},
    ]
    assert [result.rank for result in results] == [1, 2, 3]
    assert results[0].epistemic_weight == pytest.approx(1.0 / 1.25)
    assert results[0].epistemic_score > results[1].epistemic_score > results[2].epistemic_score


def test_policy_evaluator_penalizes_hard_violations() -> None:
    world = DomainCompiler().compile(_conserved_policy_schema())
    evaluator = PolicyEvaluator(
        sim_config=SimConfig(max_steps=3, level2_check_every=1, convergence_check_steps=6, fixed_point_max_iter=4, seed=11),
        planning_horizon=3,
        stability_patience=0,
    )
    safe = Policy(actor_id="planner", actions={"hold": 0.0})
    overflow = Policy(actor_id="planner", actions={"release": 1.0})

    results = evaluator.evaluate(world, [overflow, safe])

    assert results[0].policy == safe
    assert results[0].hard_violations == 0
    assert results[1].policy == overflow
    assert results[1].hard_violations >= 1
    assert results[1].stop_reason == "hard_level0_violation"


def test_policy_evaluator_stops_early_on_convergence() -> None:
    world = DomainCompiler().compile(_constant_world_schema())
    evaluator = PolicyEvaluator(
        sim_config=SimConfig(max_steps=6, level2_check_every=3, convergence_check_steps=4, fixed_point_max_iter=3, seed=3),
        planning_horizon=6,
        stability_tol=0.0,
        stability_patience=1,
        min_stability_steps=1,
    )

    results = evaluator.evaluate(world, [Policy(actor_id="planner", actions={"hold": 0.0})])

    assert results[0].steps_run < 6
    assert results[0].stop_reason == "outcome_convergence"


def test_analysis_pipeline_policy_ranking_is_optional_and_side_effect_free(tmp_path) -> None:
    schema = _japan_debt_policy_schema()
    knowledge_graph = KnowledgeGraph(json_path=tmp_path / "kg.json", auto_load=False, auto_save=False)
    pipeline = AnalysisPipeline(
        sim_config=SimConfig(max_steps=6, level2_check_every=3, convergence_check_steps=10, fixed_point_max_iter=6, seed=13),
        knowledge_graph=knowledge_graph,
    )
    evaluator = PolicyEvaluator(
        sim_config=SimConfig(max_steps=6, level2_check_every=3, convergence_check_steps=10, fixed_point_max_iter=6, seed=13),
        epistemic_log=EpistemicLog(knowledge_graph),
        planning_horizon=5,
        stability_patience=0,
    )
    candidates = [
        Policy(actor_id="japan", actions={"debt_stabilization": 1.0}),
        Policy(actor_id="japan", actions={"hold": 0.0}),
        Policy(actor_id="japan", actions={"deficit_expansion": -1.0}),
    ]

    baseline = pipeline.run(schema)
    planned = pipeline.run(schema, policy_evaluator=evaluator, candidate_policies=candidates)

    assert baseline.policy_ranking == []
    assert len(planned.policy_ranking) == 3
    assert baseline.simulation["final_outcome_probs"] == planned.simulation["final_outcome_probs"]
    assert planned.metadata["policy_ranking"][0]["rank"] == 1


def test_policy_evaluator_ranks_japan_debt_policies() -> None:
    world = DomainCompiler().compile(_japan_debt_policy_schema())
    evaluator = PolicyEvaluator(
        sim_config=SimConfig(max_steps=8, level2_check_every=4, convergence_check_steps=12, fixed_point_max_iter=8, seed=5),
        planning_horizon=6,
        stability_patience=0,
    )
    austerity = Policy(actor_id="japan", actions={"debt_stabilization": 1.0})
    neutral = Policy(actor_id="japan", actions={"hold": 0.0})
    stimulus = Policy(actor_id="japan", actions={"deficit_expansion": -1.0})

    results = evaluator.evaluate(world, [stimulus, neutral, austerity])

    assert [result.policy.actions for result in results] == [
        {"debt_stabilization": 1.0},
        {"hold": 0.0},
        {"deficit_expansion": -1.0},
    ]
    assert results[0].expected_utility > results[1].expected_utility > results[2].expected_utility
    assert results[0].dominant_outcome == "stable"
    assert results[-1].dominant_outcome == "debt_crisis"
