"""Tests for cost governance."""

from __future__ import annotations

from freeman.agent.costmodel import BudgetPolicy, CostModel


def test_costmodel_estimate_tracks_task_components() -> None:
    model = CostModel()

    estimate = model.estimate(
        task_id="task",
        llm_calls=2,
        sim_steps=10,
        actors=3,
        resources=4,
        domains=2,
        kg_updates=5,
        embedding_tokens_used=250,
    )

    assert estimate.estimated_cost > 0.0
    assert estimate.domains == 2
    assert estimate.kg_updates == 5
    assert estimate.embedding_tokens_used == 250


def test_costmodel_downgrades_deep_dive_and_stops_on_hard_limits() -> None:
    model = CostModel(
        BudgetPolicy(
            max_llm_calls_per_task=3,
            max_sim_steps_per_task=50,
            max_domains_per_task=2,
            max_deep_dive_depth=1,
            max_compute_budget_per_session=40.0,
        )
    )
    deep_estimate = model.estimate(
        task_id="deep",
        llm_calls=1,
        sim_steps=10,
        actors=2,
        resources=2,
        domains=1,
    )
    deep_decision = model.precheck(
        requested_mode="DEEP_DIVE",
        estimate=deep_estimate,
        budget_spent=0.0,
        deep_dive_depth=1,
    )

    hard_estimate = model.estimate(
        task_id="hard",
        llm_calls=5,
        sim_steps=10,
        actors=2,
        resources=2,
        domains=1,
    )
    hard_decision = model.precheck(
        requested_mode="ANALYZE",
        estimate=hard_estimate,
        budget_spent=0.0,
    )

    assert deep_decision.approved_mode == "ANALYZE"
    assert deep_decision.allowed is True
    assert deep_decision.stop_reason == "max_deep_dive_depth"
    assert hard_decision.allowed is False
    assert hard_decision.stop_reason == "max_llm_calls_per_task"
