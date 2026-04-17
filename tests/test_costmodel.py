"""Tests for cost governance."""

from __future__ import annotations

from freeman.agent.costmodel import BudgetLedger, BudgetPolicy, CostModel


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


def test_budget_ledger_summarizes_allowed_and_blocked_tasks(tmp_path) -> None:
    policy = BudgetPolicy(max_compute_budget_per_session=25.0)
    ledger = BudgetLedger(tmp_path / "cost_ledger.jsonl", policy=policy, auto_load=False)
    model = CostModel(policy)

    allowed_estimate = model.estimate(
        task_id="signal-1",
        llm_calls=1,
        sim_steps=1,
        actors=1,
        resources=1,
        domains=1,
    )
    allowed_decision = model.precheck(
        requested_mode="ANALYZE",
        estimate=allowed_estimate,
        budget_spent=0.0,
    )
    blocked_estimate = model.estimate(
        task_id="signal-2",
        llm_calls=policy.max_llm_calls_per_task + 1,
        sim_steps=0,
        actors=0,
        resources=0,
        domains=0,
    )
    blocked_decision = model.precheck(
        requested_mode="ANALYZE",
        estimate=blocked_estimate,
        budget_spent=0.0,
    )

    ledger.record(
        task_type="signal_processing",
        requested_mode="ANALYZE",
        decision=allowed_decision,
        actual_cost=5.0,
        metadata={"signal_id": "signal-1"},
    )
    ledger.record(
        task_type="answer_generation",
        requested_mode="ANALYZE",
        decision=blocked_decision,
        actual_cost=0.0,
        metadata={"query": "risk outlook"},
    )

    restored = BudgetLedger(tmp_path / "cost_ledger.jsonl", policy=policy, auto_load=True)
    summary = restored.summary()

    assert summary["tracking_enabled"] is True
    assert summary["spent_usd"] == 5.0
    assert summary["remaining_usd"] == 20.0
    assert summary["entry_count"] == 2
    assert summary["allowed_count"] == 1
    assert summary["blocked_count"] == 1
    assert summary["by_task_type"] == {"signal_processing": 1, "answer_generation": 1}
