"""Behavioral tests for deterministic agent cycles."""

from __future__ import annotations

import copy

from freeman.agent.attentionscheduler import AttentionScheduler, AttentionTask, ConflictDebt, ObligationQueue

from harness import AgentHarness


def test_shock_triggers_analyze(water_shock_stream, water_market_schema) -> None:
    result = AgentHarness(water_market_schema, water_shock_stream, seed=42).run_cycle()

    assert result.decisions >= 1


def test_kg_grows_after_shock(water_shock_stream, water_market_schema) -> None:
    result = AgentHarness(water_market_schema, water_shock_stream, seed=42).run_cycle()

    assert result.kg_nodes >= 1


def test_null_stream_stays_watch(null_stream, water_market_schema) -> None:
    result = AgentHarness(water_market_schema, null_stream, seed=42).run_cycle()

    assert result.decisions == 0


def test_obligation_forces_return() -> None:
    obligations = ObligationQueue()
    obligations.add_conflict_debt(ConflictDebt(task_id="review_task", node_id="review:1", age_steps=9))
    scheduler = AttentionScheduler(attention_budget=3.0, ucb_beta=0.0, obligation_queue=obligations)
    scheduler.add_task(
        AttentionTask(
            task_id="stale_task",
            description="High-pull routine task",
            expected_information_gain=1.2,
            cost=1.0,
            pulls=10,
        )
    )
    scheduler.add_task(
        AttentionTask(
            task_id="new_task",
            description="Fresh task without debt",
            expected_information_gain=0.8,
            cost=1.0,
            pulls=1,
        )
    )
    scheduler.add_task(
        AttentionTask(
            task_id="review_task",
            description="Return to unresolved review task",
            expected_information_gain=0.4,
            cost=1.0,
            pulls=1,
        )
    )

    decision = scheduler.select_task()

    assert decision is not None
    assert decision.task_id == "review_task"


def test_proactive_alert_on_violation(water_shock_stream, water_market_schema) -> None:
    broken_schema = copy.deepcopy(water_market_schema)
    broken_schema["causal_dag"][0]["expected_sign"] = "-"

    result = AgentHarness(broken_schema, water_shock_stream, seed=42, enable_emitter=True).run_cycle()
    alert_events = [event for event in result.proactive_events if event["event_type"] == "alert"]

    assert alert_events
