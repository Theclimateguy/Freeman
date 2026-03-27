"""Tests for UCB attention scheduling."""

from __future__ import annotations

import pytest

from freeman.agent.attentionscheduler import AttentionScheduler, AttentionTask


def test_attention_scheduler_prefers_unexplored_task_under_ucb() -> None:
    scheduler = AttentionScheduler(attention_budget=3.0, ucb_beta=0.5)
    scheduler.add_task(
        AttentionTask(
            task_id="known_task",
            description="Known task",
            expected_information_gain=1.2,
            cost=1.0,
            pulls=5,
        )
    )
    scheduler.add_task(
        AttentionTask(
            task_id="new_task",
            description="New task",
            expected_information_gain=0.8,
            cost=1.0,
            anomaly_score=0.3,
        )
    )

    decision = scheduler.select_task()

    assert decision is not None
    assert decision.task_id == "new_task"
    assert decision.state == "ACTIVE"
    assert scheduler.remaining_budget == 2.0


def test_attention_scheduler_enforces_state_machine() -> None:
    scheduler = AttentionScheduler(attention_budget=2.0, ucb_beta=1.0)
    scheduler.add_task(
        AttentionTask(
            task_id="task",
            description="Task",
            expected_information_gain=1.0,
            cost=1.0,
        )
    )

    scheduler.transition("task", "ACTIVE")
    scheduler.transition("task", "SUSPENDED")
    scheduler.transition("task", "ACTIVE")
    scheduler.transition("task", "COMPLETED")
    scheduler.transition("task", "ARCHIVED")

    assert scheduler.tasks["task"].state == "ARCHIVED"

    with pytest.raises(ValueError):
        scheduler.transition("task", "ACTIVE")
