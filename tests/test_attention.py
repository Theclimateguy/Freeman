"""Tests for UCB attention scheduling."""

from __future__ import annotations

import pytest

from freeman.agent.attentionscheduler import (
    AnomalyDebt,
    AttentionScheduler,
    AttentionTask,
    ConflictDebt,
    ForecastDebt,
    InterestNormalizer,
    ObligationQueue,
)


def test_normalizer_zero_variance() -> None:
    normalizer = InterestNormalizer(window=5)
    for _ in range(4):
        normalizer.observe("expected_information_gain", 1.0)

    normalized = normalizer.normalize("expected_information_gain", 1.0)

    assert normalized == pytest.approx(0.0)


def test_normalizer_clip() -> None:
    normalizer = InterestNormalizer(window=10)
    for value in [0.0, 0.1, -0.1, 0.05, -0.05]:
        normalizer.observe("semantic_gap", value)

    normalized = normalizer.normalize("semantic_gap", 100.0)

    assert normalized == pytest.approx(3.0)


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


def test_attention_scheduler_prioritizes_obligation_pressure() -> None:
    obligations = ObligationQueue()
    obligations.add_conflict_debt(ConflictDebt(task_id="debt_task", node_id="review:1", age_steps=8))
    scheduler = AttentionScheduler(attention_budget=3.0, ucb_beta=0.0, obligation_queue=obligations)
    scheduler.add_task(
        AttentionTask(
            task_id="high_gain",
            description="High gain but no debt",
            expected_information_gain=1.2,
            cost=1.0,
            pulls=1,
        )
    )
    scheduler.add_task(
        AttentionTask(
            task_id="debt_task",
            description="Moderate gain with debt pressure",
            expected_information_gain=0.5,
            cost=1.0,
            pulls=1,
        )
    )

    decision = scheduler.select_task()

    assert decision is not None
    assert decision.task_id == "debt_task"
    assert scheduler.tasks["debt_task"].last_interest_score == pytest.approx(1.3)


def test_obligation_urgency_grows_monotonically_with_age() -> None:
    younger_conflict = ConflictDebt(task_id="review", node_id="node:1", age_steps=2)
    older_conflict = ConflictDebt(task_id="review", node_id="node:2", age_steps=8)
    younger_anomaly = AnomalyDebt(task_id="anomaly", signal_id="signal:1", age_hours=4.0)
    older_anomaly = AnomalyDebt(task_id="anomaly", signal_id="signal:2", age_hours=18.0)
    far_forecast = ForecastDebt(task_id="forecast", domain_id="d1", horizon_remaining=6)
    near_forecast = ForecastDebt(task_id="forecast", domain_id="d1", horizon_remaining=1)

    assert younger_conflict.urgency < older_conflict.urgency <= 1.0
    assert younger_anomaly.urgency < older_anomaly.urgency <= 1.0
    assert far_forecast.urgency < near_forecast.urgency <= 1.0


def test_attention_scheduler_without_obligation_queue_preserves_legacy_interest() -> None:
    task = AttentionTask(
        task_id="legacy",
        description="Legacy task",
        expected_information_gain=1.0,
        cost=2.0,
        anomaly_score=0.2,
        semantic_gap=0.1,
        confidence_gap=0.1,
    )
    scheduler = AttentionScheduler(attention_budget=2.0, ucb_beta=0.0, obligation_queue=None)

    score = scheduler.interest_score(task)

    assert score == pytest.approx((1.0 + 0.2 + 0.1 + 0.1) / 2.0)


def test_scheduler_prefers_obligation_under_equal_eig() -> None:
    obligations = ObligationQueue()
    obligations.add_conflict_debt(ConflictDebt(task_id="obligation_task", node_id="review:1", age_steps=9))
    scheduler = AttentionScheduler(attention_budget=4.0, ucb_beta=0.0, obligation_queue=obligations)
    scheduler.add_task(
        AttentionTask(
            task_id="baseline_task",
            description="Equal EIG baseline",
            expected_information_gain=1.0,
            cost=1.0,
            pulls=1,
        )
    )
    scheduler.add_task(
        AttentionTask(
            task_id="obligation_task",
            description="Equal EIG with obligation",
            expected_information_gain=1.0,
            cost=1.0,
            pulls=1,
        )
    )

    decision = scheduler.select_task()

    assert decision is not None
    assert decision.task_id == "obligation_task"


def test_scheduler_backward_compat() -> None:
    scheduler = AttentionScheduler(attention_budget=2.0, ucb_beta=0.0)
    scheduler.add_task(
        AttentionTask(
            task_id="a",
            description="Task A",
            expected_information_gain=1.0,
            cost=1.0,
            anomaly_score=0.1,
            semantic_gap=0.1,
            confidence_gap=0.1,
            pulls=1,
        )
    )
    scheduler.add_task(
        AttentionTask(
            task_id="b",
            description="Task B",
            expected_information_gain=0.5,
            cost=1.0,
            anomaly_score=0.1,
            semantic_gap=0.1,
            confidence_gap=0.1,
            pulls=1,
        )
    )

    decision = scheduler.select_task()

    assert decision is not None
    assert decision.task_id == "a"
