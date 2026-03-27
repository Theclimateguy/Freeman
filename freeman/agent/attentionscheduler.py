"""Attention budgeting and UCB task scheduling."""

from __future__ import annotations

from dataclasses import dataclass, field
from math import inf, log, sqrt
from typing import Dict, List


VALID_STATES = {"PENDING", "ACTIVE", "SUSPENDED", "COMPLETED", "ARCHIVED"}
ALLOWED_TRANSITIONS = {
    "PENDING": {"ACTIVE", "ARCHIVED"},
    "ACTIVE": {"SUSPENDED", "COMPLETED", "ARCHIVED"},
    "SUSPENDED": {"ACTIVE", "ARCHIVED"},
    "COMPLETED": {"ARCHIVED"},
    "ARCHIVED": set(),
}


@dataclass
class AttentionTask:
    """Task competing for the finite attention budget."""

    task_id: str
    description: str
    expected_information_gain: float
    cost: float
    anomaly_score: float = 0.0
    semantic_gap: float = 0.0
    confidence_gap: float = 0.0
    state: str = "PENDING"
    pulls: int = 0
    last_interest_score: float = 0.0
    metadata: Dict[str, float | str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.state not in VALID_STATES:
            raise ValueError(f"Invalid task state: {self.state}")
        self.cost = float(self.cost)
        self.expected_information_gain = float(self.expected_information_gain)
        self.anomaly_score = float(self.anomaly_score)
        self.semantic_gap = float(self.semantic_gap)
        self.confidence_gap = float(self.confidence_gap)


@dataclass
class AttentionDecision:
    """One scheduler choice."""

    task_id: str
    state: str
    interest_score: float
    exploration_bonus: float
    ucb_score: float
    remaining_budget: float


class AttentionScheduler:
    """Finite-budget UCB scheduler over analysis tasks."""

    def __init__(self, attention_budget: float, ucb_beta: float = 1.0) -> None:
        self.attention_budget = float(attention_budget)
        self.remaining_budget = float(attention_budget)
        self.ucb_beta = float(ucb_beta)
        self.t = 0
        self.tasks: Dict[str, AttentionTask] = {}

    def add_task(self, task: AttentionTask) -> None:
        self.tasks[task.task_id] = task

    def transition(self, task_id: str, new_state: str) -> None:
        if new_state not in VALID_STATES:
            raise ValueError(f"Invalid task state: {new_state}")
        task = self.tasks[task_id]
        if new_state not in ALLOWED_TRANSITIONS[task.state]:
            raise ValueError(f"Invalid transition {task.state} -> {new_state}")
        task.state = new_state

    def interest_score(self, task: AttentionTask) -> float:
        """Expected information gain per cost with anomaly and semantic terms."""

        score = (
            task.expected_information_gain
            + task.anomaly_score
            + task.semantic_gap
            + task.confidence_gap
        ) / max(task.cost, 1.0e-8)
        task.last_interest_score = float(score)
        return float(score)

    def eligible_tasks(self) -> List[AttentionTask]:
        return [
            task
            for task in self.tasks.values()
            if task.state in {"PENDING", "ACTIVE", "SUSPENDED"} and task.cost <= self.remaining_budget
        ]

    def select_task(self) -> AttentionDecision | None:
        """Choose the next task by UCB and consume its budget."""

        eligible = self.eligible_tasks()
        if not eligible:
            return None

        self.t += 1
        best_task = None
        best_interest = 0.0
        best_bonus = 0.0
        best_score = -inf

        for task in eligible:
            interest = self.interest_score(task)
            if task.pulls == 0:
                bonus = inf
                ucb_score = inf
            else:
                bonus = self.ucb_beta * sqrt(log(max(self.t, 1)) / task.pulls)
                ucb_score = interest + bonus
            if ucb_score > best_score:
                best_task = task
                best_interest = interest
                best_bonus = bonus
                best_score = ucb_score

        if best_task is None:
            return None

        self.remaining_budget -= best_task.cost
        best_task.pulls += 1
        if best_task.state in {"PENDING", "SUSPENDED"}:
            best_task.state = "ACTIVE"

        return AttentionDecision(
            task_id=best_task.task_id,
            state=best_task.state,
            interest_score=best_interest,
            exploration_bonus=best_bonus,
            ucb_score=best_score,
            remaining_budget=self.remaining_budget,
        )


__all__ = ["AttentionDecision", "AttentionScheduler", "AttentionTask"]
