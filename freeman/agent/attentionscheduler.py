"""Attention budgeting and UCB task scheduling."""

from __future__ import annotations

from collections import deque
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
class ForecastDebt:
    """Open forecast awaiting verification at a finite horizon."""

    task_id: str
    domain_id: str
    horizon_remaining: int
    urgency: float = 0.0

    def __post_init__(self) -> None:
        self.horizon_remaining = int(self.horizon_remaining)
        self.urgency = float(1.0 / max(self.horizon_remaining, 1))


@dataclass
class ConflictDebt:
    """Open review/conflict node that has aged in the KG."""

    task_id: str
    node_id: str
    age_steps: int
    urgency: float = 0.0

    def __post_init__(self) -> None:
        self.age_steps = int(self.age_steps)
        self.urgency = float(min(self.age_steps / 10.0, 1.0))


@dataclass
class AnomalyDebt:
    """Open anomaly signal that has not been analyzed yet."""

    task_id: str
    signal_id: str
    age_hours: float
    urgency: float = 0.0

    def __post_init__(self) -> None:
        self.age_hours = float(self.age_hours)
        self.urgency = float(min(self.age_hours / 24.0, 1.0))


@dataclass
class ObligationQueue:
    """Track unresolved obligations and expose their aggregate pressure."""

    forecast_debts: List[ForecastDebt] = field(default_factory=list)
    conflict_debts: List[ConflictDebt] = field(default_factory=list)
    anomaly_debts: List[AnomalyDebt] = field(default_factory=list)

    def add_forecast_debt(self, debt: ForecastDebt) -> None:
        self.forecast_debts.append(debt)

    def add_conflict_debt(self, debt: ConflictDebt) -> None:
        self.conflict_debts.append(debt)

    def add_anomaly_debt(self, debt: AnomalyDebt) -> None:
        self.anomaly_debts.append(debt)

    def pressure(self, task_id: str) -> float:
        """Return the cumulative obligation pressure for one task."""

        forecast_pressure = sum(debt.urgency for debt in self.forecast_debts if debt.task_id == task_id)
        conflict_pressure = sum(debt.urgency for debt in self.conflict_debts if debt.task_id == task_id)
        anomaly_pressure = sum(debt.urgency for debt in self.anomaly_debts if debt.task_id == task_id)
        return float(forecast_pressure + conflict_pressure + anomaly_pressure)


@dataclass
class InterestNormalizer:
    """Rolling z-score normalizer for heterogeneous interest components."""

    window: int = 50
    epsilon: float = 1.0e-6
    _history: Dict[str, deque[float]] = field(default_factory=dict, init=False, repr=False)

    def normalize(self, component_name: str, value: float) -> float:
        """Return a clipped rolling z-score using previously observed values."""

        if self.window <= 1:
            return 0.0
        history = list(self._history.get(component_name, ()))
        if len(history) < 2:
            return 0.0
        mean = sum(history) / len(history)
        variance = sum((entry - mean) ** 2 for entry in history) / len(history)
        std = sqrt(variance)
        if std < self.epsilon:
            return 0.0
        z_score = (float(value) - mean) / std
        return float(min(max(z_score, -3.0), 3.0))

    def observe(self, component_name: str, value: float) -> None:
        """Store one raw component value in the rolling history."""

        history = self._history.setdefault(component_name, deque(maxlen=max(self.window, 1)))
        history.append(float(value))

    def is_ready(self, component_name: str) -> bool:
        """Return True when the component has enough variation for normalization."""

        if self.window <= 1:
            return False
        history = list(self._history.get(component_name, ()))
        if len(history) < 2:
            return False
        mean = sum(history) / len(history)
        variance = sum((entry - mean) ** 2 for entry in history) / len(history)
        return variance >= self.epsilon**2


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
    trail_weight: float = 0.0
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
        self.trail_weight = max(float(self.trail_weight), 0.0)


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

    def __init__(
        self,
        attention_budget: float,
        ucb_beta: float = 1.0,
        obligation_queue: ObligationQueue | None = None,
        interest_normalizer: InterestNormalizer | None = None,
    ) -> None:
        self.attention_budget = float(attention_budget)
        self.remaining_budget = float(attention_budget)
        self.ucb_beta = float(ucb_beta)
        self.obligation_queue = obligation_queue
        self.interest_normalizer = interest_normalizer or InterestNormalizer()
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

        components = self._interest_components(task)
        normalized_components = self._normalize_components(components)
        score = sum(normalized_components.values()) / max(task.cost, 1.0e-8)
        task.last_interest_score = float(score)
        return float(score)

    def _interest_components(self, task: AttentionTask) -> Dict[str, float]:
        """Return raw component values before normalization."""

        obligation = self.obligation_queue.pressure(task.task_id) if self.obligation_queue is not None else 0.0
        metadata_trail = float(task.metadata.get("trail_weight", 0.0)) if task.metadata else 0.0
        return {
            "expected_information_gain": task.expected_information_gain,
            "anomaly_score": task.anomaly_score,
            "semantic_gap": task.semantic_gap,
            "confidence_gap": task.confidence_gap,
            "obligation_pressure": obligation,
            "trail_weight": max(task.trail_weight, metadata_trail),
        }

    def _normalize_components(self, components: Dict[str, float]) -> Dict[str, float]:
        """Normalize each component when rolling statistics are available."""

        normalized: Dict[str, float] = {}
        for name, value in components.items():
            if self.interest_normalizer.is_ready(name):
                normalized[name] = self.interest_normalizer.normalize(name, value)
            else:
                normalized[name] = float(value)
        return normalized

    def _observe_components(self, components_by_task: Dict[str, Dict[str, float]]) -> None:
        """Update rolling statistics from the current decision frontier."""

        for components in components_by_task.values():
            for name, value in components.items():
                self.interest_normalizer.observe(name, value)

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
        components_by_task: Dict[str, Dict[str, float]] = {}

        for task in eligible:
            components_by_task[task.task_id] = self._interest_components(task)
            interest = self._normalize_components(components_by_task[task.task_id])
            interest_value = sum(interest.values()) / max(task.cost, 1.0e-8)
            task.last_interest_score = float(interest_value)
            if task.pulls == 0:
                bonus = inf
                ucb_score = inf
            else:
                bonus = self.ucb_beta * sqrt(log(max(self.t, 1)) / task.pulls)
                ucb_score = interest_value + bonus
            if ucb_score > best_score:
                best_task = task
                best_interest = interest_value
                best_bonus = bonus
                best_score = ucb_score

        if best_task is None:
            return None

        self._observe_components(components_by_task)
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


__all__ = [
    "AnomalyDebt",
    "AttentionDecision",
    "InterestNormalizer",
    "AttentionScheduler",
    "AttentionTask",
    "ConflictDebt",
    "ForecastDebt",
    "ObligationQueue",
]
