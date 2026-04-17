"""Benchmark metrics for FAAB."""

from __future__ import annotations

from typing import Iterable, Mapping, Sequence


def brier_score(predicted_probs: Mapping[str, float], actual_outcome: str) -> float:
    """Return the multiclass Brier score for one forecast."""

    outcomes = set(predicted_probs) | {str(actual_outcome)}
    total = 0.0
    target = str(actual_outcome)
    for outcome in outcomes:
        probability = float(predicted_probs.get(outcome, 0.0))
        indicator = 1.0 if outcome == target else 0.0
        total += (probability - indicator) ** 2
    return float(total)


def _normalize_distribution(distribution: Mapping[str, float], support: Sequence[str]) -> dict[str, float]:
    """Project one categorical distribution onto ``support`` and normalize it."""

    normalized = {outcome: max(float(distribution.get(outcome, 0.0)), 0.0) for outcome in support}
    total = sum(normalized.values())
    if total > 0.0:
        return {outcome: probability / total for outcome, probability in normalized.items()}
    if not support:
        return {}
    uniform = 1.0 / len(support)
    return {outcome: uniform for outcome in support}


def max_distribution_l1_distance(distributions: Iterable[Mapping[str, float]]) -> float | None:
    """Return the maximum pairwise L1 distance between categorical distributions."""

    items = [dict(distribution) for distribution in distributions]
    if not items:
        return None
    if len(items) == 1:
        return 0.0
    support = sorted({outcome for distribution in items for outcome in distribution})
    normalized = [_normalize_distribution(distribution, support) for distribution in items]
    max_distance = 0.0
    for index, left in enumerate(normalized):
        for right in normalized[index + 1 :]:
            distance = sum(abs(left[outcome] - right[outcome]) for outcome in support)
            max_distance = max(max_distance, float(distance))
    return float(max_distance)


def probability_tar_at_n(distributions: Iterable[Mapping[str, float]], *, epsilon: float = 1.0e-9) -> float | None:
    """Return 1 if repeated forecast distributions agree within ``epsilon`` in L1."""

    max_distance = max_distribution_l1_distance(distributions)
    if max_distance is None:
        return None
    return 1.0 if max_distance <= float(epsilon) else 0.0


def outcome_tar_at_n(outcomes: Iterable[str | None]) -> float | None:
    """Return 1 if all repeated dominant outcomes are identical."""

    values = [outcome for outcome in outcomes]
    if not values:
        return None
    first = values[0]
    return 1.0 if all(value == first for value in values[1:]) else 0.0


__all__ = [
    "brier_score",
    "max_distribution_l1_distance",
    "outcome_tar_at_n",
    "probability_tar_at_n",
]
