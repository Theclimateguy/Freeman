"""Benchmark metrics for FAAB."""

from __future__ import annotations


def brier_score(predicted_probs: dict[str, float], actual_outcome: str) -> float:
    """Return the multiclass Brier score for one forecast."""

    outcomes = set(predicted_probs) | {str(actual_outcome)}
    total = 0.0
    target = str(actual_outcome)
    for outcome in outcomes:
        probability = float(predicted_probs.get(outcome, 0.0))
        indicator = 1.0 if outcome == target else 0.0
        total += (probability - indicator) ** 2
    return float(total)


__all__ = ["brier_score"]
