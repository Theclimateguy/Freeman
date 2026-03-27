"""Outcome scoring and confidence estimation."""

from __future__ import annotations

from typing import Dict, Iterable

import numpy as np

from freeman.core.access import get_world_value
from freeman.core.types import Violation
from freeman.core.world import WorldState


def score_outcomes(world: WorldState) -> Dict[str, float]:
    """Score outcomes from world values and return a softmax distribution."""

    if not world.outcomes:
        return {}

    raw_scores: Dict[str, np.float64] = {}
    for outcome_id, outcome in world.outcomes.items():
        score = np.float64(0.0)
        for key, weight in outcome.scoring_weights.items():
            score += np.float64(weight) * np.float64(get_world_value(world, key))
        raw_scores[outcome_id] = score

    max_score = max(raw_scores.values())
    exp_scores = {key: np.exp(value - max_score) for key, value in raw_scores.items()}
    total = np.sum(list(exp_scores.values()), dtype=np.float64)
    return {key: float(np.float64(value / total)) for key, value in exp_scores.items()}


def compute_confidence(outcome_probs: Dict[str, float], violations: Iterable[Violation]) -> float:
    """Compute confidence from outcome concentration and soft-violation count."""

    if not outcome_probs:
        return 0.0

    probs = np.array(list(outcome_probs.values()), dtype=np.float64)
    entropy = -np.sum(probs * np.log(probs + np.float64(1.0e-10)), dtype=np.float64)
    entropy_max = np.log(np.float64(len(probs))) if len(probs) > 1 else np.float64(1.0)
    entropy_factor = np.float64(1.0) - entropy / entropy_max if entropy_max > 0 else np.float64(1.0)
    soft_violations = sum(1 for violation in violations if violation.severity == "soft")
    violation_penalty = max(0.0, 1.0 - 0.05 * soft_violations)
    return float(np.round(entropy_factor * np.float64(violation_penalty), 4))
