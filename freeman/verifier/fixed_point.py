"""Fixed-point style correction loop for causal inconsistencies."""

from __future__ import annotations

from typing import Dict, Iterable, Tuple

import numpy as np

from freeman.core.access import apply_delta
from freeman.core.types import Violation
from freeman.core.world import WorldState
from freeman.utils import SIGN_EPSILON
from freeman.verifier.level2 import level2_check


def compute_corrections(violations: Iterable[Violation], alpha: float = 0.1) -> Dict[str, float]:
    """Convert sign violations into additive state corrections."""

    corrections: Dict[str, float] = {}
    for violation in violations:
        edge = violation.details.get("edge", {})
        target = edge.get("target")
        if not target:
            continue
        expected_sign = edge.get("expected_sign", "+")
        observed_delta = float(violation.details.get("observed_delta", 0.0))
        magnitude = max(abs(observed_delta), SIGN_EPSILON)
        direction = 1.0 if expected_sign == "+" else -1.0
        corrections[target] = corrections.get(target, 0.0) + direction * float(np.float64(alpha) * np.float64(magnitude))
    return corrections


def apply_corrections(world: WorldState, corrections: Dict[str, float]) -> WorldState:
    """Apply additive corrections to a cloned world state."""

    corrected = world.clone()
    for key, delta in corrections.items():
        apply_delta(corrected, key, delta)
    return corrected


def find_fixed_point(
    world: WorldState,
    causal_dag,
    max_iter: int = 20,
    alpha: float = 0.1,
    tol: float = 1.0e-6,
    base_delta: float = 0.01,
    dt: float = 1.0,
) -> Tuple[WorldState, bool, int]:
    """Iteratively reduce sign violations by applying small state corrections."""

    current = world.clone()
    for iteration in range(max_iter):
        violations = level2_check(current, causal_dag, base_delta=base_delta, dt=dt)
        sign_violations = [violation for violation in violations if violation.check_name == "sign_consistency"]
        if not sign_violations:
            return current, True, iteration
        corrections = compute_corrections(sign_violations, alpha=alpha)
        if not corrections or max(abs(delta) for delta in corrections.values()) < tol:
            return current, False, iteration
        current = apply_corrections(current, corrections)
    return current, False, max_iter
