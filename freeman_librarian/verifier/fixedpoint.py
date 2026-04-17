"""Fixed-point iteration with spectral-radius guard."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Tuple

import numpy as np

from freeman_librarian.core.access import apply_delta
from freeman_librarian.core.types import CausalEdge, Violation
from freeman_librarian.core.world import WorldState
from freeman_librarian.utils import SIGN_EPSILON
from freeman_librarian.verifier.level1 import compute_jacobian, spectral_radius
from freeman_librarian.verifier.level2 import level2_check


@dataclass
class FixedPointResult:
    """Outcome of a fixed-point correction loop."""

    world: WorldState
    converged: bool
    iterations: int
    spectral_radius: float
    violations: List[Violation] = field(default_factory=list)
    history: List[Dict[str, float | int | bool]] = field(default_factory=list)


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


def iterate_fixed_point(
    world: WorldState,
    causal_dag: List[CausalEdge],
    max_iter: int = 20,
    alpha: float = 0.1,
    tol: float = 1.0e-6,
    base_delta: float = 0.01,
    dt: float = 1.0,
    jacobian_guard: float = 1.0 + 1.0e-6,
) -> FixedPointResult:
    """Iteratively reduce sign violations while enforcing a spectral-radius guard."""

    current = world.clone()
    history: List[Dict[str, float | int | bool]] = []

    for iteration in range(max_iter + 1):
        jacobian = compute_jacobian(current, dt=dt)
        rho = spectral_radius(jacobian)
        violations = level2_check(current, causal_dag, base_delta=base_delta, dt=dt)
        sign_violations = [violation for violation in violations if violation.check_name == "sign_consistency"]

        history.append(
            {
                "iteration": iteration,
                "spectral_radius": float(rho),
                "sign_violations": len(sign_violations),
                "guard_passed": bool(rho < jacobian_guard),
            }
        )

        if rho >= jacobian_guard:
            guard_violation = Violation(
                level=2,
                check_name="spectral_radius_guard",
                description=f"Jacobian spectral radius = {rho:.6f} exceeds guard {jacobian_guard:.6f}",
                severity="hard",
                details={
                    "field": "jacobian.spectral_radius",
                    "observed": float(rho),
                    "expected_max": float(jacobian_guard),
                },
            )
            return FixedPointResult(
                world=current,
                converged=False,
                iterations=iteration,
                spectral_radius=float(rho),
                violations=[guard_violation, *violations],
                history=history,
            )

        if not sign_violations:
            return FixedPointResult(
                world=current,
                converged=True,
                iterations=iteration,
                spectral_radius=float(rho),
                violations=violations,
                history=history,
            )

        if iteration >= max_iter:
            return FixedPointResult(
                world=current,
                converged=False,
                iterations=iteration,
                spectral_radius=float(rho),
                violations=violations,
                history=history,
            )

        corrections = compute_corrections(sign_violations, alpha=alpha)
        if not corrections or max(abs(delta) for delta in corrections.values()) < tol:
            return FixedPointResult(
                world=current,
                converged=False,
                iterations=iteration,
                spectral_radius=float(rho),
                violations=violations,
                history=history,
            )
        current = apply_corrections(current, corrections)

    return FixedPointResult(
        world=current,
        converged=False,
        iterations=max_iter,
        spectral_radius=float(spectral_radius(compute_jacobian(current, dt=dt))),
        violations=level2_check(current, causal_dag, base_delta=base_delta, dt=dt),
        history=history,
    )


def find_fixed_point(
    world: WorldState,
    causal_dag: List[CausalEdge],
    max_iter: int = 20,
    alpha: float = 0.1,
    tol: float = 1.0e-6,
    base_delta: float = 0.01,
    dt: float = 1.0,
    jacobian_guard: float = 1.0 + 1.0e-6,
) -> Tuple[WorldState, bool, int]:
    """Backward-compatible wrapper returning ``(world, converged, iterations)``."""

    result = iterate_fixed_point(
        world,
        causal_dag,
        max_iter=max_iter,
        alpha=alpha,
        tol=tol,
        base_delta=base_delta,
        dt=dt,
        jacobian_guard=jacobian_guard,
    )
    return result.world, result.converged, result.iterations


__all__ = [
    "FixedPointResult",
    "apply_corrections",
    "compute_corrections",
    "find_fixed_point",
    "iterate_fixed_point",
]
