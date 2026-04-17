"""Level-1 structural checks."""

from __future__ import annotations

from typing import Any, List

import numpy as np
from scipy.linalg import eigvals

from freeman_librarian.core.access import resource_vector, state_distance
from freeman_librarian.core.transition import step_world
from freeman_librarian.core.types import Policy, Violation
from freeman_librarian.core.world import WorldState
from freeman_librarian.exceptions import HardStopException


def compute_jacobian(world: WorldState, dt: float = 1.0, epsilon: float = 1.0e-6) -> np.ndarray:
    """Compute the Jacobian of the null-policy transition map numerically."""

    resource_ids = sorted(world.resources)
    if not resource_ids:
        return np.zeros((0, 0), dtype=np.float64)

    null_policies = [Policy(actor_id=actor_id, actions={}) for actor_id in world.actors]
    base_next, _ = step_world(world.clone(), null_policies, dt=dt)
    base_vector = resource_vector(base_next)
    jacobian = np.zeros((len(resource_ids), len(resource_ids)), dtype=np.float64)

    for col, res_id in enumerate(resource_ids):
        perturbed = world.clone()
        scale = max(abs(float(perturbed.resources[res_id].value)), 1.0)
        step = np.float64(epsilon * scale)
        perturbed.resources[res_id].value = np.float64(perturbed.resources[res_id].value + step)
        perturbed_next, _ = step_world(perturbed, null_policies, dt=dt)
        jacobian[:, col] = (resource_vector(perturbed_next) - base_vector) / step

    return jacobian


def spectral_radius(jacobian: np.ndarray) -> float:
    """Return the spectral radius of a square matrix."""

    if jacobian.size == 0:
        return 0.0
    return float(np.max(np.abs(eigvals(jacobian))))


def check_shock_decay(world: WorldState, resource_id: str, config: Any) -> tuple[bool, float]:
    """Return decay status and final shocked-vs-baseline distance for one resource."""

    null_policies = [Policy(actor_id=actor_id, actions={}) for actor_id in world.actors]
    baseline = world.clone()
    shocked = world.clone()
    shocked.resources[resource_id].value = np.float64(shocked.resources[resource_id].value + np.float64(1.0))
    distance = state_distance(baseline, shocked)

    try:
        for _ in range(int(config.convergence_check_steps)):
            baseline, _ = step_world(baseline, null_policies, dt=float(config.dt))
            shocked, _ = step_world(shocked, null_policies, dt=float(config.dt))
            distance = state_distance(baseline, shocked)
            if distance < float(config.convergence_epsilon):
                return True, float(distance)
    except HardStopException:
        return False, float("inf")

    return False, float(distance)


def level1_check(world: WorldState, config: Any) -> List[Violation]:
    """Run structural stability checks at domain initialization."""

    violations: List[Violation] = []
    null_world = world.clone()
    null_policies = [Policy(actor_id=actor_id, actions={}) for actor_id in world.actors]
    prev_state = null_world.snapshot()
    last_distance = state_distance(prev_state, prev_state)

    try:
        for _ in range(int(config.convergence_check_steps)):
            null_world, _ = step_world(null_world, null_policies, dt=float(config.dt))
            curr_state = null_world.snapshot()
            last_distance = state_distance(prev_state, curr_state)
            if last_distance < float(config.convergence_epsilon):
                break
            prev_state = curr_state
        else:
            violations.append(
                Violation(
                    level=1,
                    check_name="null_action_convergence",
                    description=(
                        f"World did not converge in {int(config.convergence_check_steps)} steps "
                        "under the null policy"
                    ),
                    severity="soft",
                    details={
                        "field": "null_policy_rollout",
                        "observed": float(last_distance),
                        "expected_max": float(config.convergence_epsilon),
                        "steps": int(config.convergence_check_steps),
                    },
                )
            )
    except HardStopException as exc:
        violations.append(
            Violation(
                level=1,
                check_name="null_action_convergence",
                description="Null-policy rollout triggered a hard stop",
                severity="soft",
                details={
                    "field": "null_policy_rollout",
                    "observed": "hard_stop",
                    "expected": "stable rollout",
                    "violations": [violation.snapshot() for violation in exc.violations],
                },
            )
        )

    try:
        jacobian = compute_jacobian(world, dt=float(config.dt))
        rho = spectral_radius(jacobian)
        if rho >= 1.0:
            violations.append(
                Violation(
                    level=1,
                    check_name="spectral_radius",
                    description=f"Jacobian spectral radius = {rho:.6f} >= 1.0",
                    severity="soft",
                    details={
                        "field": "jacobian.spectral_radius",
                        "spectral_radius": rho,
                        "observed": rho,
                        "expected_max": 1.0,
                    },
                )
            )
    except HardStopException as exc:
        violations.append(
            Violation(
                level=1,
                check_name="spectral_radius",
                description="Jacobian evaluation triggered a hard stop",
                severity="soft",
                details={
                    "field": "jacobian.spectral_radius",
                    "observed": "hard_stop",
                    "expected": "< 1.0",
                    "violations": [violation.snapshot() for violation in exc.violations],
                },
            )
        )

    for resource_id in list(sorted(world.resources))[:3]:
        decayed, final_distance = check_shock_decay(world, resource_id, config)
        if not decayed:
            violations.append(
                Violation(
                    level=1,
                    check_name="shock_decay",
                    description=f"Shock on resource {resource_id} does not decay",
                    severity="soft",
                    details={
                        "field": f"resources.{resource_id}.shock_decay",
                        "resource_id": resource_id,
                        "observed": float(final_distance),
                        "expected_max": float(config.convergence_epsilon),
                        "steps": int(config.convergence_check_steps),
                    },
                )
            )

    return violations
