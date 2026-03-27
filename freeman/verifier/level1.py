"""Level-1 structural checks."""

from __future__ import annotations

from typing import Any, List

import numpy as np
from scipy.linalg import eigvals

from freeman.core.access import resource_vector, state_distance
from freeman.core.transition import step_world
from freeman.core.types import Policy, Violation
from freeman.core.world import WorldState
from freeman.exceptions import HardStopException


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


def check_shock_decay(world: WorldState, resource_id: str, config: Any) -> bool:
    """Return ``True`` when a unit shock on a resource decays within ``K`` steps."""

    null_policies = [Policy(actor_id=actor_id, actions={}) for actor_id in world.actors]
    baseline = world.clone()
    shocked = world.clone()
    shocked.resources[resource_id].value = np.float64(shocked.resources[resource_id].value + np.float64(1.0))

    try:
        for _ in range(int(config.convergence_check_steps)):
            baseline, _ = step_world(baseline, null_policies, dt=float(config.dt))
            shocked, _ = step_world(shocked, null_policies, dt=float(config.dt))
            if state_distance(baseline, shocked) < float(config.convergence_epsilon):
                return True
    except HardStopException:
        return False

    return False


def level1_check(world: WorldState, config: Any) -> List[Violation]:
    """Run structural stability checks at domain initialization."""

    violations: List[Violation] = []
    null_world = world.clone()
    null_policies = [Policy(actor_id=actor_id, actions={}) for actor_id in world.actors]
    prev_state = null_world.snapshot()

    try:
        for _ in range(int(config.convergence_check_steps)):
            null_world, _ = step_world(null_world, null_policies, dt=float(config.dt))
            curr_state = null_world.snapshot()
            if state_distance(prev_state, curr_state) < float(config.convergence_epsilon):
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
                )
            )
    except HardStopException as exc:
        violations.append(
            Violation(
                level=1,
                check_name="null_action_convergence",
                description="Null-policy rollout triggered a hard stop",
                severity="soft",
                details={"violations": [violation.snapshot() for violation in exc.violations]},
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
                    details={"spectral_radius": rho},
                )
            )
    except HardStopException as exc:
        violations.append(
            Violation(
                level=1,
                check_name="spectral_radius",
                description="Jacobian evaluation triggered a hard stop",
                severity="soft",
                details={"violations": [violation.snapshot() for violation in exc.violations]},
            )
        )

    for resource_id in list(sorted(world.resources))[:3]:
        if not check_shock_decay(world, resource_id, config):
            violations.append(
                Violation(
                    level=1,
                    check_name="shock_decay",
                    description=f"Shock on resource {resource_id} does not decay",
                    severity="soft",
                    details={"resource_id": resource_id},
                )
            )

    return violations
