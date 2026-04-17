"""Level-2 rolling sign-consistency checks."""

from __future__ import annotations

from typing import List

from freeman_librarian.core.access import apply_delta, get_world_value
from freeman_librarian.core.transition import step_world
from freeman_librarian.core.types import CausalEdge, Violation
from freeman_librarian.core.world import WorldState
from freeman_librarian.exceptions import HardStopException
from freeman_librarian.utils import SIGN_EPSILON


def _repair_targets(world: WorldState, edge: CausalEdge) -> List[str]:
    """Return likely parameter paths that control the sign of ``edge``."""

    if edge.target not in world.resources:
        return []

    resource = world.resources[edge.target]
    params = resource.evolution_params
    targets: List[str] = []

    if edge.source in params.get("coupling_weights", {}):
        targets.append(f"resources.{edge.target}.evolution_params.coupling_weights.{edge.source}")
    if edge.source in params.get("phi_params", {}).get("coupling_weights", {}):
        targets.append(f"resources.{edge.target}.evolution_params.phi_params.coupling_weights.{edge.source}")
    for branch_name in ("low_params", "high_params"):
        if edge.source in params.get(branch_name, {}).get("coupling_weights", {}):
            targets.append(f"resources.{edge.target}.evolution_params.{branch_name}.coupling_weights.{edge.source}")
    for index, component in enumerate(params.get("components", [])):
        if edge.source in component.get("evolution_params", {}).get("coupling_weights", {}):
            targets.append(
                "resources."
                f"{edge.target}.evolution_params.components.{index}.evolution_params.coupling_weights.{edge.source}"
            )
        if edge.source in component.get("evolution_params", {}).get("phi_params", {}).get("coupling_weights", {}):
            targets.append(
                "resources."
                f"{edge.target}.evolution_params.components.{index}.evolution_params.phi_params.coupling_weights.{edge.source}"
            )

    return targets


def _compute_delta(world: WorldState, source_key: str, base_delta: float = 0.01) -> float:
    """Return a numerically meaningful shock size for ``source_key``."""

    current_value = abs(get_world_value(world, source_key))
    if current_value > 1.0:
        return current_value * base_delta
    return base_delta


def level2_check(
    world: WorldState,
    causal_dag: List[CausalEdge],
    base_delta: float = 0.01,
    dt: float = 1.0,
) -> List[Violation]:
    """Check whether local causal responses match the expected signs."""

    violations: List[Violation] = []

    try:
        next_base, _ = step_world(world.clone(), [], dt=dt)
    except HardStopException as exc:
        return [
            Violation(
                level=2,
                check_name="sign_consistency_execution",
                description="Base rollout for sign-consistency failed",
                severity="hard",
                details={
                    "field": "sign_consistency.base_rollout",
                    "observed": "hard_stop",
                    "expected": "successful rollout",
                    "violations": [violation.snapshot() for violation in exc.violations],
                },
            )
        ]

    for edge in causal_dag:
        shock_delta = _compute_delta(world, edge.source, base_delta=base_delta)
        shocked = apply_delta(world.clone(), edge.source, shock_delta)
        try:
            next_shocked, _ = step_world(shocked, [], dt=dt)
        except HardStopException as exc:
            violations.append(
                Violation(
                    level=2,
                    check_name="sign_consistency_execution",
                    description=f"Shocked rollout for edge {edge.source}->{edge.target} failed",
                    severity="hard",
                    details={
                        "field": f"causal_dag.{edge.source}->{edge.target}",
                        "edge": edge.snapshot(),
                        "observed": "hard_stop",
                        "expected": "successful shocked rollout",
                        "repair_targets": _repair_targets(world, edge),
                        "violations": [violation.snapshot() for violation in exc.violations],
                    },
                )
            )
            continue

        observed_delta = get_world_value(next_shocked, edge.target) - get_world_value(next_base, edge.target)
        if abs(observed_delta) <= SIGN_EPSILON:
            continue

        observed_sign = "+" if observed_delta > 0 else "-"
        if observed_sign != edge.expected_sign:
            violations.append(
                Violation(
                    level=2,
                    check_name="sign_consistency",
                    description=(
                        f"Edge {edge.source}->{edge.target}: expected {edge.expected_sign}, "
                        f"observed {observed_sign}"
                    ),
                    severity="soft" if edge.strength == "weak" else "hard",
                    details={
                        "field": f"causal_dag.{edge.source}->{edge.target}",
                        "edge": edge.snapshot(),
                        "observed": observed_sign,
                        "expected": edge.expected_sign,
                        "observed_delta": observed_delta,
                        "shock_delta": shock_delta,
                        "repair_targets": _repair_targets(world, edge),
                    },
                )
            )

    return violations
