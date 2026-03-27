"""Level-2 rolling sign-consistency checks."""

from __future__ import annotations

from typing import List

from freeman.core.access import apply_delta, get_world_value
from freeman.core.transition import step_world
from freeman.core.types import CausalEdge, Violation
from freeman.core.world import WorldState
from freeman.exceptions import HardStopException
from freeman.utils import SIGN_EPSILON


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
                details={"violations": [violation.snapshot() for violation in exc.violations]},
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
                        "edge": edge.snapshot(),
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
                        "edge": edge.snapshot(),
                        "observed_delta": observed_delta,
                        "shock_delta": shock_delta,
                    },
                )
            )

    return violations
