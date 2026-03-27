"""Level-0 invariant checks."""

from __future__ import annotations

from typing import List

import numpy as np

from freeman.core.scorer import score_outcomes
from freeman.core.types import Violation
from freeman.core.world import WorldState
from freeman.utils import EPSILON


def _infer_exogenous_inflow(world: WorldState) -> float:
    """Infer exogenous inflow when it is not explicitly stored in metadata."""

    total = np.float64(0.0)
    for resource in world.resources.values():
        params = resource.evolution_params
        if resource.evolution_type == "stock_flow":
            total += np.float64(params.get("phi_params", {}).get("base_inflow", 0.0))
        elif resource.evolution_type == "logistic":
            total += max(np.float64(0.0), np.float64(params.get("external", 0.0)))
        elif resource.evolution_type == "linear":
            total += max(np.float64(0.0), np.float64(params.get("c", 0.0)))
    return float(total)


def _total_exogenous_inflow(world: WorldState) -> float:
    """Return the exogenous inflow budget used by the conservation check."""

    if "exogenous_inflow" in world.metadata or "exogenous_inflows" in world.metadata:
        explicit_total = np.float64(world.metadata.get("exogenous_inflow", 0.0))
        inflows = world.metadata.get("exogenous_inflows", {})
        if isinstance(inflows, dict):
            explicit_total += np.sum(list(inflows.values()), dtype=np.float64)
        return float(explicit_total)
    return _infer_exogenous_inflow(world)


def level0_check(prev: WorldState, next: WorldState) -> List[Violation]:
    """Run hard invariants that must hold on every simulation step."""

    violations: List[Violation] = []
    total_prev = np.sum([resource.value for resource in prev.resources.values()], dtype=np.float64)
    total_next = np.sum([resource.value for resource in next.resources.values()], dtype=np.float64)
    external = np.float64(_total_exogenous_inflow(next))

    if total_next > total_prev + external + np.float64(EPSILON):
        violations.append(
            Violation(
                level=0,
                check_name="conservation",
                description=(
                    f"Resource sum grew by {float(total_next - total_prev):.6f} "
                    f"with exogenous allowance {float(external):.6f}"
                ),
                severity="hard",
                details={
                    "total_prev": float(total_prev),
                    "total_next": float(total_next),
                    "external": float(external),
                },
            )
        )

    for res_id, resource in next.resources.items():
        if resource.value < resource.min_value - np.float64(EPSILON):
            violations.append(
                Violation(
                    level=0,
                    check_name="nonnegativity",
                    description=f"Resource {res_id}={float(resource.value):.6f} < min {float(resource.min_value):.6f}",
                    severity="hard",
                    details={"resource_id": res_id},
                )
            )

    outcome_probs = score_outcomes(next)
    total_prob = np.sum(list(outcome_probs.values()), dtype=np.float64) if outcome_probs else np.float64(0.0)
    if not outcome_probs or abs(total_prob - np.float64(1.0)) > np.float64(EPSILON):
        violations.append(
            Violation(
                level=0,
                check_name="probability_simplex",
                description=f"Outcome probabilities sum to {float(total_prob):.12f}",
                severity="hard",
                details={"sum": float(total_prob)},
            )
        )

    for res_id, resource in next.resources.items():
        if np.isfinite(resource.max_value) and resource.value > resource.max_value + np.float64(EPSILON):
            violations.append(
                Violation(
                    level=0,
                    check_name="bounds",
                    description=f"Resource {res_id}={float(resource.value):.6f} > max {float(resource.max_value):.6f}",
                    severity="soft",
                    details={"resource_id": res_id},
                )
            )

    return violations
