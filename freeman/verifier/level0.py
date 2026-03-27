"""Level-0 invariant checks."""

from __future__ import annotations

from typing import Dict, List

import numpy as np

from freeman.core.scorer import score_outcomes
from freeman.core.types import Violation
from freeman.core.world import WorldState
from freeman.utils import EPSILON


def _infer_resource_exogenous_inflow(resource) -> float:
    """Infer resource-specific exogenous inflow when it is not explicitly configured."""

    params = resource.evolution_params
    if resource.evolution_type == "stock_flow":
        return float(np.float64(params.get("phi_params", {}).get("base_inflow", 0.0)))
    if resource.evolution_type == "logistic":
        return float(max(np.float64(0.0), np.float64(params.get("external", 0.0))))
    if resource.evolution_type == "linear":
        return float(max(np.float64(0.0), np.float64(params.get("c", 0.0))))
    return 0.0


def _explicit_inflow_map(world: WorldState) -> Dict[str, float]:
    """Return explicit per-resource exogenous inflow values from metadata."""

    inflows = world.metadata.get("exogenous_inflows", {})
    if not isinstance(inflows, dict):
        inflows = {}
    explicit = {resource_id: float(np.float64(value)) for resource_id, value in inflows.items()}
    conserved_ids = [resource_id for resource_id, resource in world.resources.items() if resource.conserved]
    if "exogenous_inflow" in world.metadata and len(conserved_ids) == 1:
        explicit.setdefault(conserved_ids[0], float(np.float64(world.metadata["exogenous_inflow"])))
    return explicit


def _resource_exogenous_inflow(world: WorldState, resource_id: str) -> float:
    """Return the exogenous inflow allowance for a single resource."""

    explicit = _explicit_inflow_map(world)
    if resource_id in explicit:
        return explicit[resource_id]
    return _infer_resource_exogenous_inflow(world.resources[resource_id])


def level0_check(prev: WorldState, next: WorldState) -> List[Violation]:
    """Run hard invariants that must hold on every simulation step."""

    violations: List[Violation] = []
    for resource_id, next_resource in next.resources.items():
        if not next_resource.conserved:
            continue
        prev_resource = prev.resources[resource_id]
        external = np.float64(_resource_exogenous_inflow(next, resource_id))
        if next_resource.value > prev_resource.value + external + np.float64(EPSILON):
            violations.append(
                Violation(
                    level=0,
                    check_name="conservation",
                    description=(
                        f"Conserved resource {resource_id} grew by "
                        f"{float(next_resource.value - prev_resource.value):.6f} "
                        f"with exogenous allowance {float(external):.6f}"
                    ),
                    severity="hard",
                    details={
                        "resource_id": resource_id,
                        "unit": next_resource.unit,
                        "prev_value": float(prev_resource.value),
                        "next_value": float(next_resource.value),
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
