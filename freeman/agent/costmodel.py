"""Formal task cost model and budget governance."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class CostEstimate:
    """Estimated and realized compute cost for one task."""

    task_id: str
    estimated_cost: float
    llm_calls: int
    sim_steps: int
    actors: int
    resources: int
    domains: int
    kg_updates: int
    actual_cost: float = 0.0
    budget_remaining: float = 0.0
    stop_reason: str | None = None


@dataclass
class BudgetPolicy:
    """Hard budget limits for v0.2 governance."""

    max_llm_calls_per_task: int = 5
    max_sim_steps_per_task: int = 100
    max_domains_per_task: int = 4
    max_deep_dive_depth: int = 2
    max_compute_budget_per_session: float = 500.0


@dataclass
class BudgetDecision:
    """Outcome of a pre-check against the budget policy."""

    approved_mode: str
    allowed: bool
    stop_reason: str | None
    estimate: CostEstimate


class CostModel:
    """Estimate task cost and downgrade/stop on budget pressure."""

    def __init__(self, policy: BudgetPolicy | None = None) -> None:
        self.policy = policy or BudgetPolicy()

    def estimate(
        self,
        *,
        task_id: str,
        llm_calls: int,
        sim_steps: int,
        actors: int,
        resources: int,
        domains: int = 1,
        kg_updates: int = 0,
    ) -> CostEstimate:
        estimated_cost = (
            8.0 * llm_calls
            + 1.5 * sim_steps
            + 0.25 * actors
            + 0.25 * resources
            + 6.0 * domains
            + 0.5 * kg_updates
        )
        return CostEstimate(
            task_id=task_id,
            estimated_cost=float(estimated_cost),
            llm_calls=llm_calls,
            sim_steps=sim_steps,
            actors=actors,
            resources=resources,
            domains=domains,
            kg_updates=kg_updates,
        )

    def precheck(
        self,
        *,
        requested_mode: str,
        estimate: CostEstimate,
        budget_spent: float,
        deep_dive_depth: int = 0,
    ) -> BudgetDecision:
        """Approve, downgrade, or stop before running a task."""

        remaining = self.policy.max_compute_budget_per_session - float(budget_spent)
        estimate.budget_remaining = remaining

        if estimate.llm_calls > self.policy.max_llm_calls_per_task:
            estimate.stop_reason = "max_llm_calls_per_task"
            return BudgetDecision("WATCH", False, estimate.stop_reason, estimate)
        if estimate.sim_steps > self.policy.max_sim_steps_per_task:
            estimate.stop_reason = "max_sim_steps_per_task"
            return BudgetDecision("WATCH", False, estimate.stop_reason, estimate)
        if estimate.domains > self.policy.max_domains_per_task:
            estimate.stop_reason = "max_domains_per_task"
            return BudgetDecision("WATCH", False, estimate.stop_reason, estimate)

        approved_mode = requested_mode
        if requested_mode == "DEEP_DIVE" and deep_dive_depth >= self.policy.max_deep_dive_depth:
            approved_mode = "ANALYZE"
            estimate.stop_reason = "max_deep_dive_depth"

        if estimate.estimated_cost > remaining:
            if approved_mode == "DEEP_DIVE":
                approved_mode = "ANALYZE"
                estimate.stop_reason = "budget_exhaustion_downgrade"
            elif approved_mode == "ANALYZE":
                approved_mode = "WATCH"
                estimate.stop_reason = "budget_exhaustion_downgrade"
            else:
                estimate.stop_reason = "budget_exhaustion_stop"
                return BudgetDecision(approved_mode, False, estimate.stop_reason, estimate)

        return BudgetDecision(approved_mode, True, estimate.stop_reason, estimate)

    def record_actual(self, estimate: CostEstimate, actual_cost: float, *, budget_spent: float) -> CostEstimate:
        estimate.actual_cost = float(actual_cost)
        estimate.budget_remaining = self.policy.max_compute_budget_per_session - float(budget_spent) - float(actual_cost)
        if estimate.budget_remaining < 0.0 and estimate.stop_reason is None:
            estimate.stop_reason = "budget_exhaustion_stop"
        return estimate


__all__ = ["BudgetDecision", "BudgetPolicy", "CostEstimate", "CostModel"]
