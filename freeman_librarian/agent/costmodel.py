"""Formal task cost model and budget governance."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
import json
from pathlib import Path
from collections.abc import Callable
from typing import Any


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
    embedding_tokens_used: int = 0
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


@dataclass
class BudgetLedgerEntry:
    """One persisted budget-governance event."""

    timestamp: str
    task_id: str
    task_type: str
    requested_mode: str
    approved_mode: str
    allowed: bool
    estimated_cost: float
    actual_cost: float
    budget_remaining: float
    stop_reason: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def snapshot(self) -> dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "task_id": self.task_id,
            "task_type": self.task_type,
            "requested_mode": self.requested_mode,
            "approved_mode": self.approved_mode,
            "allowed": bool(self.allowed),
            "estimated_cost": float(self.estimated_cost),
            "actual_cost": float(self.actual_cost),
            "budget_remaining": float(self.budget_remaining),
            "stop_reason": self.stop_reason,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_snapshot(cls, payload: dict[str, Any]) -> "BudgetLedgerEntry":
        return cls(
            timestamp=str(payload.get("timestamp", "")),
            task_id=str(payload.get("task_id", "")),
            task_type=str(payload.get("task_type", "")),
            requested_mode=str(payload.get("requested_mode", "WATCH")),
            approved_mode=str(payload.get("approved_mode", "WATCH")),
            allowed=bool(payload.get("allowed", False)),
            estimated_cost=float(payload.get("estimated_cost", 0.0)),
            actual_cost=float(payload.get("actual_cost", 0.0)),
            budget_remaining=float(payload.get("budget_remaining", 0.0)),
            stop_reason=payload.get("stop_reason"),
            metadata=dict(payload.get("metadata", {})),
        )


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
        embedding_tokens_used: int = 0,
    ) -> CostEstimate:
        estimated_cost = (
            0.015 * llm_calls
            + 0.001 * sim_steps
            + 0.0005 * actors
            + 0.0005 * resources
            + 0.002 * domains
            + 0.0005 * kg_updates
            + 0.0000002 * embedding_tokens_used
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
            embedding_tokens_used=embedding_tokens_used,
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


class BudgetLedger:
    """Append-only runtime budget ledger with cached spend summary."""

    def __init__(self, path: str | Path, *, policy: BudgetPolicy, auto_load: bool = True) -> None:
        self.path = Path(path).resolve()
        self.policy = policy
        self.entries: list[BudgetLedgerEntry] = []
        self.spent_usd: float = 0.0
        if auto_load and self.path.exists():
            self.load()

    def load(self) -> None:
        self.entries = []
        self.spent_usd = 0.0
        if not self.path.exists():
            return
        for line in self.path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            entry = BudgetLedgerEntry.from_snapshot(json.loads(line))
            self.entries.append(entry)
            self.spent_usd += float(entry.actual_cost)

    def append(self, entry: BudgetLedgerEntry) -> BudgetLedgerEntry:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(entry.snapshot(), ensure_ascii=False) + "\n")
        self.entries.append(entry)
        self.spent_usd += float(entry.actual_cost)
        return entry

    def record(
        self,
        *,
        task_type: str,
        requested_mode: str,
        decision: BudgetDecision,
        actual_cost: float | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> BudgetLedgerEntry:
        realized_cost = float(actual_cost if actual_cost is not None else (decision.estimate.estimated_cost if decision.allowed else 0.0))
        updated_estimate = CostModel(self.policy).record_actual(
            decision.estimate,
            realized_cost,
            budget_spent=self.spent_usd,
        )
        entry = BudgetLedgerEntry(
            timestamp=datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
            task_id=str(updated_estimate.task_id),
            task_type=str(task_type),
            requested_mode=str(requested_mode),
            approved_mode=str(decision.approved_mode),
            allowed=bool(decision.allowed),
            estimated_cost=float(updated_estimate.estimated_cost),
            actual_cost=float(updated_estimate.actual_cost),
            budget_remaining=float(updated_estimate.budget_remaining),
            stop_reason=updated_estimate.stop_reason,
            metadata=dict(metadata or {}),
        )
        return self.append(entry)

    def summary(self) -> dict[str, Any]:
        by_task_type: dict[str, int] = {}
        stop_reasons: dict[str, int] = {}
        allowed_count = 0
        blocked_count = 0
        for entry in self.entries:
            by_task_type[entry.task_type] = by_task_type.get(entry.task_type, 0) + 1
            effectively_allowed = bool(entry.allowed) and (
                str(entry.approved_mode).upper() != "WATCH" or str(entry.requested_mode).upper() == "WATCH"
            )
            if effectively_allowed:
                allowed_count += 1
            else:
                blocked_count += 1
            if entry.stop_reason:
                stop_reasons[entry.stop_reason] = stop_reasons.get(entry.stop_reason, 0) + 1
        remaining = self.policy.max_compute_budget_per_session - float(self.spent_usd)
        return {
            "tracking_enabled": True,
            "ledger_path": str(self.path),
            "configured_usd_per_day": float(self.policy.max_compute_budget_per_session),
            "spent_usd": float(self.spent_usd),
            "remaining_usd": float(remaining),
            "entry_count": len(self.entries),
            "allowed_count": allowed_count,
            "blocked_count": blocked_count,
            "by_task_type": by_task_type,
            "stop_reasons": stop_reasons,
        }


def build_budget_policy(config: dict[str, Any]) -> BudgetPolicy:
    """Build one budget policy from config defaults plus runtime overrides."""

    agent_cfg = dict(config.get("agent", {}))
    governance_cfg = dict(agent_cfg.get("cost_governance", {}))
    configured_budget = float(governance_cfg.get("max_compute_budget_per_session", agent_cfg.get("budget_usd_per_day", BudgetPolicy().max_compute_budget_per_session)))
    return BudgetPolicy(
        max_llm_calls_per_task=int(governance_cfg.get("max_llm_calls_per_task", BudgetPolicy().max_llm_calls_per_task)),
        max_sim_steps_per_task=int(governance_cfg.get("max_sim_steps_per_task", BudgetPolicy().max_sim_steps_per_task)),
        max_domains_per_task=int(governance_cfg.get("max_domains_per_task", BudgetPolicy().max_domains_per_task)),
        max_deep_dive_depth=int(governance_cfg.get("max_deep_dive_depth", BudgetPolicy().max_deep_dive_depth)),
        max_compute_budget_per_session=float(configured_budget),
    )


def budget_tracking_enabled(config: dict[str, Any]) -> bool:
    """Return whether runtime budget governance is enabled by config."""

    agent_cfg = dict(config.get("agent", {}))
    governance_cfg = dict(agent_cfg.get("cost_governance", {}))
    if "enabled" in governance_cfg:
        return bool(governance_cfg.get("enabled"))
    return float(agent_cfg.get("budget_usd_per_day", 0.0)) > 0.0


def resolve_budget_decision(
    *,
    cost_model: CostModel,
    requested_mode: str,
    estimate_for_mode: Callable[[str], CostEstimate],
    budget_spent: float,
    deep_dive_depth: int = 0,
) -> BudgetDecision:
    """Resolve budget gating across downgraded modes until stable."""

    current_mode = str(requested_mode or "WATCH")
    downgrade_reason: str | None = None
    visited_modes: set[str] = set()

    while True:
        estimate = estimate_for_mode(current_mode)
        decision = cost_model.precheck(
            requested_mode=current_mode,
            estimate=estimate,
            budget_spent=budget_spent,
            deep_dive_depth=deep_dive_depth,
        )
        if decision.stop_reason and downgrade_reason is None:
            downgrade_reason = str(decision.stop_reason)
        approved_mode = str(decision.approved_mode or current_mode)
        if not decision.allowed or approved_mode == current_mode or approved_mode in visited_modes:
            if downgrade_reason and decision.stop_reason is None:
                decision.stop_reason = downgrade_reason
                decision.estimate.stop_reason = downgrade_reason
            return decision
        visited_modes.add(current_mode)
        current_mode = approved_mode


__all__ = [
    "BudgetDecision",
    "BudgetLedger",
    "BudgetLedgerEntry",
    "BudgetPolicy",
    "CostEstimate",
    "CostModel",
    "build_budget_policy",
    "budget_tracking_enabled",
    "resolve_budget_decision",
]
