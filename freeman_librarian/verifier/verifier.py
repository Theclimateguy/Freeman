"""Aggregate verifier interface for Levels 0, 1, 2, and org-specific Level 3."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, List, Sequence

from freeman_librarian.core.transition import step_world
from freeman_librarian.core.types import CausalEdge, Violation
from freeman_librarian.core.world import WorldState
from freeman_librarian.exceptions import HardStopException
from freeman_librarian.verifier.fixedpoint import iterate_fixed_point
from freeman_librarian.verifier.level0 import level0_check
from freeman_librarian.verifier.level1 import compute_jacobian, level1_check, spectral_radius
from freeman_librarian.verifier.level2 import level2_check
from freeman_librarian.verifier.level3 import level3_check
from freeman_librarian.verifier.report import VerificationReport


@dataclass
class VerifierConfig:
    """Configuration shared across verifier levels."""

    dt: float = 1.0
    convergence_check_steps: int = 20
    convergence_epsilon: float = 1.0e-4
    shock_delta: float = 0.01
    fixed_point_max_iter: int = 20
    fixed_point_alpha: float = 0.1
    jacobian_guard: float = 1.0 + 1.0e-6

    @classmethod
    def from_any(cls, config: Any | None) -> "VerifierConfig":
        """Adapt a partial config object into a verifier config."""

        if isinstance(config, cls):
            return config
        if config is None:
            return cls()
        return cls(
            dt=float(getattr(config, "dt", cls.dt)),
            convergence_check_steps=int(getattr(config, "convergence_check_steps", cls.convergence_check_steps)),
            convergence_epsilon=float(getattr(config, "convergence_epsilon", cls.convergence_epsilon)),
            shock_delta=float(getattr(config, "level2_shock_delta", getattr(config, "shock_delta", cls.shock_delta))),
            fixed_point_max_iter=int(getattr(config, "fixed_point_max_iter", cls.fixed_point_max_iter)),
            fixed_point_alpha=float(getattr(config, "fixed_point_alpha", cls.fixed_point_alpha)),
            jacobian_guard=float(getattr(config, "jacobian_guard", cls.jacobian_guard)),
        )


class Verifier:
    """Unified interface around the underlying level-specific verifier functions."""

    def __init__(self, config: Any | None = None) -> None:
        self.config = VerifierConfig.from_any(config)

    def level0(self, prev_world: WorldState, next_world: WorldState) -> VerificationReport:
        """Evaluate hard mathematical invariants for one transition."""

        violations = level0_check(prev_world, next_world)
        return self._report(
            world=next_world,
            levels_run=[0],
            violations=violations,
            metadata={"transition_t": int(next_world.t)},
        )

    def level1(self, world: WorldState) -> VerificationReport:
        """Run structural checks plus a causal-DAG sign-consistency precheck."""

        violations = list(level1_check(world, self.config))
        dag_violations = level2_check(
            world,
            world.causal_dag,
            base_delta=self.config.shock_delta,
            dt=self.config.dt,
        )
        for violation in dag_violations:
            if violation.check_name not in {"sign_consistency", "sign_consistency_execution"}:
                continue
            violations.append(
                Violation(
                    level=1,
                    check_name=violation.check_name,
                    description=violation.description,
                    severity=violation.severity,
                    details=violation.details,
                )
            )

        rho = spectral_radius(compute_jacobian(world, dt=self.config.dt))
        return self._report(
            world=world,
            levels_run=[1],
            violations=violations,
            metadata={
                "spectral_radius": float(rho),
                "causal_edges_checked": len(world.causal_dag),
            },
        )

    def level2(
        self,
        world: WorldState,
        causal_dag: Sequence[CausalEdge] | None = None,
    ) -> VerificationReport:
        """Run full sign-checking and bounded correction iterations."""

        dag = list(causal_dag) if causal_dag is not None else list(world.causal_dag)
        initial_violations = level2_check(
            world,
            dag,
            base_delta=self.config.shock_delta,
            dt=self.config.dt,
        )
        sign_violations = [violation for violation in initial_violations if violation.check_name == "sign_consistency"]
        correction_budget = min(
            self.config.fixed_point_max_iter,
            len(sign_violations) + 1 if sign_violations else 0,
        )

        fixed_point_result = None
        violations = list(initial_violations)
        if correction_budget > 0:
            fixed_point_result = iterate_fixed_point(
                world,
                dag,
                max_iter=correction_budget,
                alpha=self.config.fixed_point_alpha,
                base_delta=self.config.shock_delta,
                dt=self.config.dt,
                jacobian_guard=self.config.jacobian_guard,
            )
            violations = fixed_point_result.violations

        rho = spectral_radius(compute_jacobian(world, dt=self.config.dt))
        if rho >= self.config.jacobian_guard:
            violations = [
                Violation(
                    level=2,
                    check_name="spectral_radius_guard",
                    description=(
                        f"Jacobian spectral radius = {rho:.6f} exceeds guard {self.config.jacobian_guard:.6f}"
                    ),
                    severity="hard",
                    details={
                        "field": "jacobian.spectral_radius",
                        "observed": float(rho),
                        "expected_max": float(self.config.jacobian_guard),
                    },
                ),
                *violations,
            ]

        return self._report(
            world=world,
            levels_run=[2],
            violations=violations,
            metadata={
                "dag_source": "override" if causal_dag is not None else "world",
                "initial_sign_violations": len(sign_violations),
                "correction_iterations_budget": int(correction_budget),
                "correction_iterations_run": int(fixed_point_result.iterations) if fixed_point_result else 0,
                "fixed_point_converged": bool(fixed_point_result.converged) if fixed_point_result else True,
                "spectral_radius": float(rho),
            },
        )

    def level3(self, world: WorldState) -> VerificationReport:
        """Run organization-specific invariants on top of the compiled world."""

        violations = list(level3_check(world))
        return self._report(
            world=world,
            levels_run=[3],
            violations=violations,
            metadata={
                "org_conflict_count": len(world.metadata.get("org", {}).get("conflicts", [])),
                "actor_count": len(world.actors),
                "resource_count": len(world.resources),
            },
        )

    def run(
        self,
        world: WorldState,
        *,
        levels: Iterable[int] = (1, 2),
        prev_world: WorldState | None = None,
        next_world: WorldState | None = None,
        causal_dag: Sequence[CausalEdge] | None = None,
    ) -> VerificationReport:
        """Run multiple verifier levels and return one aggregated report."""

        ordered_levels = list(levels)
        violations: List[Violation] = []
        metadata: dict[str, Any] = {}

        for level in ordered_levels:
            if level == 0:
                baseline_prev = prev_world or world
                if next_world is not None:
                    baseline_next = next_world
                    step_violations = level0_check(baseline_prev, baseline_next)
                else:
                    try:
                        baseline_next, step_violations = step_world(baseline_prev.clone(), [])
                    except HardStopException as exc:
                        baseline_next = baseline_prev.clone()
                        step_violations = exc.violations
                violations.extend(step_violations)
                metadata["level0_transition_t"] = int(baseline_next.t)
            elif level == 1:
                level1_report = self.level1(world)
                violations.extend(level1_report.violations)
                metadata["level1"] = level1_report.metadata
            elif level == 2:
                level2_report = self.level2(world, causal_dag=causal_dag)
                violations.extend(level2_report.violations)
                metadata["level2"] = level2_report.metadata
            elif level == 3:
                level3_report = self.level3(world)
                violations.extend(level3_report.violations)
                metadata["level3"] = level3_report.metadata
            else:
                raise ValueError(f"Unsupported verifier level: {level}")

        return self._report(world=world, levels_run=ordered_levels, violations=violations, metadata=metadata)

    def _report(
        self,
        *,
        world: WorldState,
        levels_run: List[int],
        violations: List[Violation],
        metadata: dict[str, Any],
    ) -> VerificationReport:
        """Create a standardized verification report."""

        hard_failures = any(violation.severity == "hard" for violation in violations)
        return VerificationReport(
            world_id=f"{world.domain_id}:{world.t}",
            domain_id=world.domain_id,
            levels_run=levels_run,
            violations=violations,
            passed=not hard_failures,
            metadata=metadata,
        )


__all__ = ["Verifier", "VerifierConfig"]
