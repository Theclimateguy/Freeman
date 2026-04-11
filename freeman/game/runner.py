"""Simulation runner."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

from freeman.core.scorer import compute_confidence, score_outcomes
from freeman.core.types import Violation
from freeman.core.transition import step_world
from freeman.core.types import Policy
from freeman.core.world import WorldState
from freeman.exceptions import HardStopException
from freeman.game.result import SimResult
from freeman.verifier.fixed_point import find_fixed_point
from freeman.verifier.level1 import level1_check
from freeman.verifier.level2 import level2_check


@dataclass
class SimConfig:
    """Configuration for a single simulation run."""

    max_steps: int = 50
    dt: float = 1.0
    level2_check_every: int = 5
    level2_shock_delta: float = 0.01
    stop_on_hard_level2: bool = True
    convergence_check_steps: int = 20
    convergence_epsilon: float = 1.0e-4
    fixed_point_max_iter: int = 20
    fixed_point_alpha: float = 0.1
    seed: int = 42


@dataclass
class PreparedSimulationState:
    """Policy-invariant simulation precomputation for one world."""

    world: WorldState
    level1_violations: List[Violation]
    converged: bool
    fixed_point_iters: int


def _distribution_l1_distance(prev: dict[str, float], curr: dict[str, float]) -> float:
    """Return the L1 distance between two categorical distributions."""

    outcome_ids = set(prev) | set(curr)
    return float(sum(abs(float(curr.get(outcome_id, 0.0)) - float(prev.get(outcome_id, 0.0))) for outcome_id in outcome_ids))


class GameRunner:
    """Run the full Freeman simulation loop."""

    def __init__(self, config: SimConfig | None = None) -> None:
        self.config = config or SimConfig()

    def prepare(self, world: WorldState) -> PreparedSimulationState:
        """Run the policy-invariant prechecks once for a world."""

        current = world.clone()
        current.seed = self.config.seed
        l1_violations = level1_check(current, self.config)
        current, fp_converged, fp_iters = find_fixed_point(
            current,
            current.causal_dag,
            max_iter=self.config.fixed_point_max_iter,
            alpha=self.config.fixed_point_alpha,
            base_delta=self.config.level2_shock_delta,
            dt=self.config.dt,
        )
        return PreparedSimulationState(
            world=current,
            level1_violations=list(l1_violations),
            converged=fp_converged,
            fixed_point_iters=fp_iters,
        )

    def run_prepared(
        self,
        prepared: PreparedSimulationState,
        policies: List[Policy],
        *,
        max_steps: int | None = None,
        stability_tol: float | None = None,
        stability_patience: int = 0,
        min_stability_steps: int = 0,
    ) -> SimResult:
        """Run a simulation from a prepared world state."""

        current = prepared.world.clone()
        trajectory = [current.snapshot()]
        all_violations = list(prepared.level1_violations)
        outcome_probs_history = []
        steps_run = 0
        stop_reason = None
        previous_probs: dict[str, float] | None = None
        stable_steps = 0
        rollout_steps = int(self.config.max_steps if max_steps is None else max_steps)
        patience = max(int(stability_patience), 0)
        min_stability_steps = max(int(min_stability_steps), 0)

        for step_idx in range(rollout_steps):
            try:
                current, step_violations = step_world(current, policies, self.config.dt)
            except HardStopException as exc:
                all_violations.extend(exc.violations)
                stop_reason = "hard_level0_violation"
                break

            all_violations.extend(step_violations)
            trajectory.append(current.snapshot())
            probs = score_outcomes(current)
            outcome_probs_history.append(probs)
            steps_run += 1

            if self.config.level2_check_every > 0 and (step_idx + 1) % self.config.level2_check_every == 0:
                l2_violations = level2_check(
                    current,
                    current.causal_dag,
                    base_delta=self.config.level2_shock_delta,
                    dt=self.config.dt,
                )
                all_violations.extend(l2_violations)
                if self.config.stop_on_hard_level2 and any(
                    violation.severity == "hard" for violation in l2_violations
                ):
                    stop_reason = "hard_level2_violation"
                    break

            if (
                stability_tol is not None
                and patience > 0
                and steps_run >= max(min_stability_steps, 1)
                and previous_probs is not None
            ):
                if _distribution_l1_distance(previous_probs, probs) <= float(stability_tol):
                    stable_steps += 1
                    if stable_steps >= patience:
                        stop_reason = "outcome_convergence"
                        break
                else:
                    stable_steps = 0
            previous_probs = probs

        final_outcome_probs = outcome_probs_history[-1] if outcome_probs_history else score_outcomes(current)
        confidence = compute_confidence(final_outcome_probs, all_violations)
        return SimResult(
            domain_id=current.domain_id,
            trajectory=trajectory,
            outcome_probs=outcome_probs_history,
            final_outcome_probs=final_outcome_probs,
            confidence=confidence,
            violations=all_violations,
            converged=prepared.converged,
            steps_run=steps_run,
            metadata={
                "fixed_point_iters": prepared.fixed_point_iters,
                "seed": self.config.seed,
                "stop_reason": stop_reason,
            },
        )

    def run(self, world: WorldState, policies: List[Policy]) -> SimResult:
        """Run a simulation from ``world`` under ``policies``."""

        prepared = self.prepare(world)
        return self.run_prepared(prepared, policies)
