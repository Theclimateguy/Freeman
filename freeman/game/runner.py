"""Simulation runner."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

from freeman.core.scorer import compute_confidence, score_outcomes
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
    convergence_check_steps: int = 20
    convergence_epsilon: float = 1.0e-4
    fixed_point_max_iter: int = 20
    fixed_point_alpha: float = 0.1
    seed: int = 42


class GameRunner:
    """Run the full Freeman simulation loop."""

    def __init__(self, config: SimConfig | None = None) -> None:
        self.config = config or SimConfig()

    def run(self, world: WorldState, policies: List[Policy]) -> SimResult:
        """Run a simulation from ``world`` under ``policies``."""

        current = world.clone()
        current.seed = self.config.seed
        l1_violations = level1_check(current, self.config)
        current, fp_converged, fp_iters = find_fixed_point(
            current,
            current.causal_dag,
            max_iter=self.config.fixed_point_max_iter,
            alpha=self.config.fixed_point_alpha,
        )

        trajectory = [current.snapshot()]
        all_violations = list(l1_violations)
        outcome_probs_history = []
        steps_run = 0

        for step_idx in range(self.config.max_steps):
            try:
                current, step_violations = step_world(current, policies, self.config.dt)
            except HardStopException as exc:
                all_violations.extend(exc.violations)
                break

            all_violations.extend(step_violations)
            trajectory.append(current.snapshot())
            probs = score_outcomes(current)
            outcome_probs_history.append(probs)
            steps_run += 1

            if self.config.level2_check_every > 0 and (step_idx + 1) % self.config.level2_check_every == 0:
                all_violations.extend(level2_check(current, current.causal_dag))

        final_outcome_probs = outcome_probs_history[-1] if outcome_probs_history else score_outcomes(current)
        confidence = compute_confidence(final_outcome_probs, all_violations)
        return SimResult(
            domain_id=current.domain_id,
            trajectory=trajectory,
            outcome_probs=outcome_probs_history,
            final_outcome_probs=final_outcome_probs,
            confidence=confidence,
            violations=all_violations,
            converged=fp_converged,
            steps_run=steps_run,
            metadata={"fixed_point_iters": fp_iters, "seed": self.config.seed},
        )
