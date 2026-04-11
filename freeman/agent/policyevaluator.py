"""Counterfactual policy evaluation on top of the deterministic simulator."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from typing import Any, Iterable, List, Sequence

from freeman.core.scorer import raw_outcome_scores
from freeman.core.types import Policy
from freeman.core.world import WorldState
from freeman.game.runner import GameRunner, SimConfig
from freeman.memory.epistemiclog import EpistemicLog


def _coerce_policy(policy_like: Policy | dict[str, Any]) -> Policy:
    """Normalize one policy payload."""

    if isinstance(policy_like, Policy):
        return policy_like
    if isinstance(policy_like, dict) and "actor_id" in policy_like:
        return Policy.from_snapshot(policy_like)
    raise TypeError("Each policy candidate must be a Policy or a policy snapshot with actor_id/actions.")


def _coerce_policy_bundle(candidate: Any) -> tuple[List[Policy], dict[str, Any]]:
    """Normalize one candidate into a policy bundle plus metadata."""

    if isinstance(candidate, Policy):
        return [candidate], {}
    if isinstance(candidate, dict):
        if "actor_id" in candidate:
            return [Policy.from_snapshot(candidate)], {}
        if "policies" in candidate:
            payload = dict(candidate)
            raw_policies = payload.pop("policies")
            if not isinstance(raw_policies, Sequence) or isinstance(raw_policies, (str, bytes)):
                raise TypeError("Candidate 'policies' must be a sequence of policy payloads.")
            return [_coerce_policy(policy_like) for policy_like in raw_policies], payload
        raise TypeError("Policy candidate dict must be a policy snapshot or include a 'policies' field.")
    if isinstance(candidate, Sequence) and not isinstance(candidate, (str, bytes)):
        return [_coerce_policy(policy_like) for policy_like in candidate], {}
    raise TypeError("Unsupported policy candidate type.")


def _expected_utility(outcome_probs: dict[str, float], raw_scores: dict[str, float]) -> float:
    """Return a scale-normalized expected outcome utility."""

    if not outcome_probs or not raw_scores:
        return 0.0
    score_scale = sum(abs(float(score)) for score in raw_scores.values()) / max(len(raw_scores), 1)
    score_scale = max(score_scale, 1.0e-8)
    return float(
        sum(float(outcome_probs.get(outcome_id, 0.0)) * float(score) for outcome_id, score in raw_scores.items())
        / score_scale
    )


@dataclass
class PolicyEvalResult:
    """Ranked result for one counterfactual policy bundle."""

    policies: List[Policy]
    dominant_outcome: str | None
    outcome_probs: dict[str, float]
    confidence: float
    expected_utility: float
    epistemic_weight: float
    epistemic_score: float
    hard_violations: int
    soft_violations: int
    rank: int = 0
    steps_run: int = 0
    stop_reason: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def policy(self) -> Policy | None:
        """Backward-compatible access for single-policy candidates."""

        if len(self.policies) == 1:
            return self.policies[0]
        return None

    def snapshot(self) -> dict[str, Any]:
        """Return a JSON-serializable result payload."""

        return {
            "rank": int(self.rank),
            "policies": [policy.snapshot() for policy in self.policies],
            "dominant_outcome": self.dominant_outcome,
            "outcome_probs": dict(self.outcome_probs),
            "confidence": float(self.confidence),
            "expected_utility": float(self.expected_utility),
            "epistemic_weight": float(self.epistemic_weight),
            "epistemic_score": float(self.epistemic_score),
            "hard_violations": int(self.hard_violations),
            "soft_violations": int(self.soft_violations),
            "steps_run": int(self.steps_run),
            "stop_reason": self.stop_reason,
            "metadata": dict(self.metadata),
        }


class PolicyEvaluator:
    """Short-horizon deterministic policy planner.

    The evaluator treats hard verifier violations as feasibility constraints and ranks
    feasible branches by epistemically weighted expected utility. To keep planning
    affordable, it reuses the world preparation that is invariant across policy
    candidates and limits the rollout horizon to a short branch-comparison window.
    """

    def __init__(
        self,
        *,
        sim_config: SimConfig | None = None,
        epistemic_log: EpistemicLog | None = None,
        planning_horizon: int | None = None,
        max_candidates: int = 8,
        stability_tol: float = 1.0e-3,
        stability_patience: int = 2,
        min_stability_steps: int = 3,
    ) -> None:
        base_config = sim_config or SimConfig(max_steps=12)
        horizon = int(planning_horizon or min(base_config.max_steps, 8))
        horizon = max(1, min(horizon, int(base_config.max_steps)))
        self.sim_config = SimConfig(
            max_steps=horizon,
            dt=float(base_config.dt),
            level2_check_every=int(base_config.level2_check_every),
            level2_shock_delta=float(base_config.level2_shock_delta),
            stop_on_hard_level2=bool(base_config.stop_on_hard_level2),
            convergence_check_steps=int(base_config.convergence_check_steps),
            convergence_epsilon=float(base_config.convergence_epsilon),
            fixed_point_max_iter=int(base_config.fixed_point_max_iter),
            fixed_point_alpha=float(base_config.fixed_point_alpha),
            seed=int(base_config.seed),
        )
        self.epistemic_log = epistemic_log
        self.max_candidates = max(int(max_candidates), 1)
        self.stability_tol = float(stability_tol)
        self.stability_patience = max(int(stability_patience), 0)
        self.min_stability_steps = max(int(min_stability_steps), 0)
        self.runner = GameRunner(self.sim_config)

    def evaluate(
        self,
        world: WorldState,
        candidate_policies: Iterable[Any],
    ) -> List[PolicyEvalResult]:
        """Run short deterministic counterfactual rollouts and rank the results."""

        normalized_candidates = self._normalize_candidates(candidate_policies)
        if not normalized_candidates:
            return []

        prepared = self.runner.prepare(world)
        epistemic_weight = self.epistemic_log.domain_weight(world.domain_id) if self.epistemic_log is not None else 1.0
        results: List[PolicyEvalResult] = []
        for candidate_index, (policy_bundle, candidate_metadata) in enumerate(normalized_candidates, start=1):
            sim_result = self.runner.run_prepared(
                prepared,
                policy_bundle,
                max_steps=self.sim_config.max_steps,
                stability_tol=self.stability_tol,
                stability_patience=self.stability_patience,
                min_stability_steps=self.min_stability_steps,
            )
            final_world = WorldState.from_snapshot(sim_result.trajectory[-1])
            outcome_probs = dict(sim_result.final_outcome_probs)
            dominant_outcome = max(outcome_probs, key=outcome_probs.get) if outcome_probs else None
            utility = _expected_utility(outcome_probs, raw_outcome_scores(final_world))
            hard_violations = sum(1 for violation in sim_result.violations if violation.severity == "hard")
            soft_violations = sum(1 for violation in sim_result.violations if violation.severity == "soft")
            epistemic_score = float(epistemic_weight * utility)
            results.append(
                PolicyEvalResult(
                    policies=[Policy.from_snapshot(policy.snapshot()) for policy in policy_bundle],
                    dominant_outcome=dominant_outcome,
                    outcome_probs=outcome_probs,
                    confidence=float(sim_result.confidence),
                    expected_utility=float(utility),
                    epistemic_weight=float(epistemic_weight),
                    epistemic_score=epistemic_score,
                    hard_violations=hard_violations,
                    soft_violations=soft_violations,
                    steps_run=int(sim_result.steps_run),
                    stop_reason=str(sim_result.metadata.get("stop_reason")) if sim_result.metadata.get("stop_reason") else None,
                    metadata={
                        "candidate_index": candidate_index,
                        "prepared_fixed_point_iters": int(prepared.fixed_point_iters),
                        "planning_horizon": int(self.sim_config.max_steps),
                        **candidate_metadata,
                    },
                )
            )

        results.sort(
            key=lambda item: (
                item.hard_violations,
                item.soft_violations,
                -item.epistemic_score,
                -item.confidence,
                -item.expected_utility,
            )
        )
        for rank, result in enumerate(results, start=1):
            result.rank = rank
        return results

    def _normalize_candidates(self, candidate_policies: Iterable[Any]) -> List[tuple[List[Policy], dict[str, Any]]]:
        """Deduplicate and bound the candidate list."""

        normalized: List[tuple[List[Policy], dict[str, Any]]] = []
        seen_keys: set[str] = set()
        for candidate in candidate_policies:
            policy_bundle, metadata = _coerce_policy_bundle(candidate)
            if not policy_bundle:
                continue
            key = json.dumps([policy.snapshot() for policy in policy_bundle], sort_keys=True)
            if key in seen_keys:
                continue
            seen_keys.add(key)
            normalized.append((policy_bundle, metadata))
            if len(normalized) >= self.max_candidates:
                break
        return normalized


__all__ = ["PolicyEvalResult", "PolicyEvaluator"]
