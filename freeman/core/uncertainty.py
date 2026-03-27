"""Monte Carlo uncertainty propagation for Freeman."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List

import numpy as np

from freeman.core.scorer import score_outcomes
from freeman.core.world import WorldState
from freeman.game.runner import GameRunner, SimConfig


@dataclass
class ParameterDistribution:
    """Distribution attached to a dotted model parameter path."""

    path: str
    distribution_type: str
    params: Dict[str, Any]

    def sample(self, rng: np.random.Generator) -> float:
        if self.distribution_type == "normal":
            return float(rng.normal(self.params["mean"], self.params["std"]))
        if self.distribution_type == "uniform":
            return float(rng.uniform(self.params["low"], self.params["high"]))
        if self.distribution_type == "discrete":
            values = self.params["values"]
            probabilities = self.params.get("probabilities")
            return float(rng.choice(values, p=probabilities))
        raise ValueError(f"Unsupported distribution_type: {self.distribution_type}")


@dataclass
class ScenarioSample:
    """One Monte Carlo scenario."""

    sample_id: str
    parameter_values: Dict[str, float]
    outcome_probs: Dict[str, float]


@dataclass
class OutcomeDistribution:
    """Distribution of outcome probabilities across scenarios."""

    samples: List[ScenarioSample]
    mean_probs: Dict[str, float]
    intervals: Dict[str, Dict[str, float]]


@dataclass
class ConfidenceReport:
    """Confidence derived from outcome-probability variance."""

    confidence: float
    variance: Dict[str, float]
    stable: bool
    metadata: Dict[str, Any] = field(default_factory=dict)


def _set_snapshot_value(snapshot: Dict[str, Any], path: str, value: float) -> None:
    parts = path.split(".")
    cursor: Any = snapshot
    for part in parts[:-1]:
        if part.isdigit():
            cursor = cursor[int(part)]
        else:
            cursor = cursor[part]
    final_part = parts[-1]
    if final_part.isdigit():
        cursor[int(final_part)] = value
    else:
        cursor[final_part] = value


class UncertaintyEngine:
    """Run Monte Carlo sampling and derive confidence from variance."""

    def __init__(self, sim_config: SimConfig | None = None) -> None:
        self.sim_config = sim_config or SimConfig(max_steps=0)

    def monte_carlo(
        self,
        world: WorldState,
        distributions: Iterable[ParameterDistribution],
        *,
        monte_carlo_samples: int = 100,
        policies: Iterable[Any] = (),
        seed: int = 42,
    ) -> OutcomeDistribution:
        rng = np.random.default_rng(seed)
        scenarios: List[ScenarioSample] = []
        outcome_matrix: Dict[str, List[float]] = {}

        for sample_index in range(int(monte_carlo_samples)):
            snapshot = world.snapshot()
            sampled_values: Dict[str, float] = {}
            for distribution in distributions:
                sampled_value = distribution.sample(rng)
                _set_snapshot_value(snapshot, distribution.path, sampled_value)
                sampled_values[distribution.path] = sampled_value
            sampled_world = WorldState.from_snapshot(snapshot)

            if self.sim_config.max_steps <= 0:
                outcome_probs = score_outcomes(sampled_world)
            else:
                simulation = GameRunner(self.sim_config).run(sampled_world, list(policies))
                outcome_probs = simulation.final_outcome_probs

            for outcome_id, probability in outcome_probs.items():
                outcome_matrix.setdefault(outcome_id, []).append(float(probability))
            scenarios.append(
                ScenarioSample(
                    sample_id=f"scenario_{sample_index + 1}",
                    parameter_values=sampled_values,
                    outcome_probs=outcome_probs,
                )
            )

        mean_probs = {
            outcome_id: float(np.mean(values, dtype=np.float64))
            for outcome_id, values in outcome_matrix.items()
        }
        intervals = {
            outcome_id: {
                "p05": float(np.quantile(values, 0.05)),
                "p50": float(np.quantile(values, 0.50)),
                "p95": float(np.quantile(values, 0.95)),
            }
            for outcome_id, values in outcome_matrix.items()
        }
        return OutcomeDistribution(samples=scenarios, mean_probs=mean_probs, intervals=intervals)

    def confidence_from_variance(
        self,
        distribution: OutcomeDistribution,
        *,
        uncertainty_threshold: float = 0.05,
    ) -> ConfidenceReport:
        variance = {
            outcome_id: float(np.var([sample.outcome_probs.get(outcome_id, 0.0) for sample in distribution.samples]))
            for outcome_id in distribution.mean_probs
        }
        mean_variance = float(np.mean(list(variance.values()), dtype=np.float64)) if variance else 0.0
        confidence = float(max(0.0, min(1.0, 1.0 - np.sqrt(mean_variance))))
        return ConfidenceReport(
            confidence=confidence,
            variance=variance,
            stable=mean_variance <= uncertainty_threshold,
            metadata={"mean_variance": mean_variance, "uncertainty_threshold": uncertainty_threshold},
        )


__all__ = [
    "ConfidenceReport",
    "OutcomeDistribution",
    "ParameterDistribution",
    "ScenarioSample",
    "UncertaintyEngine",
]
