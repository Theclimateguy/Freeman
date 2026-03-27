"""Structured proactive events emitted from pipeline results."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, List

if TYPE_CHECKING:
    from freeman.agent.analysispipeline import AnalysisPipelineResult


@dataclass
class ProactiveEvent:
    """One interface-facing proactive signal from the agent."""

    event_type: str
    domain_id: str
    outcome_id: str | None
    message: str
    severity: float
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.severity = float(min(max(self.severity, 0.0), 1.0))


class ProactiveEmitter:
    """Generate human-facing events from material pipeline changes."""

    def __init__(
        self,
        *,
        forecast_shift_threshold: float = 0.15,
        confidence_floor: float = 0.4,
    ) -> None:
        self.forecast_shift_threshold = float(forecast_shift_threshold)
        self.confidence_floor = float(confidence_floor)

    def evaluate(
        self,
        result: AnalysisPipelineResult,
        prev_probs: dict[str, float] | None = None,
    ) -> List[ProactiveEvent]:
        events: List[ProactiveEvent] = []
        domain_id = result.world.domain_id
        hard_violations = [
            violation
            for violation in result.simulation.get("violations", [])
            if violation.get("severity") == "hard"
        ]
        if hard_violations:
            events.append(
                ProactiveEvent(
                    event_type="alert",
                    domain_id=domain_id,
                    outcome_id=None,
                    message=f"Hard verifier violations detected in {domain_id}.",
                    severity=1.0,
                    metadata={"count": len(hard_violations), "violations": hard_violations},
                )
            )

        current_probs = result.simulation.get("final_outcome_probs", {}) or {}
        if prev_probs and current_probs:
            for outcome_id, current_prob in current_probs.items():
                previous_prob = float(prev_probs.get(outcome_id, current_prob))
                current_prob = float(current_prob)
                delta = current_prob - previous_prob
                if abs(delta) >= self.forecast_shift_threshold:
                    events.append(
                        ProactiveEvent(
                            event_type="forecast_update",
                            domain_id=domain_id,
                            outcome_id=outcome_id,
                            message=(
                                f"Outcome {outcome_id} shifted from {previous_prob:.3f} "
                                f"to {current_prob:.3f} in {domain_id}."
                            ),
                            severity=min(abs(delta), 1.0),
                            metadata={
                                "previous_prob": previous_prob,
                                "current_prob": current_prob,
                                "delta": delta,
                            },
                        )
                    )

        confidence = float(result.simulation.get("confidence", 1.0))
        if confidence < self.confidence_floor:
            events.append(
                ProactiveEvent(
                    event_type="question_to_human",
                    domain_id=domain_id,
                    outcome_id=result.dominant_outcome,
                    message=f"Confidence {confidence:.3f} is below the review floor for {domain_id}.",
                    severity=min(max(self.confidence_floor - confidence, 0.0) / max(self.confidence_floor, 1.0e-8), 1.0),
                    metadata={"confidence": confidence, "confidence_floor": self.confidence_floor},
                )
            )

        return events


__all__ = ["ProactiveEmitter", "ProactiveEvent"]
