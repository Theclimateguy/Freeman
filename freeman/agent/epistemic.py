"""Helpers for epistemic logging, disagreement tracking, and belief conflicts."""

from __future__ import annotations

from typing import Any, Dict

from freeman.agent.forecastregistry import Forecast
from freeman.core.world import WorldState
from freeman.memory.knowledgegraph import KGNode


def extract_reference_outcome_probs(world: WorldState) -> Dict[str, float]:
    """Return an external/reference probability surface when the world exposes one."""

    explicit = world.metadata.get("reference_outcome_probs")
    if isinstance(explicit, dict):
        return {
            str(outcome_id): float(probability)
            for outcome_id, probability in explicit.items()
            if outcome_id in world.outcomes
        }

    cutoff_probability = world.metadata.get("cutoff_probability")
    if (
        cutoff_probability is not None
        and "yes" in world.outcomes
        and "no" in world.outcomes
    ):
        yes_probability = float(cutoff_probability)
        return {"yes": yes_probability, "no": 1.0 - yes_probability}
    return {}


def compute_confidence_weighted_disagreement(
    predicted_probs: Dict[str, float],
    reference_probs: Dict[str, float],
    *,
    belief_confidence: float,
) -> Dict[str, Dict[str, Any]]:
    """Return a disagreement snapshot against an external/reference belief surface."""

    snapshot: Dict[str, Dict[str, Any]] = {}
    for outcome_id, predicted_prob in predicted_probs.items():
        if outcome_id not in reference_probs:
            continue
        reference_prob = float(reference_probs[outcome_id])
        signed_gap = float(predicted_prob) - reference_prob
        weighted_gap = float(signed_gap * float(belief_confidence))
        if abs(signed_gap) <= 1.0e-12:
            stance = "aligned"
        else:
            stance = "above_reference" if signed_gap > 0.0 else "below_reference"
        snapshot[outcome_id] = {
            "predicted_prob": float(predicted_prob),
            "reference_prob": reference_prob,
            "signed_gap": signed_gap,
            "belief_confidence": float(belief_confidence),
            "confidence_weighted_gap": weighted_gap,
            "stance": stance,
        }
    return snapshot


def summarize_primary_disagreement(
    disagreement_snapshot: Dict[str, Dict[str, Any]],
) -> tuple[str, Dict[str, Any]] | None:
    """Return the strongest disagreement outcome by weighted absolute gap."""

    if not disagreement_snapshot:
        return None
    outcome_id = max(
        disagreement_snapshot,
        key=lambda key: abs(float(disagreement_snapshot[key].get("confidence_weighted_gap", 0.0))),
    )
    return outcome_id, disagreement_snapshot[outcome_id]


def build_disagreement_node(
    *,
    domain_id: str,
    step: int,
    disagreement_snapshot: Dict[str, Dict[str, Any]],
    threshold: float = 0.05,
) -> KGNode | None:
    """Create a KG node for a meaningful external disagreement."""

    primary = summarize_primary_disagreement(disagreement_snapshot)
    if primary is None:
        return None
    outcome_id, payload = primary
    weighted_gap = float(payload.get("confidence_weighted_gap", 0.0))
    if abs(weighted_gap) < float(threshold):
        return None
    signed_gap = float(payload.get("signed_gap", 0.0))
    return KGNode(
        id=f"disagreement:{domain_id}:{step}",
        label=f"Belief Disagreement {domain_id}",
        node_type="belief_disagreement",
        content=(
            f"Outcome {outcome_id} is {signed_gap:+.4f} away from reference "
            f"with confidence-weighted gap {weighted_gap:+.4f}."
        ),
        confidence=min(max(abs(weighted_gap), 0.15), 0.95),
        metadata={
            "domain_id": domain_id,
            "step": int(step),
            "primary_outcome_id": outcome_id,
            "primary_stance": payload.get("stance", "aligned"),
            "primary_signed_gap": signed_gap,
            "primary_confidence_weighted_gap": weighted_gap,
            "disagreement_snapshot": disagreement_snapshot,
        },
    )


def detect_belief_conflict(
    prior_outcome_probs: Dict[str, float],
    posterior_outcome_probs: Dict[str, float],
    *,
    prior_threshold: float = 0.60,
    delta_threshold: float = 0.15,
) -> Dict[str, Any] | None:
    """Detect whether an update materially contradicts the prior belief state."""

    if not prior_outcome_probs or not posterior_outcome_probs:
        return None
    prior_dominant_outcome = max(prior_outcome_probs, key=prior_outcome_probs.get)
    posterior_dominant_outcome = max(posterior_outcome_probs, key=posterior_outcome_probs.get)
    prior_prob = float(prior_outcome_probs[prior_dominant_outcome])
    posterior_prob_for_prior = float(posterior_outcome_probs.get(prior_dominant_outcome, 0.0))
    all_outcomes = set(prior_outcome_probs) | set(posterior_outcome_probs)
    total_variation = 0.5 * sum(
        abs(float(posterior_outcome_probs.get(outcome_id, 0.0)) - float(prior_outcome_probs.get(outcome_id, 0.0)))
        for outcome_id in all_outcomes
    )
    dominance_flip = prior_prob >= float(prior_threshold) and posterior_dominant_outcome != prior_dominant_outcome
    sharp_downgrade = prior_prob >= float(prior_threshold) and (
        posterior_prob_for_prior <= prior_prob - float(delta_threshold)
    )
    if not dominance_flip and not sharp_downgrade:
        return None
    return {
        "prior_outcome_probs": {key: float(value) for key, value in prior_outcome_probs.items()},
        "posterior_outcome_probs": {key: float(value) for key, value in posterior_outcome_probs.items()},
        "prior_dominant_outcome": prior_dominant_outcome,
        "posterior_dominant_outcome": posterior_dominant_outcome,
        "prior_dominant_probability": prior_prob,
        "posterior_probability_for_prior_dominant": posterior_prob_for_prior,
        "total_variation": float(total_variation),
        "dominance_flip": bool(dominance_flip),
        "sharp_downgrade": bool(sharp_downgrade),
    }


def build_belief_conflict_node(
    *,
    domain_id: str,
    step: int,
    conflict_snapshot: Dict[str, Any],
    rationale: str = "",
    signal_text: str = "",
) -> KGNode:
    """Create a KG node capturing a contradiction-driven belief revision."""

    prior_outcome = str(conflict_snapshot["prior_dominant_outcome"])
    posterior_outcome = str(conflict_snapshot["posterior_dominant_outcome"])
    prior_prob = float(conflict_snapshot["prior_dominant_probability"])
    posterior_prob = float(conflict_snapshot["posterior_probability_for_prior_dominant"])
    return KGNode(
        id=f"belief_conflict:{domain_id}:{step}",
        label=f"Belief Conflict {domain_id}",
        node_type="belief_conflict",
        content=(
            f"Prior dominant outcome {prior_outcome} moved from {prior_prob:.4f} to {posterior_prob:.4f}; "
            f"posterior dominant outcome is {posterior_outcome}. Reason: {rationale or 'update applied'}"
        ),
        confidence=min(max(float(conflict_snapshot["total_variation"]), 0.2), 0.95),
        metadata={
            "domain_id": domain_id,
            "step": int(step),
            **conflict_snapshot,
            "rationale": rationale,
            "signal_excerpt": signal_text[:500],
        },
    )


def build_epistemic_log_node(forecast: Forecast) -> KGNode:
    """Create a factual post-verification epistemic memory node for one forecast."""

    if forecast.actual_prob is None or forecast.error is None or forecast.verified_at is None:
        raise ValueError("Epistemic log nodes require a verified forecast with actual_prob and error.")
    signed_error = float(forecast.predicted_prob - forecast.actual_prob)
    rationale = str(forecast.metadata.get("rationale_at_time", "")).strip()
    return KGNode(
        id=f"epistemic:{forecast.forecast_id}",
        label=f"Epistemic Log {forecast.domain_id}/{forecast.outcome_id}",
        node_type="epistemic_log",
        content=(
            f"predicted={forecast.predicted_prob:.4f}; actual={forecast.actual_prob:.4f}; "
            f"signed_error={signed_error:+.4f}; abs_error={forecast.error:.4f}"
        ),
        confidence=0.95,
        metadata={
            "forecast_id": forecast.forecast_id,
            "domain_id": forecast.domain_id,
            "outcome_id": forecast.outcome_id,
            "predicted_prob": float(forecast.predicted_prob),
            "actual_prob": float(forecast.actual_prob),
            "signed_error": signed_error,
            "abs_error": float(forecast.error),
            "session_id": forecast.session_id,
            "created_step": int(forecast.created_step),
            "horizon_steps": int(forecast.horizon_steps),
            "rationale_at_time": rationale,
            "analysis_node_id": forecast.metadata.get("analysis_node_id"),
            "belief_confidence_at_time": forecast.metadata.get("belief_confidence"),
            "reference_prob_at_time": forecast.metadata.get("reference_prob"),
            "reference_gap_at_time": forecast.metadata.get("reference_gap"),
            "confidence_weighted_gap_at_time": forecast.metadata.get("confidence_weighted_gap"),
            "created_at": forecast.created_at.isoformat(),
            "verified_at": forecast.verified_at.isoformat(),
            "parameter_vector_at_time": forecast.metadata.get("parameter_vector"),
        },
    )


__all__ = [
    "build_belief_conflict_node",
    "build_disagreement_node",
    "build_epistemic_log_node",
    "compute_confidence_weighted_disagreement",
    "detect_belief_conflict",
    "extract_reference_outcome_probs",
    "summarize_primary_disagreement",
]
