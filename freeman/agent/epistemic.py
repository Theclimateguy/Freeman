"""Helpers for epistemic logging, disagreement tracking, and belief conflicts."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict

from freeman.agent.forecastregistry import Forecast
from freeman.core.world import WorldState
from freeman.memory.knowledgegraph import KGNode
from freeman.memory.epistemiclog import infer_domain_family, normalize_causal_chain


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
    momentum_reference_outcome_probs: Dict[str, float] | None = None,
    signal_source: str = "update_signal",
    signal_text: str = "",
    rationale: str = "",
    parameter_conflict_flag: bool = False,
    prior_threshold: float = 0.60,
    delta_threshold: float = 0.15,
    momentum_threshold: float = 0.05,
    signal_strength_threshold: float = 0.05,
) -> Dict[str, Any] | None:
    """Detect whether an update materially contradicts the prior belief state."""

    if not prior_outcome_probs or not posterior_outcome_probs:
        return None
    tracked_outcome_id = "yes" if "yes" in prior_outcome_probs and "yes" in posterior_outcome_probs else max(
        prior_outcome_probs,
        key=prior_outcome_probs.get,
    )
    belief_before = float(prior_outcome_probs.get(tracked_outcome_id, 0.0))
    belief_after = float(posterior_outcome_probs.get(tracked_outcome_id, 0.0))
    signal_strength = float(belief_after - belief_before)
    if abs(signal_strength) < signal_strength_threshold:
        signal_direction = "flat"
    else:
        signal_direction = "up" if signal_strength > 0.0 else "down"
    current_momentum = None
    current_momentum_direction = "unknown"
    trend_conflict = False
    if momentum_reference_outcome_probs is not None and tracked_outcome_id in momentum_reference_outcome_probs:
        current_momentum = belief_before - float(momentum_reference_outcome_probs[tracked_outcome_id])
        if abs(current_momentum) < momentum_threshold:
            current_momentum_direction = "flat"
        else:
            current_momentum_direction = "up" if current_momentum > 0.0 else "down"
        trend_conflict = (
            signal_direction in {"up", "down"}
            and current_momentum_direction in {"up", "down"}
            and signal_direction != current_momentum_direction
        )
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
    if not trend_conflict and not dominance_flip and not sharp_downgrade and not parameter_conflict_flag:
        return None
    if parameter_conflict_flag:
        resolution = "dampened"
        conflict_reason = "parameter_vector contradicted rationale and was dampened"
    elif signal_direction == "flat":
        resolution = "rejected"
        conflict_reason = "signal had insufficient directional force to overcome prior belief"
    else:
        resolution = "accepted"
        if trend_conflict:
            conflict_reason = f"signal contradicts {current_momentum_direction} momentum"
        elif dominance_flip:
            conflict_reason = "signal reversed the dominant prior outcome"
        else:
            conflict_reason = "signal sharply downgraded the prior dominant outcome"
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
        "tracked_outcome_id": tracked_outcome_id,
        "belief_before": belief_before,
        "belief_after": belief_after,
        "signal": {
            "source": str(signal_source),
            "direction": signal_direction,
            "strength": float(abs(signal_strength)),
            "raw_delta": signal_strength,
        },
        "signal_excerpt": signal_text[:500],
        "current_momentum": current_momentum,
        "current_momentum_direction": current_momentum_direction,
        "trend_conflict": bool(trend_conflict),
        "conflict_reason": conflict_reason,
        "resolution": resolution,
        "rationale": rationale,
        "parameter_conflict_flag": bool(parameter_conflict_flag),
        "timestamp": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
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
    belief_before = float(conflict_snapshot.get("belief_before", 0.0))
    belief_after = float(conflict_snapshot.get("belief_after", 0.0))
    signal = dict(conflict_snapshot.get("signal", {}))
    conflict_reason = str(conflict_snapshot.get("conflict_reason", rationale or "update applied"))
    resolution = str(conflict_snapshot.get("resolution", "accepted"))
    return KGNode(
        id=f"belief_conflict:{domain_id}:{step}",
        label=f"Belief Conflict {domain_id}",
        node_type="belief_conflict",
        content=(
            f"Belief on {conflict_snapshot.get('tracked_outcome_id', prior_outcome)} moved from "
            f"{belief_before:.4f} to {belief_after:.4f}; "
            f"signal_direction={signal.get('direction', 'flat')}; "
            f"resolution={resolution}. Reason: {conflict_reason}"
        ),
        confidence=min(max(float(conflict_snapshot["total_variation"]), 0.2), 0.95),
        metadata={
            "domain_id": domain_id,
            "step": int(step),
            **conflict_snapshot,
            "rationale": rationale,
            "signal_excerpt": signal_text[:500],
            "belief_before": belief_before,
            "belief_after": belief_after,
            "signal": {
                "source": signal.get("source", "update_signal"),
                "direction": signal.get("direction", "flat"),
                "strength": float(signal.get("strength", 0.0)),
            },
            "conflict_reason": conflict_reason,
            "resolution": resolution,
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
            "domain_family": forecast.metadata.get(
                "domain_family",
                infer_domain_family(forecast.domain_id, forecast.metadata),
            ),
            "causal_chain": normalize_causal_chain(forecast.metadata.get("causal_chain")),
            "predicted_prob": float(forecast.predicted_prob),
            "actual_prob": float(forecast.actual_prob),
            "predicted": float(forecast.predicted_prob),
            "actual": float(forecast.actual_prob),
            "delta": signed_error,
            "signed_error": signed_error,
            "abs_error": float(forecast.error),
            "session_id": forecast.session_id,
            "created_step": int(forecast.created_step),
            "horizon_steps": int(forecast.horizon_steps),
            "rationale_at_time": rationale,
            "rationale_at_cutoff": rationale,
            "analysis_node_id": forecast.metadata.get("analysis_node_id"),
            "belief_confidence_at_time": forecast.metadata.get("belief_confidence"),
            "reference_prob_at_time": forecast.metadata.get("reference_prob"),
            "reference_gap_at_time": forecast.metadata.get("reference_gap"),
            "confidence_weighted_gap_at_time": forecast.metadata.get("confidence_weighted_gap"),
            "created_at": forecast.created_at.isoformat(),
            "verified_at": forecast.verified_at.isoformat(),
            "timestamp_cutoff": forecast.created_at.isoformat(),
            "timestamp_resolution": forecast.verified_at.isoformat(),
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
