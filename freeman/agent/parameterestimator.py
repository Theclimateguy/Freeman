"""LLM-backed estimator for the lite runtime parameter vector."""

from __future__ import annotations

import difflib
import json
import re
from typing import Any

from freeman.core.scorer import pre_modifier_outcome_scores, scored_outcome_scores
from freeman.core.types import ParameterVector
from freeman.core.world import WorldState


class ParameterEstimator:
    """Ask the LLM to calibrate one dynamic parameter vector."""

    SYSTEM_PROMPT = """You are a causal calibration engine for a deterministic world simulator.
You will receive:
1. A snapshot of the current world and current outcome probabilities.
2. A new incoming signal.

Return exactly one JSON object with:
{
  "outcome_modifiers": {"<outcome_id>": <float>},
  "shock_decay": <float 0.0-1.0>,
  "edge_weight_deltas": {"<source_key>.<target_key>": <float>},
  "rationale": "<one sentence>"
}

Rules:
- Use only the provided outcome ids.
- outcome_modifiers should stay between 0.5 and 4.0.
- shock_decay should stay between 0.0 and 1.0.
- edge_weight_deltas should be small adjustments.
- Return JSON only.
"""

    def __init__(self, llm_client: Any) -> None:
        self.llm = llm_client

    def estimate(
        self,
        previous_world: WorldState,
        new_signal_text: str,
    ) -> ParameterVector:
        """Call the LLM and validate the returned parameter vector."""

        world_context = self._world_summary(previous_world)
        messages = [
            {"role": "system", "content": self.SYSTEM_PROMPT},
            {
                "role": "user",
                "content": f"CURRENT WORLD STATE:\n{world_context}\n\nNEW SIGNAL:\n{new_signal_text}",
            },
        ]
        result = self.llm.chat_json(messages, temperature=0.0, max_tokens=500)
        repaired_result = dict(result)
        repaired_modifiers, repair_conflicts = self._repair_outcome_ids(
            repaired_result.get("outcome_modifiers", {}),
            valid_outcome_ids=set(previous_world.outcomes),
        )
        repaired_result["outcome_modifiers"] = repaired_modifiers
        repaired_result["repair_conflicts"] = repair_conflicts
        repaired_result["conflict_flag"] = bool(repair_conflicts)
        vector = ParameterVector.from_snapshot(
            repaired_result,
            valid_outcome_ids=set(previous_world.outcomes),
        )
        return self._validate_sign_consistency(previous_world, vector)

    def _world_summary(self, world: WorldState) -> str:
        return json.dumps(
            {
                "domain_id": world.domain_id,
                "outcomes": {
                    outcome_id: {
                        "label": outcome.label,
                        "scoring_weights": dict(outcome.scoring_weights),
                    }
                    for outcome_id, outcome in world.outcomes.items()
                },
                "current_outcome_scores_pre_modifier": pre_modifier_outcome_scores(world),
                "current_outcome_scores_post_modifier": scored_outcome_scores(world),
                "current_actor_states": {
                    actor_id: dict(actor.state)
                    for actor_id, actor in world.actors.items()
                },
                "current_resource_values": {
                    resource_id: float(resource.value)
                    for resource_id, resource in world.resources.items()
                },
                "current_parameter_vector": world.parameter_vector.snapshot(),
                "causal_dag": [edge.snapshot() for edge in world.causal_dag],
            },
            indent=2,
            ensure_ascii=False,
        )

    def _binary_outcome_ids(self, world: WorldState) -> tuple[str | None, str | None]:
        yes_id = None
        no_id = None
        for outcome_id, outcome in world.outcomes.items():
            label = str(getattr(outcome, "label", outcome_id)).strip().lower()
            canonical = str(outcome_id).strip().lower()
            if canonical == "yes" or label == "yes":
                yes_id = outcome_id
            if canonical == "no" or label == "no":
                no_id = outcome_id
        return yes_id, no_id

    def _repair_outcome_ids(
        self,
        modifiers: dict[str, Any],
        *,
        valid_outcome_ids: set[str],
        fuzzy_threshold: float = 0.85,
    ) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        if not modifiers:
            return {}, []

        valid_ids = [str(outcome_id) for outcome_id in valid_outcome_ids]
        valid_lookup = {outcome_id: outcome_id for outcome_id in valid_ids}
        casefold_lookup = {outcome_id.lower(): outcome_id for outcome_id in valid_ids}
        repaired: dict[str, Any] = {}
        conflicts: list[dict[str, Any]] = []

        for raw_key, value in modifiers.items():
            key = str(raw_key)
            if key in valid_lookup:
                repaired[key] = value
                continue
            normalized = casefold_lookup.get(key.lower())
            if normalized is not None:
                repaired[normalized] = value
                conflicts.append(
                    {
                        "mismatch_type": "unknown_outcome_id",
                        "hallucinated_outcome_id": key,
                        "corrected_to": normalized,
                        "dropped": False,
                        "modifier": float(value),
                        "fuzzy_ratio": 1.0,
                    }
                )
                continue

            best_match = None
            best_ratio = 0.0
            for candidate in valid_ids:
                ratio = difflib.SequenceMatcher(None, key.lower(), candidate.lower()).ratio()
                if ratio > best_ratio:
                    best_match = candidate
                    best_ratio = ratio
            if best_match is not None and best_ratio >= fuzzy_threshold:
                repaired[best_match] = value
                conflicts.append(
                    {
                        "mismatch_type": "unknown_outcome_id",
                        "hallucinated_outcome_id": key,
                        "corrected_to": best_match,
                        "dropped": False,
                        "modifier": float(value),
                        "fuzzy_ratio": float(best_ratio),
                    }
                )
                continue

            conflicts.append(
                {
                    "mismatch_type": "unknown_outcome_id",
                    "hallucinated_outcome_id": key,
                    "corrected_to": None,
                    "dropped": True,
                    "modifier": float(value),
                    "fuzzy_ratio": float(best_ratio),
                }
            )
        return repaired, conflicts

    def _extract_directional_intent(self, rationale: str) -> str:
        lowered = rationale.strip().lower()
        if not lowered:
            return "neutral"
        if re.search(r"\b(no should rise|favor no|boost no|raise no|towards no|against yes)\b", lowered):
            return "no"
        if re.search(r"\b(yes should rise|favor yes|boost yes|raise yes|towards yes|against no)\b", lowered):
            return "yes"
        return "neutral"

    def _validate_sign_consistency(
        self,
        world: WorldState,
        vector: ParameterVector,
    ) -> ParameterVector:
        intent = self._extract_directional_intent(vector.rationale)
        if intent == "neutral":
            return vector
        yes_id, no_id = self._binary_outcome_ids(world)
        if yes_id is None or no_id is None:
            return vector

        modifiers = dict(vector.outcome_modifiers)
        conflicts = list(vector.repair_conflicts)
        if intent == "yes":
            if float(modifiers.get(yes_id, 1.0)) < 1.0:
                modifiers[yes_id] = 1.0
                conflicts.append({"mismatch_type": "intent_conflict", "outcome_id": yes_id, "corrected_to": 1.0})
            if float(modifiers.get(no_id, 1.0)) > 1.0:
                modifiers[no_id] = 1.0
                conflicts.append({"mismatch_type": "intent_conflict", "outcome_id": no_id, "corrected_to": 1.0})
        if intent == "no":
            if float(modifiers.get(no_id, 1.0)) < 1.0:
                modifiers[no_id] = 1.0
                conflicts.append({"mismatch_type": "intent_conflict", "outcome_id": no_id, "corrected_to": 1.0})
            if float(modifiers.get(yes_id, 1.0)) > 1.0:
                modifiers[yes_id] = 1.0
                conflicts.append({"mismatch_type": "intent_conflict", "outcome_id": yes_id, "corrected_to": 1.0})
        return ParameterVector(
            outcome_modifiers=modifiers,
            shock_decay=vector.shock_decay,
            edge_weight_deltas=vector.edge_weight_deltas,
            rationale=vector.rationale,
            conflict_flag=bool(conflicts),
            repair_conflicts=conflicts,
            valid_outcome_ids=tuple(world.outcomes),
        )


__all__ = ["ParameterEstimator"]
