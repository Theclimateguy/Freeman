"""LLM-backed estimator for the dynamic simulator parameter vector."""

from __future__ import annotations

import difflib
import json
import logging
import re
from typing import Any

from freeman.core.scorer import pre_modifier_outcome_scores, scored_outcome_scores
from freeman.core.types import ParameterVector
from freeman.core.world import WorldState
from freeman.memory.beliefconflictlog import BeliefConflictLog
from freeman.memory.epistemiclog import EpistemicLog

LOGGER = logging.getLogger(__name__)


class ParameterEstimator:
    """Ask the LLM to calibrate the dynamic parameter vector for a world update."""

    SYSTEM_PROMPT = """You are a causal calibration engine for a deterministic world simulator.
You will receive:
1. A snapshot of the current world (actors, resources, DAG structure, current outcome scores).
2. A new incoming signal (news, data, event).

Your job is to output a JSON ParameterVector that adjusts the simulator's sensitivity to the new signal.
Rules:
- `current_outcome_scores_post_modifier` is the actual simulator output after the active ParameterVector was applied.
- `current_outcome_scores_pre_modifier` is the same score surface before outcome modifiers were applied.
- `domain_polarity` says whether literal YES is favorable (`positive`) or adverse/risk-like (`negative`); preserve the literal YES semantics of the market question.
- `epistemic_memory` contains verified past forecast errors for similar domains. Use it to avoid repeating systematic mistakes.
- `recent_belief_conflicts` contains recent contradictions between new signals and prior momentum. Respect dampened updates and conflict traces.
- Only use outcome ids that exactly match the provided `outcomes` keys. Do not invent or rename outcome ids.
- outcome_modifiers: multiply the base score of outcomes that the new signal fundamentally changes.
  Use values between 0.5 (suppress) and 4.0 (amplify). Only include outcomes that need adjustment.
  Default is 1.0 (no change).
- shock_decay: how much prior accumulated state should decay before applying new shocks.
  1.0 = full memory, 0.0 = full reset, 0.5 = half decay.
  Use 0.5-0.8 for paradigm shifts, 0.9-1.0 for gradual updates.
- edge_weight_deltas: small adjustments (+/-) to causal influence between world variables.
  Only include if the new signal changes how two variables relate to each other.
- rationale: one sentence explaining the key calibration decision.

Output ONLY valid JSON matching this schema:
{
  "outcome_modifiers": {"<outcome_id>": <float>},
  "shock_decay": <float 0.0-1.0>,
  "edge_weight_deltas": {"<source_key>.<target_key>": <float>},
  "rationale": "<string>"
}"""

    def __init__(
        self,
        llm_client: Any,
        *,
        epistemic_log: EpistemicLog | None = None,
        belief_conflict_log: BeliefConflictLog | None = None,
        max_epistemic_examples: int = 5,
    ) -> None:
        self.llm = llm_client
        self.epistemic_log = epistemic_log
        self.belief_conflict_log = belief_conflict_log
        self.max_epistemic_examples = int(max_epistemic_examples)

    def estimate(
        self,
        previous_world: WorldState,
        new_signal_text: str,
    ) -> ParameterVector:
        """Call the LLM to produce a dynamic parameter vector for the next update."""

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
        repaired_result["conflict_flag"] = bool(repaired_result.get("conflict_flag", False) or repair_conflicts)
        vector = ParameterVector.from_snapshot(
            repaired_result,
            valid_outcome_ids=set(previous_world.outcomes),
        )
        return self._validate_sign_consistency(previous_world, vector)

    def _world_summary(self, world: WorldState) -> str:
        """Return a compact JSON summary of world structure and active scores."""

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
                "current_outcome_scores": scored_outcome_scores(world),
                "domain_polarity": world.metadata.get("domain_polarity", "positive"),
                "modifier_mode": world.metadata.get("modifier_mode", "legacy"),
                "causal_dag": [edge.snapshot() for edge in world.causal_dag],
                "current_actor_states": {
                    actor_id: dict(actor.state)
                    for actor_id, actor in world.actors.items()
                },
                "current_resource_values": {
                    resource_id: float(resource.value)
                    for resource_id, resource in world.resources.items()
                },
                "current_parameter_vector": world.parameter_vector.snapshot(),
                "epistemic_memory": self._epistemic_memory_context(world),
                "recent_belief_conflicts": self._belief_conflict_context(world),
            },
            indent=2,
            ensure_ascii=False,
        )

    def _epistemic_memory_context(self, world: WorldState) -> list[dict[str, Any]]:
        """Return relevant verified forecast errors for this domain family."""

        if self.epistemic_log is None:
            return []
        return self.epistemic_log.context_for_world(world, limit=self.max_epistemic_examples)

    def _belief_conflict_context(self, world: WorldState) -> list[dict[str, Any]]:
        """Return recent domain conflict traces for prompt conditioning."""

        if self.belief_conflict_log is None:
            return []
        records = self.belief_conflict_log.query(domain_id=world.domain_id, limit=self.max_epistemic_examples)
        return [record.prompt_payload() for record in records]

    def _binary_outcome_ids(self, world: WorldState) -> tuple[str | None, str | None]:
        """Return the literal YES/NO outcome ids when the world exposes them."""

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
        """Repair hallucinated outcome ids before constructing a validated ParameterVector."""

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
        """Infer whether the rationale favors literal YES or NO."""

        lowered = rationale.strip().lower()
        if not lowered:
            return "neutral"
        no_patterns = [
            r"\bfavo?u?rs?\s+no\b",
            r"\bfavor\s+the\s+no\b",
            r"\bprefer(?:s)?\s+no\b",
            r"\bsupports?\s+no\b",
            r"\bboosts?\s+no\b",
            r"\braises?\s+no\b",
            r"\bpush(?:es)?\s+towards?\s+no\b",
            r"\bfavor(?:s)?\s+against\s+yes\b",
            r"\b(lower|lowers|reduce|reduces|decrease|decreases|suppress|suppresses|weaken|weakens)\s+yes\b",
            r"\byes\s+(should\s+)?(fall|drop|decline|decrease|be\s+lower)\b",
        ]
        yes_patterns = [
            r"\bfavo?u?rs?\s+yes\b",
            r"\bfavor\s+the\s+yes\b",
            r"\bprefer(?:s)?\s+yes\b",
            r"\bsupports?\s+yes\b",
            r"\bboosts?\s+yes\b",
            r"\braises?\s+yes\b",
            r"\bpush(?:es)?\s+towards?\s+yes\b",
            r"\b(lower|lowers|reduce|reduces|decrease|decreases|suppress|suppresses|weaken|weakens)\s+no\b",
            r"\bno\s+(should\s+)?(fall|drop|decline|decrease|be\s+lower)\b",
        ]
        if any(re.search(pattern, lowered) for pattern in no_patterns):
            return "no"
        if any(re.search(pattern, lowered) for pattern in yes_patterns):
            return "yes"
        return "neutral"

    def _validate_sign_consistency(
        self,
        world: WorldState,
        vector: ParameterVector,
    ) -> ParameterVector:
        """Dampen modifiers that contradict the direction stated in the rationale."""

        intent = self._extract_directional_intent(vector.rationale)
        yes_id, no_id = self._binary_outcome_ids(world)
        if intent == "neutral" or yes_id is None or no_id is None:
            return vector

        modifiers = dict(vector.outcome_modifiers)
        conflict_messages: list[str] = []
        yes_modifier = float(modifiers.get(yes_id, 1.0))
        no_modifier = float(modifiers.get(no_id, 1.0))

        if intent == "no" and yes_modifier > 1.0:
            modifiers[yes_id] = 1.0
            conflict_messages.append(f"dampened {yes_id} from {yes_modifier:.3f} to 1.0")
        if intent == "yes" and no_modifier > 1.0:
            modifiers[no_id] = 1.0
            conflict_messages.append(f"dampened {no_id} from {no_modifier:.3f} to 1.0")

        if not conflict_messages:
            return vector

        LOGGER.warning(
            "parameter_vector sign conflict domain_id=%s intent=%s rationale=%r actions=%s",
            world.domain_id,
            intent,
            vector.rationale,
            "; ".join(conflict_messages),
        )
        return ParameterVector(
            outcome_modifiers=modifiers,
            shock_decay=vector.shock_decay,
            edge_weight_deltas=vector.edge_weight_deltas,
            rationale=vector.rationale,
            conflict_flag=True,
            repair_conflicts=vector.repair_conflicts,
            valid_outcome_ids=tuple(world.outcomes),
        )


__all__ = ["ParameterEstimator"]
