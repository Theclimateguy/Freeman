"""LLM-backed estimator for the dynamic simulator parameter vector."""

from __future__ import annotations

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
        vector = ParameterVector.from_snapshot(result)
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
        )


__all__ = ["ParameterEstimator"]
