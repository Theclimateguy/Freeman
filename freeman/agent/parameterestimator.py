"""LLM-backed estimator for the dynamic simulator parameter vector."""

from __future__ import annotations

import json
from typing import Any

from freeman.core.scorer import pre_modifier_outcome_scores, scored_outcome_scores
from freeman.core.types import ParameterVector
from freeman.core.world import WorldState


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

    def __init__(self, llm_client: Any) -> None:
        self.llm = llm_client

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
        return ParameterVector.from_snapshot(result)

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
            },
            indent=2,
            ensure_ascii=False,
        )


__all__ = ["ParameterEstimator"]
