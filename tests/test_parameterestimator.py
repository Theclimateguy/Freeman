"""Tests for ParameterEstimator score summaries."""

from __future__ import annotations

import json

from freeman.agent.parameterestimator import ParameterEstimator
from freeman.core.scorer import pre_modifier_outcome_scores, scored_outcome_scores
from freeman.core.types import ParameterVector


class _StubLLM:
    def chat_json(self, messages, *, temperature=0.0, max_tokens=None):  # noqa: ANN001
        del messages, temperature, max_tokens
        return {
            "outcome_modifiers": {},
            "shock_decay": 1.0,
            "edge_weight_deltas": {},
            "rationale": "stub",
        }


def test_world_summary_exposes_pre_and_post_modifier_scores(water_market_world) -> None:
    water_market_world.parameter_vector = ParameterVector(
        outcome_modifiers={
            "cooperation": 2.0,
            "water_crisis": 0.5,
        }
    )
    estimator = ParameterEstimator(_StubLLM())

    payload = json.loads(estimator._world_summary(water_market_world))

    expected_pre = pre_modifier_outcome_scores(water_market_world)
    expected_post = scored_outcome_scores(water_market_world)

    assert payload["current_outcome_scores_pre_modifier"] == expected_pre
    assert payload["current_outcome_scores_post_modifier"] == expected_post
    assert payload["current_outcome_scores"] == expected_post
    assert payload["current_outcome_scores_pre_modifier"]["cooperation"] != payload["current_outcome_scores_post_modifier"]["cooperation"]
    assert payload["current_parameter_vector"] == water_market_world.parameter_vector.snapshot()


def test_system_prompt_mentions_post_modifier_scores() -> None:
    assert "current_outcome_scores_post_modifier" in ParameterEstimator.SYSTEM_PROMPT
    assert "current_outcome_scores_pre_modifier" in ParameterEstimator.SYSTEM_PROMPT
