"""Tests for ParameterEstimator score summaries."""

from __future__ import annotations

from datetime import datetime, timezone
import json

import pytest

from freeman.agent.forecastregistry import Forecast
from freeman.agent.parameterestimator import ParameterEstimator
from freeman.core.scorer import pre_modifier_outcome_scores, scored_outcome_scores
from freeman.core.types import ParameterVector
from freeman.memory.epistemiclog import EpistemicLog
from freeman.memory.knowledgegraph import KnowledgeGraph


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


class _ConflictingStubLLM:
    def chat_json(self, messages, *, temperature=0.0, max_tokens=None):  # noqa: ANN001
        del messages, temperature, max_tokens
        return {
            "outcome_modifiers": {"yes": 2.5, "no": 0.6},
            "shock_decay": 1.0,
            "edge_weight_deltas": {},
            "rationale": "Mild season, favor NO.",
        }


def test_estimate_dampens_modifier_when_rationale_favors_no(water_market_world) -> None:
    world = water_market_world.clone()
    world.outcomes = {
        "yes": world.outcomes["cooperation"],
        "no": world.outcomes["water_crisis"],
    }
    world.outcomes["yes"].id = "yes"
    world.outcomes["yes"].label = "YES"
    world.outcomes["no"].id = "no"
    world.outcomes["no"].label = "NO"
    estimator = ParameterEstimator(_ConflictingStubLLM())

    vector = estimator.estimate(world, "Record wildfire season but a mild insured-loss readthrough.")

    assert vector.outcome_modifiers["yes"] == 1.0
    assert vector.outcome_modifiers["no"] == 0.6
    assert vector.conflict_flag is True


def test_world_summary_includes_relevant_epistemic_memory(water_market_world, tmp_path) -> None:
    knowledge_graph = KnowledgeGraph(json_path=tmp_path / "kg.json", auto_load=False, auto_save=False)
    epistemic_log = EpistemicLog(knowledge_graph)
    forecast = Forecast(
        forecast_id="water_market:3:cooperation",
        domain_id="water_market",
        outcome_id="cooperation",
        predicted_prob=0.72,
        session_id="session-1",
        horizon_steps=3,
        created_at=datetime(2026, 3, 27, tzinfo=timezone.utc),
        created_step=3,
        verified_at=datetime(2026, 3, 29, tzinfo=timezone.utc),
        actual_prob=1.0,
        error=0.28,
        status="verified",
        metadata={
            "domain_family": "water_market",
            "causal_chain": ["water_stock", "conflict_level"],
            "rationale_at_time": "Prior conflict risk was overstated.",
        },
    )
    epistemic_log.record(forecast)
    water_market_world.metadata["domain_family"] = "water_market"
    water_market_world.metadata["causal_chain"] = ["water_stock", "conflict_level"]
    estimator = ParameterEstimator(_StubLLM(), epistemic_log=epistemic_log)

    payload = json.loads(estimator._world_summary(water_market_world))

    assert payload["epistemic_memory"]
    assert payload["epistemic_memory"][0]["domain_family"] == "water_market"
    assert payload["epistemic_memory"][0]["delta"] == -0.28


class _HallucinatedOutcomeStubLLM:
    def chat_json(self, messages, *, temperature=0.0, max_tokens=None):  # noqa: ANN001
        del messages, temperature, max_tokens
        return {
            "outcome_modifiers": {
                "water_crisys": 1.7,
                "phantom_outcome": 0.6,
            },
            "shock_decay": 0.9,
            "edge_weight_deltas": {},
            "rationale": "Water crisis risk should rise while the phantom tail case fades.",
        }


def test_estimate_repairs_hallucinated_outcome_ids_and_logs_conflicts(water_market_world) -> None:
    estimator = ParameterEstimator(_HallucinatedOutcomeStubLLM())

    vector = estimator.estimate(water_market_world, "Reservoir stress deepens under prolonged drought.")

    assert vector.outcome_modifiers["water_crisis"] == 1.7
    assert "water_crisys" not in vector.outcome_modifiers
    assert "phantom_outcome" not in vector.outcome_modifiers
    assert vector.conflict_flag is True
    assert len(vector.repair_conflicts) == 2
    assert vector.repair_conflicts[0]["hallucinated_outcome_id"] == "water_crisys"
    assert vector.repair_conflicts[0]["corrected_to"] == "water_crisis"
    assert vector.repair_conflicts[1]["dropped"] is True


def test_parameter_vector_rejects_unknown_outcome_ids() -> None:
    with pytest.raises(ValueError, match="Unknown outcome_ids"):
        ParameterVector.from_snapshot(
            {"outcome_modifiers": {"ghost": 1.2}},
            valid_outcome_ids={"yes", "no"},
        )
