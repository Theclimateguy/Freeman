"""Tests for proactive event generation."""

from __future__ import annotations

from freeman.agent.analysispipeline import AnalysisPipeline, AnalysisPipelineResult
from freeman.agent.proactiveemitter import ProactiveEmitter
from freeman.memory.knowledgegraph import KnowledgeGraph


def test_proactive_emitter_generates_alert_for_hard_violations(water_market_world) -> None:
    emitter = ProactiveEmitter()
    result = AnalysisPipelineResult(
        world=water_market_world,
        simulation={
            "violations": [{"severity": "hard", "message": "boom"}],
            "final_outcome_probs": {"cooperation": 0.8},
            "confidence": 0.9,
        },
        verification={},
        raw_scores={},
        dominant_outcome="cooperation",
        knowledge_graph_path="memory.json",
    )

    events = emitter.evaluate(result)

    assert len(events) == 1
    assert events[0].event_type == "alert"


def test_proactive_emitter_generates_forecast_update_for_material_shift(water_market_world) -> None:
    emitter = ProactiveEmitter(forecast_shift_threshold=0.15)
    result = AnalysisPipelineResult(
        world=water_market_world,
        simulation={
            "violations": [],
            "final_outcome_probs": {"cooperation": 0.55},
            "confidence": 0.8,
        },
        verification={},
        raw_scores={},
        dominant_outcome="cooperation",
        knowledge_graph_path="memory.json",
    )

    events = emitter.evaluate(result, prev_probs={"cooperation": 0.30})

    assert len(events) == 1
    assert events[0].event_type == "forecast_update"
    assert events[0].outcome_id == "cooperation"


def test_proactive_emitter_generates_question_to_human_for_low_confidence(water_market_world) -> None:
    emitter = ProactiveEmitter(confidence_floor=0.4)
    result = AnalysisPipelineResult(
        world=water_market_world,
        simulation={
            "violations": [],
            "final_outcome_probs": {"cooperation": 0.55},
            "confidence": 0.2,
        },
        verification={},
        raw_scores={},
        dominant_outcome="cooperation",
        knowledge_graph_path="memory.json",
    )

    events = emitter.evaluate(result)

    assert len(events) == 1
    assert events[0].event_type == "question_to_human"


def test_analysis_pipeline_without_emitter_returns_no_proactive_events(tmp_path, water_market_schema) -> None:
    pipeline = AnalysisPipeline(
        knowledge_graph=KnowledgeGraph(json_path=tmp_path / "kg.json", auto_load=False, auto_save=False),
        emitter=None,
    )

    result = pipeline.run(water_market_schema)

    assert result.proactive_events == []
