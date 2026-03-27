"""Tests for forecast registration and verification."""

from __future__ import annotations

from datetime import datetime, timezone

from freeman.agent.analysispipeline import AnalysisPipeline
from freeman.agent.attentionscheduler import ObligationQueue
from freeman.agent.forecastregistry import Forecast, ForecastRegistry
from freeman.game.runner import SimConfig
from freeman.memory.knowledgegraph import KnowledgeGraph
from freeman.memory.sessionlog import SessionLog


def test_forecast_registry_due_uses_created_step_plus_horizon(tmp_path) -> None:
    registry = ForecastRegistry(json_path=tmp_path / "forecasts.json", auto_load=False, auto_save=False)
    due_forecast = Forecast(
        forecast_id="water:5:cooperation",
        domain_id="water",
        outcome_id="cooperation",
        predicted_prob=0.7,
        session_id="session-1",
        horizon_steps=3,
        created_at=datetime(2026, 3, 27, tzinfo=timezone.utc),
        created_step=5,
    )
    future_forecast = Forecast(
        forecast_id="water:6:crisis",
        domain_id="water",
        outcome_id="crisis",
        predicted_prob=0.2,
        session_id="session-1",
        horizon_steps=4,
        created_at=datetime(2026, 3, 27, tzinfo=timezone.utc),
        created_step=6,
    )
    registry.record(due_forecast)
    registry.record(future_forecast)

    due = registry.due(current_step=8)

    assert [forecast.forecast_id for forecast in due] == ["water:5:cooperation"]


def test_forecast_registry_verify_sets_error_and_roundtrips_json(tmp_path) -> None:
    registry = ForecastRegistry(json_path=tmp_path / "forecasts.json", auto_load=False, auto_save=True)
    forecast = Forecast(
        forecast_id="water:5:cooperation",
        domain_id="water",
        outcome_id="cooperation",
        predicted_prob=0.7,
        session_id="session-1",
        horizon_steps=3,
        created_at=datetime(2026, 3, 27, tzinfo=timezone.utc),
        created_step=5,
    )
    registry.record(forecast)

    verified = registry.verify(
        "water:5:cooperation",
        actual_prob=0.55,
        verified_at=datetime(2026, 3, 30, tzinfo=timezone.utc),
    )
    reloaded = ForecastRegistry(json_path=tmp_path / "forecasts.json", auto_load=True, auto_save=False)

    assert verified.status == "verified"
    assert verified.error == 0.15
    assert reloaded.snapshot()[0]["status"] == "verified"
    assert reloaded.snapshot()[0]["error"] == 0.15


def test_analysis_pipeline_records_forecasts_and_creates_forecast_debt(tmp_path, water_market_schema) -> None:
    obligations = ObligationQueue()
    registry = ForecastRegistry(auto_load=False, auto_save=False, obligation_queue=obligations)
    knowledge_graph = KnowledgeGraph(json_path=tmp_path / "kg.json", auto_load=False, auto_save=False)
    session_log = SessionLog(session_id="forecast-session")
    pipeline = AnalysisPipeline(
        sim_config=SimConfig(max_steps=5, convergence_check_steps=100, convergence_epsilon=3.0e-2, seed=11),
        knowledge_graph=knowledge_graph,
        forecast_registry=registry,
    )

    result = pipeline.run(water_market_schema, session_log=session_log)
    recorded = registry.pending()

    assert len(recorded) == len(result.simulation["final_outcome_probs"])
    assert len(obligations.forecast_debts) == len(recorded)
    assert {forecast.forecast_id for forecast in recorded} == set(result.metadata["forecast_ids"])
    assert all(forecast.session_id == "forecast-session" for forecast in recorded)
    assert all(forecast.horizon_steps == 5 for forecast in recorded)
