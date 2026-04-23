from __future__ import annotations

from freeman.agent.analysispipeline import AnalysisPipeline
from freeman.agent.forecastregistry import ForecastRegistry
from freeman.core.types import ParameterVector
from freeman.game.runner import SimConfig
from freeman.memory.knowledgegraph import KnowledgeGraph


def test_pipeline_runs_and_verifies_forecasts(tmp_path, water_market_schema) -> None:
    kg = KnowledgeGraph(json_path=tmp_path / "kg_state.json", auto_load=False, auto_save=False)
    forecasts = ForecastRegistry(json_path=tmp_path / "forecasts.json", auto_load=False, auto_save=False)
    pipeline = AnalysisPipeline(
        knowledge_graph=kg,
        forecast_registry=forecasts,
        sim_config=SimConfig(max_steps=10, dt=1.0),
    )

    initial = pipeline.run(water_market_schema, source_text="Baseline water market schema.")
    assert initial.dominant_outcome is not None
    assert initial.metadata["forecast_ids"]
    assert kg.get_node("analysis:water_market:10") is not None
    assert any(node.node_type == "causal_edge" for node in kg.nodes())

    updated = pipeline.update(
        initial.world,
        ParameterVector(outcome_modifiers={"water_crisis": 1.5}, rationale="risk rises"),
        signal_text="Drought lowers inflows and raises crisis risk.",
        signal_id="sig:test",
    )
    assert updated.metadata["verified_forecast_ids"]
    assert any(item.status == "verified" for item in forecasts.all())
    assert kg.get_node("signal:sig:test") is not None
