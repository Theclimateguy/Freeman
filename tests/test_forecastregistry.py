"""Tests for forecast registration and verification."""

from __future__ import annotations

import copy
from datetime import datetime, timezone

from freeman.agent.analysispipeline import AnalysisPipeline
from freeman.agent.attentionscheduler import ObligationQueue
from freeman.agent.epistemic import detect_belief_conflict
from freeman.agent.forecastregistry import Forecast, ForecastRegistry
from freeman.core.types import ParameterVector
from freeman.game.runner import SimConfig
from freeman.memory.beliefconflictlog import BeliefConflictLog
from freeman.memory.epistemiclog import EpistemicLog
from freeman.memory.knowledgegraph import KGEdge, KGNode, KnowledgeGraph
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


def test_analysis_pipeline_records_confidence_weighted_disagreement(tmp_path, water_market_schema) -> None:
    schema = copy.deepcopy(water_market_schema)
    schema["metadata"]["reference_outcome_probs"] = {
        "cooperation": 0.05,
        "water_crisis": 0.90,
        "conflict_escalation": 0.05,
    }
    registry = ForecastRegistry(auto_load=False, auto_save=False)
    knowledge_graph = KnowledgeGraph(json_path=tmp_path / "kg.json", auto_load=False, auto_save=False)
    pipeline = AnalysisPipeline(
        sim_config=SimConfig(max_steps=5, convergence_check_steps=100, convergence_epsilon=3.0e-2, seed=11),
        knowledge_graph=knowledge_graph,
        forecast_registry=registry,
    )

    result = pipeline.run(schema, session_log=SessionLog(session_id="disagreement-session"))
    recorded = {forecast.outcome_id: forecast for forecast in registry.pending()}
    disagreement_nodes = [node for node in knowledge_graph.nodes() if node.node_type == "belief_disagreement"]

    assert recorded["cooperation"].metadata["reference_prob"] == 0.05
    assert recorded["cooperation"].metadata["confidence_weighted_gap"] is not None
    assert disagreement_nodes
    assert disagreement_nodes[0].id in result.metadata["epistemic_event_ids"]


def test_analysis_pipeline_update_logs_belief_conflict(tmp_path) -> None:
    schema = {
        "domain_id": "belief_flip",
        "name": "Belief Flip",
        "description": "A tiny schema for contradiction testing.",
        "actors": [],
        "resources": [
            {
                "id": "signal",
                "name": "Signal",
                "value": 1.0,
                "unit": "u",
                "min_value": -10.0,
                "max_value": 10.0,
                "evolution_type": "linear",
                "evolution_params": {"a": 1.0, "b": 0.0, "c": 0.0, "coupling_weights": {}},
            }
        ],
        "relations": [],
        "outcomes": [
            {"id": "yes", "label": "YES", "scoring_weights": {"signal": 2.0}},
            {"id": "no", "label": "NO", "scoring_weights": {"signal": 1.0}},
        ],
        "causal_dag": [],
        "metadata": {},
    }
    knowledge_graph = KnowledgeGraph(json_path=tmp_path / "kg.json", auto_load=False, auto_save=False)
    pipeline = AnalysisPipeline(
        sim_config=SimConfig(max_steps=1, convergence_check_steps=25, convergence_epsilon=1.0e-5, seed=11),
        knowledge_graph=knowledge_graph,
    )
    world = pipeline.compiler.compile(schema)
    pipeline.run(schema, session_log=SessionLog(session_id="belief-prior-1"))
    pipeline.run(schema, session_log=SessionLog(session_id="belief-prior-2"))

    result = pipeline.update(
        world,
        ParameterVector(
            outcome_modifiers={"yes": 0.5, "no": 4.0},
            rationale="New evidence strongly contradicts the prior YES case.",
        ),
        signal_text="Incoming evidence reverses the prior case.",
        session_log=SessionLog(session_id="belief-conflict"),
    )
    conflict_nodes = [node for node in knowledge_graph.nodes() if node.node_type == "belief_conflict"]

    assert conflict_nodes
    assert conflict_nodes[0].metadata["prior_dominant_outcome"] == "yes"


def test_analysis_pipeline_update_exports_causal_edges_and_forecast_paths(tmp_path, water_market_schema) -> None:
    registry = ForecastRegistry(auto_load=False, auto_save=False)
    knowledge_graph = KnowledgeGraph(json_path=tmp_path / "kg.json", auto_load=False, auto_save=False)
    pipeline = AnalysisPipeline(
        sim_config=SimConfig(max_steps=3, convergence_check_steps=50, convergence_epsilon=1.0e-4, seed=17),
        knowledge_graph=knowledge_graph,
        forecast_registry=registry,
    )
    world = pipeline.compiler.compile(water_market_schema)

    result = pipeline.update(
        world,
        ParameterVector(
            outcome_modifiers={"water_crisis": 1.0, "cooperation": -0.4},
            rationale="Drought shock worsens cooperation and raises crisis risk.",
        ),
        signal_text="Drought shock escalates across the basin.",
        signal_id="signal-123",
        session_log=SessionLog(session_id="causal-export"),
    )
    recorded = registry.pending()
    relation_types = {edge.relation_type for edge in knowledge_graph.edges()}

    assert result.metadata["forecast_ids"]
    assert {"causes", "propagates_to", "threshold_exceeded"} <= relation_types
    assert all(forecast.causal_path for forecast in recorded)
    assert any(edge.source == "signal:signal-123" and edge.relation_type == "causes" for edge in knowledge_graph.edges())


def test_detect_belief_conflict_uses_signal_direction_vs_momentum() -> None:
    snapshot = detect_belief_conflict(
        {"yes": 0.78, "no": 0.22},
        {"yes": 0.71, "no": 0.29},
        momentum_reference_outcome_probs={"yes": 0.65, "no": 0.35},
        signal_source="historical_news",
        signal_text="Incoming signal weakens the YES case.",
        rationale="The new evidence weakens YES.",
    )

    assert snapshot is not None
    assert snapshot["trend_conflict"] is True
    assert snapshot["signal"]["direction"] == "down"
    assert snapshot["current_momentum_direction"] == "up"
    assert snapshot["conflict_reason"] == "signal contradicts up momentum"


def test_belief_conflict_log_is_queryable_per_domain(tmp_path) -> None:
    knowledge_graph = KnowledgeGraph(json_path=tmp_path / "kg.json", auto_load=False, auto_save=False)
    pipeline = AnalysisPipeline(
        sim_config=SimConfig(max_steps=1, convergence_check_steps=25, convergence_epsilon=1.0e-5, seed=11),
        knowledge_graph=knowledge_graph,
    )
    schema = {
        "domain_id": "binary_case",
        "name": "Binary Case",
        "description": "Conflict log query coverage.",
        "actors": [],
        "resources": [
            {
                "id": "signal",
                "name": "Signal",
                "value": 1.0,
                "unit": "u",
                "min_value": -10.0,
                "max_value": 10.0,
                "evolution_type": "linear",
                "evolution_params": {"a": 1.0, "b": 0.0, "c": 0.0, "coupling_weights": {}},
            }
        ],
        "relations": [],
        "outcomes": [
            {"id": "yes", "label": "YES", "scoring_weights": {"signal": 2.0}},
            {"id": "no", "label": "NO", "scoring_weights": {"signal": 1.0}},
        ],
        "causal_dag": [],
        "metadata": {},
    }
    world = pipeline.compiler.compile(schema)
    pipeline.run(schema, session_log=SessionLog(session_id="binary-prior-1"))
    pipeline.run(schema, session_log=SessionLog(session_id="binary-prior-2"))
    pipeline.update(
        world,
        ParameterVector(
            outcome_modifiers={"yes": 0.5, "no": 4.0},
            rationale="Incoming signal should favor NO.",
        ),
        signal_text="Loss severity now points down for YES.",
        session_log=SessionLog(session_id="binary-update"),
    )

    records = BeliefConflictLog(knowledge_graph).query(domain_id="binary_case")

    assert len(records) == 1
    assert records[0].signal_direction == "down"
    assert records[0].domain_id == "binary_case"


def test_analysis_pipeline_verify_forecast_creates_epistemic_log(tmp_path, water_market_schema) -> None:
    registry = ForecastRegistry(auto_load=False, auto_save=False)
    knowledge_graph = KnowledgeGraph(json_path=tmp_path / "kg.json", auto_load=False, auto_save=False)
    pipeline = AnalysisPipeline(
        sim_config=SimConfig(max_steps=3, convergence_check_steps=100, convergence_epsilon=3.0e-2, seed=11),
        knowledge_graph=knowledge_graph,
        forecast_registry=registry,
    )
    pipeline.run(water_market_schema, session_log=SessionLog(session_id="verify-run"))
    forecast = registry.pending()[0]

    verified = pipeline.verify_forecast(
        forecast.forecast_id,
        actual_prob=0.0,
        verified_at=datetime(2026, 3, 30, tzinfo=timezone.utc),
        session_log=SessionLog(session_id="verify-session"),
    )
    epistemic_node = knowledge_graph.get_node(f"epistemic:{forecast.forecast_id}")
    self_node = knowledge_graph.get_node(f"self:forecast_error:{forecast.domain_id}:{forecast.outcome_id}")

    assert verified.status == "verified"
    assert epistemic_node is not None
    assert epistemic_node.metadata["forecast_id"] == forecast.forecast_id
    assert epistemic_node.metadata["actual_prob"] == 0.0
    assert epistemic_node.metadata["analysis_node_id"] == forecast.metadata["analysis_node_id"]
    assert self_node is not None


def test_explain_forecast_verified(tmp_path, water_market_schema) -> None:
    registry = ForecastRegistry(auto_load=False, auto_save=False)
    knowledge_graph = KnowledgeGraph(json_path=tmp_path / "kg.json", auto_load=False, auto_save=False)
    pipeline = AnalysisPipeline(
        sim_config=SimConfig(max_steps=3, convergence_check_steps=50, convergence_epsilon=1.0e-4, seed=17),
        knowledge_graph=knowledge_graph,
        forecast_registry=registry,
    )
    world = pipeline.compiler.compile(water_market_schema)

    pipeline.update(
        world,
        ParameterVector(
            outcome_modifiers={"water_crisis": 0.8, "cooperation": -0.3},
            rationale="Drought shock worsens crisis risk.",
        ),
        signal_text="Drought shock escalates across the basin.",
        signal_id="signal-verified",
        session_log=SessionLog(session_id="explain-verified"),
    )
    forecast = registry.pending()[0]
    pipeline.verify_forecast(
        forecast.forecast_id,
        actual_prob=float(forecast.predicted_prob),
        verified_at=datetime(2026, 4, 13, tzinfo=timezone.utc),
        session_log=SessionLog(session_id="verify-verified"),
    )

    explanation = pipeline.explain_forecast(forecast.forecast_id)

    assert explanation.status == "verified"
    assert explanation.causal_chain
    assert explanation.causal_path_confirmed == len(explanation.causal_chain)
    assert explanation.causal_path_refuted == 0
    assert "Causal chain:" in explanation.to_text()


def test_explain_forecast_refuted(tmp_path) -> None:
    registry = ForecastRegistry(auto_load=False, auto_save=False)
    knowledge_graph = KnowledgeGraph(json_path=tmp_path / "kg.json", auto_load=False, auto_save=True)
    pipeline = AnalysisPipeline(
        knowledge_graph=knowledge_graph,
        forecast_registry=registry,
        sim_config=SimConfig(max_steps=1),
    )

    for node in (
        KGNode(id="signal:s1", label="Signal s1", node_type="signal_event", content="signal s1", confidence=0.8),
        KGNode(id="signal:s2", label="Signal s2", node_type="signal_event", content="signal s2", confidence=0.8),
        KGNode(
            id="param_delta:outcome_modifier:crisis:+1.000000",
            label="Param Delta crisis +",
            node_type="param_delta",
            content="crisis +1.0",
            confidence=0.8,
        ),
        KGNode(
            id="param_delta:outcome_modifier:crisis:-0.900000",
            label="Param Delta crisis -",
            node_type="param_delta",
            content="crisis -0.9",
            confidence=0.6,
        ),
        KGNode(
            id="variable:water_stock:t=1",
            label="Variable water_stock",
            node_type="variable_state",
            content="water_stock at t=1",
            confidence=0.8,
        ),
        KGNode(
            id="outcome:crisis:p=0.800000",
            label="Outcome crisis",
            node_type="outcome_projection",
            content="crisis p=0.8",
            confidence=0.8,
        ),
    ):
        knowledge_graph.add_node(node)

    knowledge_graph.add_edge(
        KGEdge(
            id="causes:s1:param_a",
            source="signal:s1",
            target="param_delta:outcome_modifier:crisis:+1.000000",
            relation_type="causes",
            confidence=0.8,
            metadata={"param_name": "outcome_modifier:crisis", "delta_sign": 1, "signal_id": "s1", "world_step": 1},
        )
    )
    knowledge_graph.add_edge(
        KGEdge(
            id="propagates_to:s1:param_a:x",
            source="param_delta:outcome_modifier:crisis:+1.000000",
            target="variable:water_stock:t=1",
            relation_type="propagates_to",
            confidence=0.8,
            metadata={
                "param_name": "outcome_modifier:crisis",
                "resource_id": "water_stock",
                "variable_sign": -1,
                "signal_id": "s1",
                "world_step": 1,
            },
        )
    )
    knowledge_graph.add_edge(
        KGEdge(
            id="threshold_exceeded:s1:x:crisis",
            source="variable:water_stock:t=1",
            target="outcome:crisis:p=0.800000",
            relation_type="threshold_exceeded",
            confidence=0.8,
            metadata={
                "resource_id": "water_stock",
                "outcome_id": "crisis",
                "contribution_sign": 1,
                "signal_id": "s1",
                "world_step": 1,
            },
        )
    )
    knowledge_graph.add_edge(
        KGEdge(
            id="causes:s2:param_a",
            source="signal:s2",
            target="param_delta:outcome_modifier:crisis:-0.900000",
            relation_type="causes",
            confidence=0.6,
            metadata={"param_name": "outcome_modifier:crisis", "delta_sign": -1, "signal_id": "s2", "world_step": 2},
        )
    )

    forecast = Forecast(
        forecast_id="f-causal",
        domain_id="water",
        outcome_id="crisis",
        predicted_prob=0.8,
        session_id="s1",
        horizon_steps=3,
        created_at=datetime(2026, 3, 27, tzinfo=timezone.utc),
        created_step=0,
        causal_path=[
            "causes:s1:param_a",
            "propagates_to:s1:param_a:x",
            "threshold_exceeded:s1:x:crisis",
        ],
    )
    registry.record(forecast)
    pipeline.verify_forecast(
        forecast.forecast_id,
        actual_prob=0.4,
        verified_at=datetime(2026, 3, 30, tzinfo=timezone.utc),
        current_signal_id="s2",
        session_log=SessionLog(session_id="verify-refuted"),
    )

    explanation = pipeline.explain_forecast(forecast.forecast_id)

    assert explanation.status == "failed"
    assert explanation.causal_path_refuted == 1
    assert explanation.refuted_at_node == "param_delta:outcome_modifier:crisis:+1.000000"
    assert explanation.refutation_signal == "signal:s2"
    assert any(step.confirmed is False for step in explanation.causal_chain)
    assert "Contradicted by signal:s2" in explanation.to_text()


def test_explain_forecast_legacy(tmp_path) -> None:
    registry = ForecastRegistry(auto_load=False, auto_save=False)
    knowledge_graph = KnowledgeGraph(json_path=tmp_path / "kg.json", auto_load=False, auto_save=False)
    pipeline = AnalysisPipeline(
        knowledge_graph=knowledge_graph,
        forecast_registry=registry,
        sim_config=SimConfig(max_steps=1),
    )
    forecast = Forecast(
        forecast_id="legacy:1:outcome",
        domain_id="legacy",
        outcome_id="outcome",
        predicted_prob=0.42,
        session_id="legacy-session",
        horizon_steps=2,
        created_at=datetime(2026, 4, 13, tzinfo=timezone.utc),
        created_step=1,
        causal_path=[],
    )
    registry.record(forecast)

    explanation = pipeline.explain_forecast(forecast.forecast_id)

    assert explanation.causal_chain == []
    assert explanation.note is not None
    assert "Legacy forecast" in explanation.note


def test_update_records_parameter_effect_mismatch_when_probability_moves_opposite(tmp_path) -> None:
    knowledge_graph = KnowledgeGraph(json_path=tmp_path / "kg.json", auto_load=False, auto_save=False)
    pipeline = AnalysisPipeline(
        sim_config=SimConfig(max_steps=1, convergence_check_steps=25, convergence_epsilon=1.0e-5, seed=11),
        knowledge_graph=knowledge_graph,
    )
    schema = {
        "domain_id": "mismatch_case",
        "name": "Mismatch Case",
        "description": "Modifier intent can diverge from posterior after normalization.",
        "actors": [],
        "resources": [
            {
                "id": "signal",
                "name": "Signal",
                "value": 1.0,
                "unit": "u",
                "min_value": -10.0,
                "max_value": 10.0,
                "evolution_type": "linear",
                "evolution_params": {"a": 1.0, "b": 0.0, "c": 0.0, "coupling_weights": {}},
            }
        ],
        "relations": [],
        "outcomes": [
            {"id": "yes", "label": "YES", "scoring_weights": {"signal": 2.0}},
            {"id": "no", "label": "NO", "scoring_weights": {"signal": 1.0}},
        ],
        "causal_dag": [],
        "metadata": {},
    }
    world = pipeline.compiler.compile(schema)
    prior_result = pipeline.run(schema, session_log=SessionLog(session_id="mismatch-prior"))

    result = pipeline.update(
        world,
        ParameterVector(
            outcome_modifiers={"yes": 1.1, "no": 4.0},
            rationale="Stress should intensify both risks, especially NO.",
        ),
        signal_text="Insurance losses surged while offsetting support weakened.",
        session_log=SessionLog(session_id="mismatch-update"),
    )

    mismatches = result.metadata["parameter_effect_mismatches"]
    mismatch_nodes = [node for node in knowledge_graph.nodes() if node.node_type == "parameter_effect_conflict"]

    assert prior_result.simulation["final_outcome_probs"]["yes"] > result.simulation["final_outcome_probs"]["yes"]
    assert mismatches
    assert any(entry["outcome_id"] == "yes" for entry in mismatches)
    assert mismatch_nodes
    assert result.world.parameter_vector.conflict_flag is True
    assert mismatch_nodes[0].metadata["mismatch_count"] >= 1


def test_epistemic_log_is_queryable_by_domain_family_and_causal_chain(tmp_path, water_market_schema) -> None:
    water_market_schema["metadata"]["domain_family"] = "water_risk"
    water_market_schema["metadata"]["causal_chain"] = ["water_stock", "conflict_level"]
    registry = ForecastRegistry(auto_load=False, auto_save=False)
    knowledge_graph = KnowledgeGraph(json_path=tmp_path / "kg.json", auto_load=False, auto_save=False)
    pipeline = AnalysisPipeline(
        sim_config=SimConfig(max_steps=3, convergence_check_steps=100, convergence_epsilon=3.0e-2, seed=11),
        knowledge_graph=knowledge_graph,
        forecast_registry=registry,
    )
    pipeline.run(water_market_schema, session_log=SessionLog(session_id="verify-run"))
    forecast = registry.pending()[0]
    pipeline.verify_forecast(
        forecast.forecast_id,
        actual_prob=0.0,
        verified_at=datetime(2026, 3, 30, tzinfo=timezone.utc),
        session_log=SessionLog(session_id="verify-session"),
    )

    records = EpistemicLog(knowledge_graph).query(
        domain_family="water_risk",
        causal_chain=["conflict_level"],
    )

    assert records
    assert records[0].domain_family == "water_risk"
    assert "conflict_level" in records[0].causal_chain
