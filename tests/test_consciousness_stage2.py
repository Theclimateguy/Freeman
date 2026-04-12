"""Stage 2 tests for deterministic consciousness operators."""

from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone

import pytest

from freeman.agent.analysispipeline import AnalysisPipeline
from freeman.agent.consciousness import ConsciousState, ConsciousnessEngine, ENGINE_TOKEN
from freeman.memory.knowledgegraph import KGNode, KnowledgeGraph
from freeman.memory.selfmodel import SelfModelGraph, SelfModelNode


def _now() -> datetime:
    return datetime(2026, 4, 12, 12, 0, tzinfo=timezone.utc)


def _state_with_kg(tmp_path) -> tuple[KnowledgeGraph, ConsciousState]:
    kg = KnowledgeGraph(json_path=tmp_path / "kg.json", auto_load=False, auto_save=False)
    state = ConsciousState(
        world_ref="world:test:0",
        self_model_ref=SelfModelGraph(kg),
        goal_state=[],
        attention_state={},
        trace_state=[],
        runtime_metadata={},
    )
    return kg, state


def _self_observation(domain_id: str, mae: float, n_forecasts: int = 10) -> KGNode:
    return KGNode(
        id=f"self:forecast_error:{domain_id}:outcome",
        label=f"Self-model: {domain_id}/outcome",
        node_type="self_observation",
        content=f"n={n_forecasts} forecasts; MAE={mae:.4f}",
        confidence=0.9,
        metadata={
            "domain_id": domain_id,
            "outcome_id": "outcome",
            "mean_abs_error": mae,
            "bias": 0.0,
            "n_forecasts": n_forecasts,
        },
    )


def _pipeline_result_snapshot(result) -> dict:
    return {
        "world": result.world.snapshot(),
        "simulation": result.simulation,
        "verification": result.verification,
        "raw_scores": result.raw_scores,
        "dominant_outcome": result.dominant_outcome,
        "reconciliation": asdict(result.reconciliation) if result.reconciliation is not None else None,
        "metadata": result.metadata,
        "policy_ranking": [item.snapshot() for item in result.policy_ranking],
    }


def test_capability_review_decreases_on_high_mae(tmp_path) -> None:
    kg, state = _state_with_kg(tmp_path)
    kg.add_node(_self_observation("water", mae=0.9))
    state.self_model_ref.write(
        SelfModelNode(
            node_id="sm:self_capability:water",
            node_type="self_capability",
            domain="water",
            payload={"trait": "baseline"},
            confidence=0.95,
            created_at=_now(),
            updated_at=_now(),
            source_trace_id=None,
        ),
        caller_token=ENGINE_TOKEN,
    )
    engine = ConsciousnessEngine(state, {})

    diff = engine._capability_review()
    node = SelfModelNode.from_dict(diff["nodes"][0])

    assert node.confidence < 0.95


def test_capability_review_increases_on_low_mae(tmp_path) -> None:
    kg, state = _state_with_kg(tmp_path)
    kg.add_node(_self_observation("water", mae=0.05))
    state.self_model_ref.write(
        SelfModelNode(
            node_id="sm:self_capability:water",
            node_type="self_capability",
            domain="water",
            payload={"trait": "baseline"},
            confidence=0.20,
            created_at=_now(),
            updated_at=_now(),
            source_trace_id=None,
        ),
        caller_token=ENGINE_TOKEN,
    )
    engine = ConsciousnessEngine(state, {})

    diff = engine._capability_review()
    node = SelfModelNode.from_dict(diff["nodes"][0])

    assert node.confidence > 0.20


def test_attention_rebalance_weights_sum_to_one(tmp_path) -> None:
    kg, state = _state_with_kg(tmp_path)
    state.self_model_ref.write(
        SelfModelNode(
            node_id="sm:goal_state:water",
            node_type="goal_state",
            domain="water",
            payload={"urgency": 0.8},
            confidence=0.8,
            created_at=_now(),
            updated_at=_now(),
            source_trace_id=None,
        ),
        caller_token=ENGINE_TOKEN,
    )
    state.goal_state = ["sm:goal_state:water"]
    kg.add_node(_self_observation("water", mae=0.2))
    engine = ConsciousnessEngine(state, {})

    diff = engine._attention_rebalance()

    assert sum(diff["attention_state"].values()) == pytest.approx(1.0, abs=1.0e-6)


def test_attention_rebalance_increases_on_uncertainty(tmp_path) -> None:
    kg, state = _state_with_kg(tmp_path)
    state.self_model_ref.write(
        SelfModelNode(
            node_id="sm:self_uncertainty:water",
            node_type="self_uncertainty",
            domain="water",
            payload={"uncertainty": 0.9},
            confidence=0.9,
            created_at=_now(),
            updated_at=_now(),
            source_trace_id=None,
        ),
        caller_token=ENGINE_TOKEN,
    )
    state.self_model_ref.write(
        SelfModelNode(
            node_id="sm:self_uncertainty:macro",
            node_type="self_uncertainty",
            domain="macro",
            payload={"uncertainty": 0.1},
            confidence=0.1,
            created_at=_now(),
            updated_at=_now(),
            source_trace_id=None,
        ),
        caller_token=ENGINE_TOKEN,
    )
    engine = ConsciousnessEngine(state, {})

    diff = engine._attention_rebalance()

    assert diff["attention_state"]["water"] > diff["attention_state"]["macro"]


def test_trait_consolidation_weakens_on_mae_deterioration(tmp_path) -> None:
    _, state = _state_with_kg(tmp_path)
    state.self_model_ref.write(
        SelfModelNode(
            node_id="sm:identity_trait:conservative",
            node_type="identity_trait",
            domain="water",
            payload={"trait_support": 0.8, "pattern_observed": False, "delta_mae": 0.5},
            confidence=0.8,
            created_at=_now(),
            updated_at=_now(),
            source_trace_id=None,
        ),
        caller_token=ENGINE_TOKEN,
    )
    engine = ConsciousnessEngine(state, {})

    diff = engine._trait_consolidation()
    node = SelfModelNode.from_dict(diff["nodes"][0])

    assert float(node.payload["trait_support"]) < 0.8


def test_operator_writes_trace_event(tmp_path, water_market_schema) -> None:
    pipeline = AnalysisPipeline(
        knowledge_graph=KnowledgeGraph(json_path=tmp_path / "kg.json", auto_load=False, auto_save=True),
    )
    before = len(pipeline.conscious_state.trace_state)

    pipeline.run(water_market_schema)

    assert len(pipeline.conscious_state.trace_state) >= before + 1


def test_pipeline_projects_goal_state_and_active_hypotheses(tmp_path, water_market_schema) -> None:
    pipeline = AnalysisPipeline(
        knowledge_graph=KnowledgeGraph(json_path=tmp_path / "kg.json", auto_load=False, auto_save=True),
    )

    result = pipeline.run(water_market_schema)

    goal_nodes = pipeline.conscious_state.self_model_ref.get_nodes_by_type("goal_state")
    hypothesis_nodes = pipeline.conscious_state.self_model_ref.get_nodes_by_type("active_hypothesis")

    assert result.dominant_outcome is not None
    assert pipeline.conscious_state.goal_state
    assert goal_nodes
    assert hypothesis_nodes


def test_epistemic_refresh_projects_identity_traits_and_capability(tmp_path) -> None:
    kg, state = _state_with_kg(tmp_path)
    kg.add_node(_self_observation("water", mae=0.2, n_forecasts=12))
    engine = ConsciousnessEngine(state, {})

    refreshed = engine.refresh_after_epistemic_update(world_ref="world:water:12")

    capability_nodes = refreshed.self_model_ref.get_nodes_by_type("self_capability")
    trait_nodes = refreshed.self_model_ref.get_nodes_by_type("identity_trait")

    assert capability_nodes
    assert trait_nodes
    assert capability_nodes[0].domain == "water"
    assert trait_nodes[0].domain == "water"


def test_pipeline_output_unchanged(tmp_path, water_market_schema, monkeypatch) -> None:
    baseline_pipeline = AnalysisPipeline(
        knowledge_graph=KnowledgeGraph(json_path=tmp_path / "baseline.json", auto_load=False, auto_save=True),
    )
    monkeypatch.setattr(
        "freeman.agent.consciousness.ConsciousnessEngine.post_pipeline_update",
        lambda self, pipeline_result, kg: self.state,
    )
    baseline_result = baseline_pipeline.run(water_market_schema)
    baseline_snapshot = _pipeline_result_snapshot(baseline_result)

    monkeypatch.undo()

    pipeline = AnalysisPipeline(
        knowledge_graph=KnowledgeGraph(json_path=tmp_path / "integrated.json", auto_load=False, auto_save=True),
    )
    integrated_result = pipeline.run(water_market_schema)
    integrated_snapshot = _pipeline_result_snapshot(integrated_result)

    assert integrated_snapshot == baseline_snapshot
