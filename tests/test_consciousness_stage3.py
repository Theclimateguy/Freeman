"""Stage 3 tests for idle deliberation."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
import time

from freeman.agent.consciousness import ConsciousState, ConsciousnessEngine, IdleScheduler
from freeman.memory.knowledgegraph import KGNode, KnowledgeGraph
from freeman.memory.selfmodel import SelfModelEdge, SelfModelGraph, SelfModelNode


def _now() -> datetime:
    return datetime(2026, 4, 12, 12, 0, tzinfo=timezone.utc)


def _state(tmp_path) -> ConsciousState:
    kg = KnowledgeGraph(json_path=tmp_path / "kg.json", auto_load=False, auto_save=False)
    return ConsciousState(
        world_ref="world:test:0",
        self_model_ref=SelfModelGraph(kg),
        goal_state=[],
        attention_state={"water": 0.5, "macro": 0.5},
        trace_state=[],
        runtime_metadata={
            "last_update_at": _now().isoformat(),
            "idle_scheduler_stats": {
                "time_since_last_update": {"min": 0.0, "max": 10000.0},
                "confidence_gap": {"min": 0.0, "max": 1.0},
                "hypothesis_age": {"min": 0.0, "max": 100.0},
                "attention_deficit": {"min": 0.0, "max": 1.0},
            },
        },
    )


def _hypothesis(node_id: str, confidence: float, age_steps: float) -> SelfModelNode:
    return SelfModelNode(
        node_id=node_id,
        node_type="active_hypothesis",
        domain="water",
        payload={"age_steps": age_steps},
        confidence=confidence,
        created_at=_now(),
        updated_at=_now(),
        source_trace_id=None,
    )


def test_idle_score_increases_with_time(tmp_path) -> None:
    state = _state(tmp_path)
    scheduler = IdleScheduler({"idle_scheduler": {"threshold": 0.60}})

    early = scheduler.score(state, _now() + timedelta(seconds=100))
    later = scheduler.score(state, _now() + timedelta(seconds=5000))

    assert later > early


def test_idle_score_deterministic(tmp_path) -> None:
    state = _state(tmp_path)
    scheduler = IdleScheduler({"idle_scheduler": {"threshold": 0.60}})
    now = _now() + timedelta(seconds=2500)

    assert scheduler.score(state, now) == scheduler.score(state, now)


def test_hypothesis_aging_decays_confidence(tmp_path) -> None:
    state = _state(tmp_path)
    state.self_model_ref.write(_hypothesis("sm:active_hypothesis:h1", 0.8, 50.0), caller_token="freeman_consciousness_engine")
    engine = ConsciousnessEngine(state, {})

    diff = engine._hypothesis_aging()
    aged = [node for node in diff["nodes"] if node["node_type"] == "active_hypothesis"][0]

    assert aged["confidence"] < 0.8


def test_hypothesis_aging_migrates_at_threshold(tmp_path) -> None:
    state = _state(tmp_path)
    state.self_model_ref.write(_hypothesis("sm:active_hypothesis:h1", 0.051, 50.0), caller_token="freeman_consciousness_engine")
    engine = ConsciousnessEngine(state, {})

    diff = engine._hypothesis_aging()
    engine._apply_diff({key: value for key, value in diff.items() if key != "rationale"})

    assert not state.self_model_ref.get_nodes_by_type("active_hypothesis")
    uncertainty_nodes = state.self_model_ref.get_nodes_by_type("self_uncertainty")
    assert uncertainty_nodes


def test_consistency_check_emits_uncertainty_node(tmp_path) -> None:
    state = _state(tmp_path)
    left = _hypothesis("sm:active_hypothesis:left", 0.8, 10.0)
    right = _hypothesis("sm:active_hypothesis:right", 0.9, 12.0)
    state.self_model_ref.write(left, caller_token="freeman_consciousness_engine")
    state.self_model_ref.write(right, caller_token="freeman_consciousness_engine")
    state.self_model_ref.write(
        SelfModelEdge(
            edge_id="sm-edge:contradicts:left:right",
            edge_type="contradicts",
            source_id=left.node_id,
            target_id=right.node_id,
            weight=1.0,
            created_at=_now(),
            trace_id=None,
        ),
        caller_token="freeman_consciousness_engine",
    )
    engine = ConsciousnessEngine(state, {})

    diff = engine._consistency_check()

    assert any(node["node_type"] == "self_uncertainty" for node in diff["nodes"])


def test_maybe_deliberate_no_thread(tmp_path) -> None:
    state = _state(tmp_path)
    state.runtime_metadata["confidence_gap"] = 1.0
    state.self_model_ref.write(_hypothesis("sm:active_hypothesis:h1", 0.8, 80.0), caller_token="freeman_consciousness_engine")
    engine = ConsciousnessEngine(state, {"idle_scheduler": {"threshold": 0.10}})
    start = time.monotonic()

    result = engine.maybe_deliberate(_now() + timedelta(seconds=6000))
    elapsed = time.monotonic() - start

    assert result is state
    assert elapsed < 1.0


def test_anomaly_cluster_escalation(tmp_path) -> None:
    state = _state(tmp_path)
    state.runtime_metadata["last_runtime_step"] = 100
    kg = state.self_model_ref.knowledge_graph
    for index in range(3):
        kg.add_node(
            KGNode(
                id=f"anomaly_candidate:a{index}",
                label="Anomaly candidate: biosurveillance",
                node_type="anomaly_candidate",
                content="Novel biosurveillance anomaly from wastewater monitoring.",
                confidence=0.8,
                metadata={
                    "domain": "runtime",
                    "signal_id": f"a{index}",
                    "topic": "biosurveillance",
                    "text_snippet": "Novel biosurveillance anomaly from wastewater monitoring.",
                    "runtime_step": 95,
                    "reviewed": False,
                    "review_outcome": None,
                },
            )
        )

    engine = ConsciousnessEngine(
        state,
        {
            "idle_scheduler": {"threshold": 0.10},
            "anomaly_review": {"trigger_count": 3, "min_cluster_size": 3, "max_age_steps": 50},
        },
    )

    result = engine.maybe_deliberate(_now() + timedelta(seconds=6000))
    trait_nodes = [
        node
        for node in state.self_model_ref.get_nodes_by_type("identity_trait")
        if node.payload.get("trait_key") == "ontology_gap"
    ]
    anomaly_nodes = kg.query(node_type="anomaly_candidate", metadata_filters={"reviewed": True})

    assert result is state
    assert trait_nodes
    assert trait_nodes[0].payload["cluster_size"] == 3
    assert all(node.metadata["review_outcome"] == "escalate" for node in anomaly_nodes)


def test_anomaly_singleton_noise(tmp_path) -> None:
    state = _state(tmp_path)
    state.runtime_metadata["last_runtime_step"] = 100
    kg = state.self_model_ref.knowledge_graph
    kg.add_node(
        KGNode(
            id="anomaly_candidate:singleton",
            label="Anomaly candidate: stray_topic",
            node_type="anomaly_candidate",
            content="Completely novel singleton event.",
            confidence=0.8,
            metadata={
                "domain": "runtime",
                "signal_id": "singleton",
                "topic": "stray_topic",
                "text_snippet": "Completely novel singleton event.",
                "runtime_step": 1,
                "reviewed": False,
                "review_outcome": None,
            },
        )
    )
    engine = ConsciousnessEngine(
        state,
        {
            "anomaly_review": {"trigger_count": 1, "min_cluster_size": 3, "max_age_steps": 50},
        },
    )

    diff = engine._anomaly_review()
    reviewed = kg.get_node("anomaly_candidate:singleton", lazy_embed=False)

    assert diff["runtime_metadata"]["anomaly_review"]["noise_ids"] == ["anomaly_candidate:singleton"]
    assert reviewed is not None
    assert reviewed.metadata["reviewed"] is True
    assert reviewed.metadata["review_outcome"] == "noise"
