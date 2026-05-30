"""Tests for multi-agent hive-mind primitives."""

from __future__ import annotations

import math

import pytest

from freeman.agent.attentionscheduler import AttentionScheduler, AttentionTask
from freeman.agent.consciousness import ConsciousState
from freeman.core.types import TrailType
from freeman.memory.knowledgegraph import KGEdge, KGNode, KnowledgeGraph
from freeman.memory.reconciler import Reconciler
from freeman.memory.sessionlog import SessionLog
from freeman.memory.selfmodel import SelfModelGraph
from freeman.runtime.hive_runtime import (
    HIVE_ROLE_ORDER,
    HiveMindRuntime,
    HiveRuntimeConfig,
    RoleClientBinding,
    build_hive_role_clients,
)


def _state(kg: KnowledgeGraph) -> ConsciousState:
    return ConsciousState(
        world_ref="world:hive:0",
        self_model_ref=SelfModelGraph(kg),
        agent_role="planner",
        runtime_metadata={"schema_version": 1},
    )


def test_knowledge_graph_node_lock_and_unlock_with_ttl(tmp_path) -> None:
    kg = KnowledgeGraph(json_path=tmp_path / "kg.json", auto_load=False, auto_save=False)
    kg.add_node(KGNode(id="n1", label="Node 1", confidence=0.8))

    assert kg.try_lock("n1", "agent-a")
    assert not kg.try_lock("n1", "agent-b")
    assert not kg.unlock("n1", "agent-b")
    assert kg.unlock("n1", "agent-a")
    assert kg.try_lock("n1", "agent-b")

    kg.graph.nodes["n1"]["locked_at"] = float(kg.graph.nodes["n1"]["locked_at"]) - 20.0
    assert kg.try_lock("n1", "agent-c", lock_ttl_seconds=5.0)


def test_knowledge_graph_deposit_trail_updates_causal_edges_only(tmp_path) -> None:
    kg = KnowledgeGraph(json_path=tmp_path / "kg.json", auto_load=False, auto_save=False)
    for node_id in ("a", "b", "c"):
        kg.add_node(KGNode(id=node_id, label=node_id, confidence=0.7))
    kg.add_edge(KGEdge(id="e1", source="a", target="b", relation_type="causes", confidence=0.8, trail_weight=0.0))
    kg.add_edge(KGEdge(id="e2", source="b", target="c", relation_type="threshold_exceeded", confidence=0.8, trail_weight=0.0))

    updated = kg.deposit_trail(["e1", "e2"], quality=0.4)
    edge1 = kg.get_edge("e1")
    edge2 = kg.get_edge("e2")

    assert updated == 1
    assert edge1 is not None
    assert edge1.trail_weight == pytest.approx(0.4)
    assert edge2 is not None
    assert edge2.trail_weight == pytest.approx(0.0)


def test_knowledge_graph_update_node_decays_node_trail_metadata(tmp_path) -> None:
    kg = KnowledgeGraph(json_path=tmp_path / "kg.json", auto_load=False, auto_save=False)
    trail_type: TrailType = "ingest"
    kg.add_node(
        KGNode(
            id="n1",
            label="Node 1",
            confidence=0.8,
            metadata={"trail_type": trail_type, "trail_intensity": 1.0},
        )
    )

    kg.update_node(KGNode(id="n1", label="Node 1", confidence=0.85, metadata={"owner": "agent-a"}))
    updated = kg.get_node("n1")

    assert updated is not None
    assert updated.metadata["owner"] == "agent-a"
    assert updated.metadata["trail_type"] == "ingest"
    assert updated.metadata["trail_intensity"] == pytest.approx(0.9)


def test_knowledge_graph_update_node_evaporates_weak_node_trail_metadata(tmp_path) -> None:
    kg = KnowledgeGraph(json_path=tmp_path / "kg.json", auto_load=False, auto_save=False)
    kg.add_node(
        KGNode(
            id="n1",
            label="Node 1",
            confidence=0.8,
            metadata={"trail_type": "repair", "trail_intensity": 0.051},
        )
    )

    kg.update_node(KGNode(id="n1", label="Node 1", confidence=0.81))
    updated = kg.get_node("n1")

    assert updated is not None
    assert "trail_type" not in updated.metadata
    assert "trail_intensity" not in updated.metadata


def test_attention_scheduler_prefers_higher_trail_weight() -> None:
    scheduler = AttentionScheduler(attention_budget=10.0, ucb_beta=0.0)
    low_trail = AttentionTask(
        task_id="low",
        description="low trail",
        expected_information_gain=1.0,
        cost=1.0,
        trail_weight=0.1,
        pulls=1,
    )
    high_trail = AttentionTask(
        task_id="high",
        description="high trail",
        expected_information_gain=1.0,
        cost=1.0,
        trail_weight=2.0,
        pulls=1,
    )
    scheduler.add_task(low_trail)
    scheduler.add_task(high_trail)

    decision = scheduler.select_task()

    assert decision is not None
    assert decision.task_id == "high"


def test_reconciler_evaporates_trail_weights(tmp_path) -> None:
    kg = KnowledgeGraph(json_path=tmp_path / "kg.json", auto_load=False, auto_save=True)
    kg.add_node(KGNode(id="signal:s1", label="Signal", confidence=0.8))
    kg.add_node(KGNode(id="param:p1", label="Param", confidence=0.8))
    kg.add_edge(
        KGEdge(
            id="causes:s1:p1",
            source="signal:s1",
            target="param:p1",
            relation_type="causes",
            confidence=0.8,
            trail_weight=1.0,
        )
    )

    reconciler = Reconciler(gamma=math.log(2.0))
    result = reconciler.reconcile(kg, SessionLog(session_id="empty"))
    edge = kg.get_edge("causes:s1:p1")

    assert edge is not None
    assert edge.trail_weight == pytest.approx(0.5, rel=1.0e-3)
    assert result.kg_health["evaporated_trail_edges"] == 1


def test_hive_runtime_routes_raw_node_through_fixed_role_chain(tmp_path) -> None:
    kg = KnowledgeGraph(json_path=tmp_path / "kg.json", auto_load=False, auto_save=True)
    kg.add_node(KGNode(id="claim:raw", label="Raw claim", content="Unprocessed external signal.", confidence=0.8))
    state = _state(kg)
    runtime = HiveMindRuntime(
        state=state,
        knowledge_graph=kg,
        runtime_path=tmp_path,
        config=HiveRuntimeConfig(runtime_id="test-hive", llm_enabled=False),
    )

    report = runtime.run(cycles=1)
    node = kg.get_node("claim:raw")

    assert [action.role for action in report.actions] == list(HIVE_ROLE_ORDER)
    assert node is not None
    assert node.metadata["trail_type"] == "verified"
    assert [item["role"] for item in node.metadata["hive_history"]] == list(HIVE_ROLE_ORDER)
    assert node.metadata["hive_runtime"]["role_counts"] == {
        "ingestor": 1,
        "repairer": 1,
        "planner": 1,
        "narrator": 1,
        "verifier": 1,
    }
    assert state.agent_role == "planner"
    assert len(state.trace_state) == 5
    assert (tmp_path / "hive_checkpoint.json").exists()
    assert (tmp_path / "hive_event_log.jsonl").exists()


def test_hive_runtime_respects_role_revisit_limit(tmp_path) -> None:
    kg = KnowledgeGraph(json_path=tmp_path / "kg.json", auto_load=False, auto_save=False)
    kg.add_node(
        KGNode(
            id="claim:verified",
            label="Verified claim",
            confidence=0.8,
            metadata={"trail_type": "verified", "hive_runtime": {"role_counts": {"planner": 1}}},
        )
    )
    runtime = HiveMindRuntime(
        state=_state(kg),
        knowledge_graph=kg,
        runtime_path=tmp_path,
        config=HiveRuntimeConfig(role_order=("planner",), llm_enabled=False),
    )

    report = runtime.run(cycles=1)

    assert report.actions == []
    assert report.skipped["role_revisit_limit"] >= 1


def test_hive_runtime_uses_role_scoped_llm_when_enabled(tmp_path) -> None:
    class FakeChatClient:
        model = "fake-qwen"
        base_url = "memory://fake"

        def __init__(self) -> None:
            self.calls = 0

        def chat_text(self, messages, *, temperature, max_tokens):  # noqa: ANN001
            self.calls += 1
            assert messages[0]["role"] == "system"
            assert temperature == pytest.approx(0.1)
            assert max_tokens == 32
            return "structured narrator proposal"

    kg = KnowledgeGraph(json_path=tmp_path / "kg.json", auto_load=False, auto_save=False)
    kg.add_node(
        KGNode(
            id="claim:plan",
            label="Planner handoff",
            content="Candidate mechanism ready for narration.",
            confidence=0.8,
            metadata={"trail_type": "read_plan"},
        )
    )
    fake_client = FakeChatClient()
    runtime = HiveMindRuntime(
        state=_state(kg),
        knowledge_graph=kg,
        runtime_path=tmp_path,
        config=HiveRuntimeConfig(
            role_order=("narrator",),
            llm_enabled=True,
            llm_roles=("narrator",),
            llm_max_tokens=32,
        ),
        role_clients={
            "narrator": RoleClientBinding(
                role="narrator",
                provider="ollama",
                model="fake-qwen",
                base_url="memory://fake",
                client=fake_client,
            )
        },
    )

    report = runtime.run(cycles=1)
    node = kg.get_node("claim:plan")

    assert fake_client.calls == 1
    assert report.actions[0].llm_used is True
    assert report.actions[0].summary == "structured narrator proposal"
    assert node is not None
    assert node.metadata["trail_type"] == "llm_propose"
    assert node.metadata["hive_role_outputs"]["narrator"]["summary"] == "structured narrator proposal"


def test_build_hive_role_clients_accepts_qwen_and_openai_compatible_localhost(monkeypatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    config = {
        "agent_stack": {
            "role_order": ["narrator", "verifier"],
            "llm": {
                "role_models": {
                    "default": {
                        "provider": "qwen",
                        "model": "qwen2.5-coder:14b",
                        "base_url": "http://127.0.0.1:11434",
                    },
                    "verifier": {
                        "provider": "openai-compatible",
                        "model": "qwen2.5-coder:32b",
                        "base_url": "http://127.0.0.1:8000/v1",
                    },
                }
            },
        }
    }

    bindings = build_hive_role_clients(config)

    assert bindings["narrator"].provider == "ollama"
    assert bindings["narrator"].model == "qwen2.5-coder:14b"
    assert bindings["narrator"].available is True
    assert bindings["verifier"].provider == "openai-compatible"
    assert bindings["verifier"].model == "qwen2.5-coder:32b"
    assert bindings["verifier"].available is True
