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
    build_hive_runtime_from_config,
)
from freeman.runtime.lock_backend import FileSystemLockBackend, RedisLockBackend


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


def test_filesystem_lock_backend_blocks_cross_process_same_node(tmp_path) -> None:
    graph_path = tmp_path / "kg.json"
    lock_dir = tmp_path / "node_locks"
    first = KnowledgeGraph(
        json_path=graph_path,
        auto_load=False,
        auto_save=False,
        lock_backend=FileSystemLockBackend(lock_dir),
    )
    second = KnowledgeGraph(
        json_path=graph_path,
        auto_load=False,
        auto_save=False,
        lock_backend=FileSystemLockBackend(lock_dir),
    )
    for kg in (first, second):
        kg.add_node(KGNode(id="n1", label="Node 1", confidence=0.8))

    assert first.try_lock("n1", "agent-a", lock_ttl_seconds=120.0)
    assert not second.try_lock("n1", "agent-b", lock_ttl_seconds=120.0)
    assert not second.unlock("n1", "agent-b")
    assert first.unlock("n1", "agent-a")
    assert second.try_lock("n1", "agent-b", lock_ttl_seconds=120.0)


def test_filesystem_lock_backend_recovers_stale_lock(tmp_path) -> None:
    graph_path = tmp_path / "kg.json"
    lock_dir = tmp_path / "node_locks"
    first = KnowledgeGraph(
        json_path=graph_path,
        auto_load=False,
        auto_save=False,
        lock_backend=FileSystemLockBackend(lock_dir),
    )
    second = KnowledgeGraph(
        json_path=graph_path,
        auto_load=False,
        auto_save=False,
        lock_backend=FileSystemLockBackend(lock_dir),
    )
    for kg in (first, second):
        kg.add_node(KGNode(id="n1", label="Node 1", confidence=0.8))

    assert first.try_lock("n1", "agent-a", lock_ttl_seconds=120.0)
    lock_files = list(lock_dir.glob("*.lock"))
    assert len(lock_files) == 1
    payload = lock_files[0].read_text(encoding="utf-8")
    lock_files[0].write_text(payload.replace('"locked_at":', '"locked_at": -1000000, "old_locked_at":'), encoding="utf-8")

    assert second.try_lock("n1", "agent-b", lock_ttl_seconds=1.0)


class FakeRedisClient:
    def __init__(self) -> None:
        self.values: dict[str, str] = {}
        self.expirations: dict[str, int] = {}
        self.delete_calls: list[str] = []
        self.get_calls: list[str] = []
        self.registered_scripts: list[str] = []
        self.script_calls: list[tuple[str, list[str], list[str]]] = []

    def set(self, key: str, value: str, *, nx: bool = False, px: int | None = None):  # noqa: ANN001
        if nx and key in self.values:
            return False
        self.values[key] = value
        if px is not None:
            self.expirations[key] = int(px)
        return True

    def get(self, key: str):  # noqa: ANN001
        self.get_calls.append(key)
        value = self.values.get(key)
        return value.encode("utf-8") if value is not None else None

    def pexpire(self, key: str, px: int) -> bool:
        if key not in self.values:
            return False
        self.expirations[key] = int(px)
        return True

    def delete(self, key: str) -> int:
        self.delete_calls.append(key)
        existed = key in self.values
        self.values.pop(key, None)
        self.expirations.pop(key, None)
        return int(existed)

    def register_script(self, script: str):  # noqa: ANN001
        self.registered_scripts.append(script)

        def run(*, keys: list[str], args: list[str]) -> int:
            self.script_calls.append((script, list(keys), list(args)))
            key = keys[0]
            owner = args[0]
            if self.values.get(key) != owner:
                return 0
            self.values.pop(key, None)
            self.expirations.pop(key, None)
            return 1

        return run


def test_redis_lock_backend_uses_set_nx_px_and_owner_guard(tmp_path) -> None:
    client = FakeRedisClient()
    graph_path = tmp_path / "kg.json"
    first = KnowledgeGraph(
        json_path=graph_path,
        auto_load=False,
        auto_save=False,
        lock_backend=RedisLockBackend(client=client),
    )
    second = KnowledgeGraph(
        json_path=graph_path,
        auto_load=False,
        auto_save=False,
        lock_backend=RedisLockBackend(client=client),
    )
    for kg in (first, second):
        kg.add_node(KGNode(id="n1", label="Node 1", confidence=0.8))

    assert first.try_lock("n1", "agent-a", lock_ttl_seconds=2.5)
    assert not second.try_lock("n1", "agent-b", lock_ttl_seconds=2.5)
    assert first.try_lock("n1", "agent-a", lock_ttl_seconds=3.0)
    assert list(client.expirations.values())[-1] == 3000
    get_calls_before_unlock = len(client.get_calls)
    assert not second.unlock("n1", "agent-b")
    assert len(client.get_calls) == get_calls_before_unlock
    assert not client.delete_calls
    assert client.script_calls[-1][2] == ["agent-b"]
    assert first.unlock("n1", "agent-a")
    assert client.script_calls[-1][2] == ["agent-a"]
    assert second.try_lock("n1", "agent-b", lock_ttl_seconds=2.5)


def test_redis_lock_backend_force_unlock_uses_plain_delete(tmp_path) -> None:
    client = FakeRedisClient()
    graph_path = tmp_path / "kg.json"
    kg = KnowledgeGraph(
        json_path=graph_path,
        auto_load=False,
        auto_save=False,
        lock_backend=RedisLockBackend(client=client),
    )
    kg.add_node(KGNode(id="n1", label="Node 1", confidence=0.8))

    assert kg.try_lock("n1", "agent-a", lock_ttl_seconds=2.5)
    assert kg.unlock("n1", "agent-b", force=True)
    assert client.delete_calls
    assert not client.script_calls


def test_build_hive_runtime_from_config_selects_filesystem_lock_backend(tmp_path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
agent_stack:
  lock_backend: "filesystem"
  lock_filesystem_path: "./locks"
memory:
  json_path: "./kg.json"
  embedding_provider: "hashing"
  hashing_embedding_dimension: 32
  vector_store:
    enabled: false
runtime:
  runtime_path: "./runtime"
""",
        encoding="utf-8",
    )

    runtime = build_hive_runtime_from_config(config_path, resume=False)

    assert isinstance(runtime.knowledge_graph.lock_backend, FileSystemLockBackend)
    assert runtime.knowledge_graph.lock_backend.lock_dir == (tmp_path / "locks").resolve()


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


def test_hive_runtime_graceful_stop_checkpoints_partial_run(tmp_path) -> None:
    kg = KnowledgeGraph(json_path=tmp_path / "kg.json", auto_load=False, auto_save=True)
    kg.add_node(KGNode(id="claim:raw", label="Raw claim", content="Unprocessed external signal.", confidence=0.8))
    runtime = HiveMindRuntime(
        state=_state(kg),
        knowledge_graph=kg,
        runtime_path=tmp_path,
        config=HiveRuntimeConfig(runtime_id="test-hive-stop", llm_enabled=False),
    )
    original_process = runtime._process_node

    def _stop_after_first(role, node):  # noqa: ANN001
        action = original_process(role, node)
        runtime.request_stop()
        return action

    runtime._process_node = _stop_after_first  # type: ignore[method-assign]

    report = runtime.run(cycles=5)

    assert report.cycles == 1
    assert len(report.actions) == 1
    assert runtime.stop_requested is True
    assert (tmp_path / "hive_checkpoint.json").exists()


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


def test_hive_runtime_planner_uses_role_scoped_llm_when_enabled(tmp_path) -> None:
    class FakeChatClient:
        model = "fake-qwen"
        base_url = "memory://fake"

        def chat_text(self, messages, *, temperature, max_tokens):  # noqa: ANN001
            assert messages[0]["role"] == "system"
            assert temperature == pytest.approx(0.1)
            assert max_tokens == 48
            return "planner selected the dominant causal frontier"

    kg = KnowledgeGraph(json_path=tmp_path / "kg.json", auto_load=False, auto_save=False)
    kg.add_node(
        KGNode(
            id="claim:repair",
            label="Repaired mechanism",
            content="A repaired mechanism is ready for planner attention.",
            confidence=0.8,
            metadata={"trail_type": "repair"},
        )
    )
    runtime = HiveMindRuntime(
        state=_state(kg),
        knowledge_graph=kg,
        runtime_path=tmp_path,
        config=HiveRuntimeConfig(
            role_order=("planner",),
            llm_enabled=True,
            llm_roles=("planner",),
            llm_max_tokens=48,
        ),
        role_clients={
            "planner": RoleClientBinding(
                role="planner",
                provider="ollama",
                model="fake-qwen",
                base_url="memory://fake",
                client=FakeChatClient(),
            )
        },
    )

    report = runtime.run(cycles=1)
    node = kg.get_node("claim:repair")

    assert report.actions[0].llm_used is True
    assert report.actions[0].summary == "planner selected the dominant causal frontier"
    assert node is not None
    assert node.metadata["trail_type"] == "read_plan"
    assert node.metadata["hive_role_outputs"]["planner"]["summary"] == "planner selected the dominant causal frontier"


def test_hive_runtime_planner_missing_api_key_falls_back_to_deterministic_summary(monkeypatch, tmp_path) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    config = {
        "agent_stack": {
            "role_order": ["planner"],
            "llm": {
                "enabled": True,
                "roles": ["planner"],
                "role_models": {
                    "default": {
                        "provider": "openai-compatible",
                        "model": "gpt-4o-mini",
                        "base_url": "https://api.openai.com/v1",
                        "api_key_env": "OPENAI_API_KEY",
                    }
                },
            },
        }
    }
    bindings = build_hive_role_clients(config)
    kg = KnowledgeGraph(json_path=tmp_path / "kg.json", auto_load=False, auto_save=False)
    kg.add_node(KGNode(id="claim:verified", label="Verified claim", confidence=0.8, metadata={"trail_type": "verified"}))
    runtime = HiveMindRuntime(
        state=_state(kg),
        knowledge_graph=kg,
        runtime_path=tmp_path,
        config=HiveRuntimeConfig(role_order=("planner",), llm_enabled=True, llm_roles=("planner",)),
        role_clients=bindings,
    )

    report = runtime.run(cycles=1)

    assert bindings["planner"].available is False
    assert report.actions[0].llm_used is False
    assert report.actions[0].summary == "planner routed node claim:verified from verified to read_plan"
    assert report.actions[0].metadata["llm_error"] == "OPENAI_API_KEY or LLM_API_KEY is required for openai-compatible"


def test_hive_runtime_planner_llm_exception_records_error_and_falls_back(tmp_path) -> None:
    class RaisingChatClient:
        model = "broken-qwen"
        base_url = "memory://broken"

        def chat_text(self, messages, *, temperature, max_tokens):  # noqa: ANN001
            del messages, temperature, max_tokens
            raise RuntimeError("planner model unavailable")

    kg = KnowledgeGraph(json_path=tmp_path / "kg.json", auto_load=False, auto_save=False)
    kg.add_node(KGNode(id="claim:verified", label="Verified claim", confidence=0.8, metadata={"trail_type": "verified"}))
    runtime = HiveMindRuntime(
        state=_state(kg),
        knowledge_graph=kg,
        runtime_path=tmp_path,
        config=HiveRuntimeConfig(role_order=("planner",), llm_enabled=True, llm_roles=("planner",)),
        role_clients={
            "planner": RoleClientBinding(
                role="planner",
                provider="ollama",
                model="broken-qwen",
                base_url="memory://broken",
                client=RaisingChatClient(),
            )
        },
    )

    report = runtime.run(cycles=1)

    assert report.actions[0].llm_used is False
    assert report.actions[0].summary == "planner routed node claim:verified from verified to read_plan"
    assert report.actions[0].metadata["llm_error"] == "planner model unavailable"


def test_hive_runtime_deterministic_fallback_summary_format_is_stable(tmp_path) -> None:
    kg = KnowledgeGraph(json_path=tmp_path / "kg.json", auto_load=False, auto_save=False)
    kg.add_node(KGNode(id="claim:verified", label="Verified claim", confidence=0.8, metadata={"trail_type": "verified"}))
    runtime = HiveMindRuntime(
        state=_state(kg),
        knowledge_graph=kg,
        runtime_path=tmp_path,
        config=HiveRuntimeConfig(role_order=("planner",), llm_enabled=False),
    )

    report = runtime.run(cycles=1)

    assert report.actions[0].summary == "planner routed node claim:verified from verified to read_plan"


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
