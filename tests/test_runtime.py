"""Tests for local runtime checkpointing and cursor semantics."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from freeman.agent.analysispipeline import AnalysisPipeline
from freeman.agent.costmodel import BudgetLedger, CostModel, build_budget_policy
from freeman.agent.forecastregistry import ForecastRegistry
from freeman.agent.signalingestion import SignalIngestionEngine
from freeman.agent.signalingestion import ShockClassification, Signal, SignalTrigger
from freeman.game.runner import SimConfig
from freeman.core.types import ParameterVector
from freeman.agent.consciousness import ConsciousState
from freeman.interface.cli import main
from freeman.memory.knowledgegraph import KGNode, KnowledgeGraph
from freeman.memory.sessionlog import SessionLog
from freeman.memory.selfmodel import SelfModelGraph
from freeman.runtime.agent_runtime import AgentRuntime
from freeman.runtime.checkpoint import CheckpointManager
from freeman.runtime.queryengine import RuntimeAnswerEngine, load_runtime_artifacts
from freeman.runtime import stream_runtime
from freeman.runtime.event_log import EventLog
from freeman.runtime.stream import StreamCursorStore
import yaml


def _state(tmp_path) -> ConsciousState:
    kg = KnowledgeGraph(json_path=tmp_path / "kg.json", auto_load=False, auto_save=False)
    return ConsciousState(
        world_ref="world:test:0",
        self_model_ref=SelfModelGraph(kg),
        goal_state=[],
        attention_state={},
        trace_state=[],
        runtime_metadata={"schema_version": 1},
    )


def _signals() -> list[dict[str, str]]:
    return [
        {"signal_id": "s1", "timestamp": "2026-04-12T12:00:00+00:00"},
        {"signal_id": "s2", "timestamp": "2026-04-12T12:01:00+00:00"},
        {"signal_id": "s3", "timestamp": "2026-04-12T12:02:00+00:00"},
    ]


def test_cursor_dedup(tmp_path) -> None:
    cursor = StreamCursorStore()
    state = _state(tmp_path)
    runtime = AgentRuntime(state=state, signals=[_signals()[0], _signals()[0]], cursor_store=cursor, runtime_path=tmp_path)

    runtime.run_oneshot()

    assert cursor.is_committed("s1") is True
    assert state.runtime_metadata["processed_signals"] == ["s1"]


def test_cursor_survives_restart(tmp_path) -> None:
    cursor = StreamCursorStore()
    cursor.commit("s1")
    path = Path(tmp_path) / "cursors.json"
    cursor.save(path)

    restored = StreamCursorStore()
    restored.load(path)

    assert restored.is_committed("s1") is True


def test_checkpoint_atomic_write(tmp_path, monkeypatch) -> None:
    manager = CheckpointManager()
    state = _state(tmp_path)
    checkpoint_path = Path(tmp_path) / "checkpoint.json"
    manager.save(state, checkpoint_path)
    before = checkpoint_path.read_text(encoding="utf-8")

    def fail_replace(self, target):  # noqa: ANN001
        del target
        raise RuntimeError("rename failed")

    monkeypatch.setattr(Path, "replace", fail_replace)
    with pytest.raises(RuntimeError):
        manager.save(state, checkpoint_path)

    assert checkpoint_path.read_text(encoding="utf-8") == before


def test_checkpoint_round_trip(tmp_path) -> None:
    manager = CheckpointManager()
    state = _state(tmp_path)
    checkpoint_path = Path(tmp_path) / "checkpoint.json"

    manager.save(state, checkpoint_path)
    restored = manager.load(checkpoint_path)

    assert restored == state


def test_replay_from_checkpoint_plus_events(tmp_path) -> None:
    signals = _signals()
    original_state = _state(tmp_path / "original")
    original_runtime = AgentRuntime(state=original_state, signals=signals, runtime_path=tmp_path / "original")
    original_runtime.run_oneshot()

    manager = CheckpointManager()
    checkpoint_state = _state(tmp_path / "checkpoint")
    runtime_first_leg = AgentRuntime(state=checkpoint_state, signals=signals[:2], runtime_path=tmp_path / "checkpoint")
    runtime_first_leg.run_oneshot()
    checkpoint_path = Path(tmp_path) / "checkpoint" / "checkpoint.json"
    manager.save(runtime_first_leg.state, checkpoint_path)

    restored_state = manager.load(checkpoint_path)
    runtime_second_leg = AgentRuntime(state=restored_state, signals=signals[2:], runtime_path=tmp_path / "checkpoint")
    runtime_second_leg.run_oneshot()

    assert runtime_second_leg.state == original_runtime.state


def test_event_log_lookup_after_restart(tmp_path) -> None:
    runtime_path = Path(tmp_path) / "runtime"
    state = _state(tmp_path)
    runtime = AgentRuntime(state=state, signals=_signals()[:2], runtime_path=runtime_path)

    runtime.run_oneshot()

    event_log = EventLog(runtime_path / "event_log.jsonl")
    restored = EventLog(runtime_path / "event_log.jsonl")
    event = restored.lookup("trace:signal:s2")

    assert event_log.path.exists()
    assert event is not None
    assert event.event_id == "trace:signal:s2"


def test_replay_from_checkpoint_plus_event_log(tmp_path) -> None:
    signals = _signals()
    original_state = _state(tmp_path / "original")
    original_runtime = AgentRuntime(state=original_state, signals=signals, runtime_path=tmp_path / "original")
    original_runtime.run_oneshot()

    checkpoint_runtime_path = Path(tmp_path) / "checkpoint-runtime"
    partial_state = _state(tmp_path / "partial")
    partial_runtime = AgentRuntime(state=partial_state, signals=signals[:2], runtime_path=checkpoint_runtime_path)
    partial_runtime.run_oneshot()
    CheckpointManager().save(partial_runtime.state, checkpoint_runtime_path / "checkpoint.json")

    event_log = EventLog(checkpoint_runtime_path / "event_log.jsonl")
    event_log.append(original_runtime.state.trace_state[2])
    replayed = event_log.replay(CheckpointManager().load(checkpoint_runtime_path / "checkpoint.json"))

    assert replayed == original_runtime.state


def test_duplicate_signal_no_double_mutation(tmp_path) -> None:
    state = _state(tmp_path)
    runtime = AgentRuntime(
        state=state,
        signals=[_signals()[0], _signals()[0]],
        runtime_path=tmp_path,
    )

    runtime.run_oneshot()

    assert [event.event_id for event in state.trace_state] == ["trace:signal:s1"]


def test_explain_cli_resolves_from_event_log(tmp_path, capsys) -> None:
    runtime_path = Path(tmp_path) / "runtime"
    state = _state(tmp_path)
    runtime = AgentRuntime(state=state, signals=_signals()[:1], runtime_path=runtime_path)
    runtime.run_oneshot()
    CheckpointManager().save(state, runtime_path / "checkpoint.json")

    config_path = Path(tmp_path) / "config.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "memory": {
                    "json_path": str(Path(tmp_path) / "kg.json"),
                    "session_log_path": str(Path(tmp_path) / "sessions"),
                },
                "runtime": {
                    "runtime_path": str(runtime_path),
                    "event_log_path": str(runtime_path / "event_log.jsonl"),
                },
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    exit_code = main(["--config", str(config_path), "explain", "--trace-id", "trace:signal:s1"])
    captured = capsys.readouterr()
    payload = json.loads(captured.out)

    assert exit_code == 0
    assert payload["found"] is True
    assert payload["explanation"]


def test_stream_runtime_llm_bootstrap_falls_back_to_schema(tmp_path, monkeypatch) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    fallback_schema = repo_root / "freeman" / "domain" / "profiles" / "gim15.json"
    runtime_path = Path(tmp_path) / "runtime"
    kg_path = Path(tmp_path) / "kg.json"
    config_path = Path(tmp_path) / "config.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "agent": {
                    "bootstrap": {
                        "mode": "llm_synthesize",
                        "fallback_schema_path": str(fallback_schema),
                        "domain_brief": "Compact climate-risk domain for deterministic fallback testing.",
                    },
                    "sources": [{"type": "rss", "url": "https://example.invalid/feed.xml"}],
                },
                "llm": {
                    "provider": "ollama",
                    "model": "fake-model",
                    "base_url": "http://127.0.0.1:11434",
                    "timeout_seconds": 1.0,
                },
                "memory": {"json_path": str(kg_path)},
                "runtime": {
                    "runtime_path": str(runtime_path),
                    "event_log_path": str(runtime_path / "event_log.jsonl"),
                },
                "sim": {"max_steps": 2},
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    attempt_log = [
        {"attempt": 1, "phase": "compile", "verifier_error": "missing field: outcomes", "repair_stage": "standard"},
        {
            "attempt": 2,
            "phase": "level1",
            "verifier_error": "spectral_radius: jacobian.spectral_radius | unstable dynamics",
            "repair_stage": "accumulated",
        },
    ]

    class _FakeClient:
        def repair_schema(self, *args, **kwargs):  # noqa: ANN002, ANN003
            raise AssertionError("repair_schema should not be called directly in fallback test")

    def _failing_compile_and_repair(self, *args, **kwargs):  # noqa: ANN002, ANN003
        self.last_bootstrap_attempts = json.loads(json.dumps(attempt_log))
        raise RuntimeError("synthetic bootstrap failure")

    monkeypatch.setattr(stream_runtime, "_build_chat_client", lambda **kwargs: _FakeClient())
    monkeypatch.setattr(stream_runtime, "_source_configs", lambda config, default_sources=None: [])
    monkeypatch.setattr(
        stream_runtime.FreemanOrchestrator,
        "compile_and_repair",
        _failing_compile_and_repair,
    )

    exit_code = stream_runtime.main(
        [
            "--config-path",
            str(config_path),
            "--hours",
            "0.00005",
            "--analysis-interval-seconds",
            "0.1",
            "--poll-seconds",
            "60",
            "--log-level",
            "WARNING",
        ]
    )

    payload = json.loads((runtime_path / "bootstrap_package.json").read_text(encoding="utf-8"))
    assert exit_code == 0
    assert payload["bootstrap_mode"] == "llm_synthesize_fallback"
    assert payload["bootstrap_attempts"] == attempt_log


def test_stream_runtime_prefers_domain_brief_bootstrap_when_mode_omitted(
    tmp_path,
    monkeypatch,
    water_market_schema,
) -> None:
    runtime_path = Path(tmp_path) / "runtime"
    kg_path = Path(tmp_path) / "kg.json"
    schema_path = Path(tmp_path) / "schema.json"
    schema_path.write_text(json.dumps(water_market_schema), encoding="utf-8")
    brief_path = Path(tmp_path) / "domain_brief.md"
    brief_path.write_text("Compact water-risk domain synthesized from a local brief.", encoding="utf-8")
    config_path = Path(tmp_path) / "config.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "agent": {
                    "bootstrap": {
                        "schema_path": str(schema_path),
                        "domain_brief_path": str(brief_path),
                    },
                    "sources": [],
                },
                "llm": {
                    "provider": "ollama",
                    "model": "fake-model",
                    "base_url": "http://127.0.0.1:11434",
                    "timeout_seconds": 1.0,
                },
                "memory": {"json_path": str(kg_path)},
                "runtime": {
                    "runtime_path": str(runtime_path),
                    "event_log_path": str(runtime_path / "event_log.jsonl"),
                },
                "sim": {"max_steps": 1},
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    class _FakeClient:
        def repair_schema(self, *args, **kwargs):  # noqa: ANN002, ANN003
            raise AssertionError("runtime should call orchestrator.compile_and_repair directly in this test")

    def _successful_compile_and_repair(self, domain_description, *args, **kwargs):  # noqa: ANN002, ANN003
        assert "local brief" in domain_description
        self.last_bootstrap_attempts = []
        return (
            {"schema": water_market_schema, "policies": [], "assumptions": ["synthetic success"]},
            "water_market:bootstrap",
            1,
            [],
        )

    monkeypatch.setattr(stream_runtime, "_build_chat_client", lambda **kwargs: _FakeClient())
    monkeypatch.setattr(stream_runtime, "_source_configs", lambda config, default_sources=None: [])
    monkeypatch.setattr(
        stream_runtime.FreemanOrchestrator,
        "compile_and_repair",
        _successful_compile_and_repair,
    )

    exit_code = stream_runtime.main(
        [
            "--config-path",
            str(config_path),
            "--hours",
            "0.00001",
            "--analysis-interval-seconds",
            "0.1",
            "--poll-seconds",
            "60",
            "--log-level",
            "WARNING",
        ]
    )

    payload = json.loads((runtime_path / "bootstrap_package.json").read_text(encoding="utf-8"))
    assert exit_code == 0
    assert payload["bootstrap_mode"] == "llm_synthesize"
    assert payload["schema"]["domain_id"] == water_market_schema["domain_id"]


def test_stream_runtime_runtime_step_verifies_forecasts_across_fallback(tmp_path, monkeypatch, water_market_schema) -> None:
    runtime_path = Path(tmp_path) / "runtime"
    kg_path = Path(tmp_path) / "kg.json"
    schema_path = Path(tmp_path) / "schema.json"
    schema_path.write_text(json.dumps(water_market_schema), encoding="utf-8")
    config_path = Path(tmp_path) / "config.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "agent": {
                    "bootstrap": {
                        "mode": "schema_path",
                        "schema_path": str(schema_path),
                    },
                    "sources": [{"type": "rss", "url": "https://example.invalid/feed.xml"}],
                },
                "llm": {
                    "provider": "ollama",
                    "model": "fake-model",
                    "base_url": "http://127.0.0.1:11434",
                    "timeout_seconds": 1.0,
                },
                "memory": {"json_path": str(kg_path)},
                "runtime": {
                    "runtime_path": str(runtime_path),
                    "event_log_path": str(runtime_path / "event_log.jsonl"),
                },
                "sim": {"max_steps": 1, "convergence_check_steps": 5, "convergence_epsilon": 1.0e-4},
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    signals = [
        Signal(
            signal_id="s1",
            source_type="manual",
            text="Major water conflict risk escalates.",
            topic="water_risk",
            timestamp="2026-04-12T12:00:00+00:00",
        ),
        Signal(
            signal_id="s2",
            source_type="manual",
            text="Second severe water conflict escalation.",
            topic="water_risk",
            timestamp="2026-04-12T12:01:00+00:00",
        ),
    ]

    class _Source:
        def __init__(self, values):
            self._values = list(values)
            self._done = False

        def fetch(self):
            if self._done:
                return []
            self._done = True
            return list(self._values)

    class _FakeClient:
        def chat_json(self, *args, **kwargs):  # noqa: ANN002, ANN003
            return {
                "shock_type": "routine",
                "severity": 0.2,
                "semantic_gap": 0.1,
                "rationale": "test stub",
            }

    monkeypatch.setattr(stream_runtime, "_build_chat_client", lambda **kwargs: _FakeClient())
    monkeypatch.setattr(stream_runtime, "_source_configs", lambda config, default_sources=None: [{"type": "manual"}])
    monkeypatch.setattr(stream_runtime, "build_signal_source", lambda cfg: _Source(signals))
    monkeypatch.setattr(
        stream_runtime.SignalIngestionEngine,
        "ingest",
        lambda self, source, classifier=None, signal_memory=None, skip_duplicates_within_hours=1.0: [
            SignalTrigger(
                signal_id=signals[0].signal_id,
                mahalanobis_score=1.0,
                classification=ShockClassification("shock", 0.9, 0.9),
                mode="ANALYZE",
            )
        ],
    )
    monkeypatch.setattr(
        stream_runtime.ParameterEstimator,
        "estimate",
        lambda self, world, text: ParameterVector(outcome_modifiers={"water_crisis": 1.2}, rationale=text),
    )

    original_update = stream_runtime.AnalysisPipeline.update
    state = {"fallback_triggered": False}

    def _update_with_forced_fallback(self, previous_world, parameter_vector, **kwargs):  # noqa: ANN001
        if not state["fallback_triggered"]:
            state["fallback_triggered"] = True
            raise RuntimeError("force fallback for runtime_step test")
        return original_update(self, previous_world, parameter_vector, **kwargs)

    monkeypatch.setattr(stream_runtime.AnalysisPipeline, "update", _update_with_forced_fallback)

    exit_code = stream_runtime.main(
        [
            "--config-path",
            str(config_path),
            "--hours",
            "0.0002",
            "--analysis-interval-seconds",
            "0.1",
            "--poll-seconds",
            "60",
            "--log-level",
            "WARNING",
        ]
    )

    world_state = json.loads((runtime_path / "world_state.json").read_text(encoding="utf-8"))
    forecasts = ForecastRegistry(json_path=runtime_path / "forecasts.json", auto_load=True, auto_save=False)
    kg = KnowledgeGraph(json_path=kg_path, auto_load=True, auto_save=False)
    self_observations = kg.query(node_type="self_observation")

    assert exit_code == 0
    assert world_state["runtime_step"] >= 1
    assert any(forecast["status"] == "verified" for forecast in forecasts.snapshot())
    assert self_observations


def test_stream_runtime_fallback_preserves_monotonic_world_step(tmp_path, monkeypatch, water_market_schema) -> None:
    runtime_path = Path(tmp_path) / "runtime"
    kg_path = Path(tmp_path) / "kg.json"
    schema_path = Path(tmp_path) / "schema.json"
    schema_path.write_text(json.dumps(water_market_schema), encoding="utf-8")
    config_path = Path(tmp_path) / "config.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "agent": {
                    "bootstrap": {
                        "mode": "schema_path",
                        "schema_path": str(schema_path),
                    },
                    "sources": [{"type": "rss", "url": "https://example.invalid/feed.xml"}],
                },
                "llm": {
                    "provider": "ollama",
                    "model": "fake-model",
                    "base_url": "http://127.0.0.1:11434",
                    "timeout_seconds": 1.0,
                },
                "memory": {"json_path": str(kg_path)},
                "runtime": {
                    "runtime_path": str(runtime_path),
                    "event_log_path": str(runtime_path / "event_log.jsonl"),
                },
                "sim": {"max_steps": 1, "convergence_check_steps": 5, "convergence_epsilon": 1.0e-4},
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    signals = [
        Signal(
            signal_id="s1",
            source_type="manual",
            text="Major water conflict risk escalates.",
            topic="water_risk",
            timestamp="2026-04-12T12:00:00+00:00",
        ),
        Signal(
            signal_id="s2",
            source_type="manual",
            text="Second severe water conflict escalation.",
            topic="water_risk",
            timestamp="2026-04-12T12:01:00+00:00",
        ),
    ]

    class _Source:
        def __init__(self, values):
            self._values = list(values)
            self._done = False

        def fetch(self):
            if self._done:
                return []
            self._done = True
            return list(self._values)

    class _FakeClient:
        def chat_json(self, *args, **kwargs):  # noqa: ANN002, ANN003
            return {
                "shock_type": "routine",
                "severity": 0.2,
                "semantic_gap": 0.1,
                "rationale": "test stub",
            }

    monkeypatch.setattr(stream_runtime, "_build_chat_client", lambda **kwargs: _FakeClient())
    monkeypatch.setattr(stream_runtime, "_source_configs", lambda config, default_sources=None: [{"type": "manual"}])
    monkeypatch.setattr(stream_runtime, "build_signal_source", lambda cfg: _Source(signals))
    monkeypatch.setattr(
        stream_runtime.SignalIngestionEngine,
        "ingest",
        lambda self, source, classifier=None, signal_memory=None, skip_duplicates_within_hours=1.0: [
            SignalTrigger(
                signal_id=signals[0].signal_id,
                mahalanobis_score=1.0,
                classification=ShockClassification("shock", 0.9, 0.9),
                mode="ANALYZE",
            )
        ],
    )
    monkeypatch.setattr(
        stream_runtime.ParameterEstimator,
        "estimate",
        lambda self, world, text: ParameterVector(outcome_modifiers={"water_crisis": 1.2}, rationale=text),
    )

    original_update = stream_runtime.AnalysisPipeline.update
    state = {"call_count": 0}

    def _update_with_second_call_fallback(self, previous_world, parameter_vector, **kwargs):  # noqa: ANN001
        state["call_count"] += 1
        if state["call_count"] == 2:
            raise RuntimeError("force fallback after first successful update")
        return original_update(self, previous_world, parameter_vector, **kwargs)

    monkeypatch.setattr(stream_runtime.AnalysisPipeline, "update", _update_with_second_call_fallback)

    exit_code = stream_runtime.main(
        [
            "--config-path",
            str(config_path),
            "--hours",
            "0.0002",
            "--analysis-interval-seconds",
            "0.1",
            "--poll-seconds",
            "60",
            "--log-level",
            "WARNING",
        ]
    )

    world_state = json.loads((runtime_path / "world_state.json").read_text(encoding="utf-8"))
    events = [json.loads(line) for line in (runtime_path / "event_log.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    world_steps = [
        event.get("diff", {}).get("runtime_metadata", {}).get("last_world_step")
        for event in events
        if event.get("operator") == "sync_runtime_metrics"
    ]

    assert exit_code == 0
    assert len(world_steps) >= 2
    assert world_steps[-1] == world_state["t"]
    assert world_steps[-1] > world_steps[-2]


def test_stream_runtime_persists_bootstrap_attempts_on_success(tmp_path, monkeypatch, water_market_schema) -> None:
    runtime_path = Path(tmp_path) / "runtime"
    kg_path = Path(tmp_path) / "kg.json"
    config_path = Path(tmp_path) / "config.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "agent": {
                    "bootstrap": {
                        "mode": "llm_synthesize",
                        "domain_brief": "Compact water-risk domain.",
                    },
                    "sources": [],
                },
                "llm": {
                    "provider": "ollama",
                    "model": "fake-model",
                    "base_url": "http://127.0.0.1:11434",
                    "timeout_seconds": 1.0,
                },
                "memory": {"json_path": str(kg_path)},
                "runtime": {
                    "runtime_path": str(runtime_path),
                    "event_log_path": str(runtime_path / "event_log.jsonl"),
                },
                "sim": {"max_steps": 1},
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    attempt_log = [
        {"attempt": 1, "phase": "compile", "verifier_error": "missing field: outcomes", "repair_stage": "standard"},
        {"attempt": 2, "phase": "level1", "verifier_error": "sign_consistency: causal edge", "repair_stage": "standard"},
    ]

    class _FakeClient:
        def repair_schema(self, *args, **kwargs):  # noqa: ANN002, ANN003
            raise AssertionError("runtime should call orchestrator.compile_and_repair directly in this test")

    def _successful_compile_and_repair(self, *args, **kwargs):  # noqa: ANN002, ANN003
        self.last_bootstrap_attempts = json.loads(json.dumps(attempt_log))
        return (
            {"schema": water_market_schema, "policies": [], "assumptions": ["synthetic success"]},
            "water_market:bootstrap",
            3,
            json.loads(json.dumps(attempt_log)),
        )

    monkeypatch.setattr(stream_runtime, "_build_chat_client", lambda **kwargs: _FakeClient())
    monkeypatch.setattr(stream_runtime, "_source_configs", lambda config, default_sources=None: [])
    monkeypatch.setattr(
        stream_runtime.FreemanOrchestrator,
        "compile_and_repair",
        _successful_compile_and_repair,
    )

    exit_code = stream_runtime.main(
        [
            "--config-path",
            str(config_path),
            "--hours",
            "0.00001",
            "--analysis-interval-seconds",
            "0.1",
            "--poll-seconds",
            "60",
            "--log-level",
            "WARNING",
        ]
    )

    payload = json.loads((runtime_path / "bootstrap_package.json").read_text(encoding="utf-8"))
    checkpoint_payload = json.loads((runtime_path / "checkpoint.json").read_text(encoding="utf-8"))
    assert exit_code == 0
    assert payload["bootstrap_mode"] == "llm_synthesize"
    assert payload["schema"]["domain_id"] == water_market_schema["domain_id"]
    assert payload["bootstrap_attempts"] == attempt_log
    assert payload["synthesis_attempts"] == 3
    assert "kg_health" in checkpoint_payload["state"]["runtime_metadata"]


def test_stream_runtime_hard_filter_drops_noisy_signals_before_event_log(tmp_path, monkeypatch, water_market_schema) -> None:
    runtime_path = Path(tmp_path) / "runtime"
    kg_path = Path(tmp_path) / "kg.json"
    schema_path = Path(tmp_path) / "schema.json"
    schema_path.write_text(json.dumps(water_market_schema), encoding="utf-8")
    config_path = Path(tmp_path) / "config.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "agent": {
                    "bootstrap": {"mode": "schema_path", "schema_path": str(schema_path)},
                    "stream_keywords": ["climate", "emissions", "adaptation"],
                    "stream_filter": {"min_keyword_matches": 2, "min_relevance_score": 0.05},
                    "sources": [{"type": "rss", "url": "https://example.invalid/feed.xml"}],
                },
                "llm": {"provider": "ollama", "model": "fake-model", "base_url": "http://127.0.0.1:11434", "timeout_seconds": 1.0},
                "memory": {"json_path": str(kg_path)},
                "runtime": {"runtime_path": str(runtime_path), "event_log_path": str(runtime_path / "event_log.jsonl")},
                "sim": {"max_steps": 1},
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    signals = [
        Signal(
            signal_id="noise-1",
            source_type="manual",
            text="Cartoon review and gold mine lifestyle feature.",
            topic="noise",
            timestamp="2026-04-12T12:00:00+00:00",
        ),
        Signal(
            signal_id="relevant-1",
            source_type="manual",
            text="Climate emissions adaptation policy update.",
            topic="climate_news",
            timestamp="2026-04-12T12:01:00+00:00",
        ),
    ]

    class _Source:
        def __init__(self, values):
            self._values = list(values)
            self._done = False

        def fetch(self):
            if self._done:
                return []
            self._done = True
            return list(self._values)

    class _FakeClient:
        def chat_json(self, *args, **kwargs):  # noqa: ANN002, ANN003
            return {
                "shock_type": "routine",
                "severity": 0.2,
                "semantic_gap": 0.1,
                "rationale": "test stub",
            }

    monkeypatch.setattr(stream_runtime, "_build_chat_client", lambda **kwargs: _FakeClient())
    monkeypatch.setattr(stream_runtime, "_source_configs", lambda config, default_sources=None: [{"type": "manual"}])
    monkeypatch.setattr(stream_runtime, "build_signal_source", lambda cfg: _Source(signals))

    exit_code = stream_runtime.main(
        [
            "--config-path",
            str(config_path),
            "--hours",
            "0.00002",
            "--analysis-interval-seconds",
            "0.1",
            "--poll-seconds",
            "60",
            "--log-level",
            "WARNING",
        ]
    )

    events = [json.loads(line) for line in (runtime_path / "event_log.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    signal_ids = [event.get("diff", {}).get("signal_id") for event in events if event.get("operator") == "runtime_signal_ingest"]

    assert exit_code == 0
    assert "noise-1" not in signal_ids
    assert "relevant-1" in signal_ids


def test_stream_runtime_agent_relevance_soft_rejects_and_updates_self_model(tmp_path, monkeypatch, water_market_schema) -> None:
    runtime_path = Path(tmp_path) / "runtime"
    kg_path = Path(tmp_path) / "kg.json"
    schema_path = Path(tmp_path) / "schema.json"
    schema_path.write_text(json.dumps(water_market_schema), encoding="utf-8")
    config_path = Path(tmp_path) / "config.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "agent": {
                    "bootstrap": {"mode": "schema_path", "schema_path": str(schema_path)},
                    "stream_filter": {"min_relevance_score": 0.0, "agent_min_relevance_score": 0.25},
                    "sources": [{"type": "rss", "url": "https://example.invalid/feed.xml"}],
                },
                "llm": {"provider": "ollama", "model": "fake-model", "base_url": "http://127.0.0.1:11434", "timeout_seconds": 1.0},
                "memory": {"json_path": str(kg_path)},
                "runtime": {"runtime_path": str(runtime_path), "event_log_path": str(runtime_path / "event_log.jsonl")},
                "sim": {"max_steps": 1},
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    kg = KnowledgeGraph(json_path=kg_path, auto_load=False, auto_save=True)
    kg.add_node(
        KGNode(
            id="self:forecast_error:water_market:cooperation",
            label="Self-model: water_market/cooperation",
            node_type="self_observation",
            content="n=12 forecasts; MAE=0.12; bias=+0.01",
            confidence=0.9,
            metadata={
                "domain_id": "water_market",
                "outcome_id": "cooperation",
                "mean_abs_error": 0.12,
                "bias": 0.01,
                "n_forecasts": 12,
                "errors_json": json.dumps([0.1, -0.08, 0.12]),
            },
        )
    )

    signals = [
        Signal(
            signal_id="noise-2",
            source_type="manual",
            text="Water cooperation celebrity cartoon gala and restaurant awards.",
            topic="noise",
            timestamp="2026-04-12T12:00:00+00:00",
        ),
    ]

    class _Source:
        def __init__(self, values):
            self._values = list(values)
            self._done = False

        def fetch(self):
            if self._done:
                return []
            self._done = True
            return list(self._values)

    class _FakeClient:
        def chat_json(self, *args, **kwargs):  # noqa: ANN002, ANN003
            return {}

    monkeypatch.setattr(stream_runtime, "_build_chat_client", lambda **kwargs: _FakeClient())
    monkeypatch.setattr(stream_runtime, "_source_configs", lambda config, default_sources=None: [{"type": "manual"}])
    monkeypatch.setattr(stream_runtime, "build_signal_source", lambda cfg: _Source(signals))

    exit_code = stream_runtime.main(
        [
            "--config-path",
            str(config_path),
            "--hours",
            "0.00002",
            "--analysis-interval-seconds",
            "0.1",
            "--poll-seconds",
            "60",
            "--log-level",
            "WARNING",
        ]
    )

    events = [json.loads(line) for line in (runtime_path / "event_log.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    filtered_events = [event for event in events if event.get("diff", {}).get("signal_id") == "noise-2"]
    refreshed_kg = KnowledgeGraph(json_path=kg_path, auto_load=True, auto_save=True)
    relevance_nodes = refreshed_kg.query(text="stream_relevance", node_type="self_capability")

    assert exit_code == 0
    assert filtered_events
    assert filtered_events[-1]["diff"]["filtered_out"] is True
    assert filtered_events[-1]["diff"]["filter_phase"] == "agent"
    assert relevance_nodes


def test_anomaly_candidate_preserved(tmp_path, monkeypatch, water_market_schema) -> None:
    runtime_path = Path(tmp_path) / "runtime"
    kg_path = Path(tmp_path) / "kg.json"
    schema_path = Path(tmp_path) / "schema.json"
    schema_path.write_text(json.dumps(water_market_schema), encoding="utf-8")
    config_path = Path(tmp_path) / "config.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "agent": {
                    "bootstrap": {"mode": "schema_path", "schema_path": str(schema_path)},
                    "stream_filter": {
                        "min_relevance_score": 0.0,
                        "agent_min_relevance_score": 0.25,
                        "anomaly_review_trigger_count": 10,
                    },
                    "sources": [{"type": "rss", "url": "https://example.invalid/feed.xml"}],
                },
                "llm": {"provider": "ollama", "model": "fake-model", "base_url": "http://127.0.0.1:11434", "timeout_seconds": 1.0},
                "memory": {"json_path": str(kg_path)},
                "runtime": {"runtime_path": str(runtime_path), "event_log_path": str(runtime_path / "event_log.jsonl")},
                "sim": {"max_steps": 1},
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    kg = KnowledgeGraph(json_path=kg_path, auto_load=False, auto_save=True)
    kg.add_node(
        KGNode(
            id="self:forecast_error:water_market:cooperation",
            label="Self-model: water_market/cooperation",
            node_type="self_observation",
            content="n=12 forecasts; MAE=0.12; bias=+0.01",
            confidence=0.9,
            metadata={
                "domain_id": "water_market",
                "outcome_id": "cooperation",
                "mean_abs_error": 0.12,
                "bias": 0.01,
                "n_forecasts": 12,
                "errors_json": json.dumps([0.1, -0.08, 0.12]),
            },
        )
    )

    signals = [
        Signal(
            signal_id="anomaly-1",
            source_type="manual",
            text="Novel zoonotic biosurveillance alert from polar wastewater sampling network.",
            topic="biosurveillance",
            timestamp="2026-04-12T12:00:00+00:00",
        ),
    ]

    class _Source:
        def __init__(self, values):
            self._values = list(values)
            self._done = False

        def fetch(self):
            if self._done:
                return []
            self._done = True
            return list(self._values)

    class _FakeClient:
        def chat_json(self, *args, **kwargs):  # noqa: ANN002, ANN003
            return {}

    monkeypatch.setattr(stream_runtime, "_build_chat_client", lambda **kwargs: _FakeClient())
    monkeypatch.setattr(stream_runtime, "_source_configs", lambda config, default_sources=None: [{"type": "manual"}])
    monkeypatch.setattr(stream_runtime, "build_signal_source", lambda cfg: _Source(signals))

    exit_code = stream_runtime.main(
        [
            "--config-path",
            str(config_path),
            "--hours",
            "0.00002",
            "--analysis-interval-seconds",
            "0.1",
            "--poll-seconds",
            "60",
            "--log-level",
            "WARNING",
        ]
    )

    events = [json.loads(line) for line in (runtime_path / "event_log.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    signal_events = [event for event in events if event.get("diff", {}).get("signal_id") == "anomaly-1"]
    refreshed_kg = KnowledgeGraph(json_path=kg_path, auto_load=True, auto_save=True)
    anomaly_nodes = refreshed_kg.query(node_type="anomaly_candidate", metadata_filters={"signal_id": "anomaly-1"})

    assert exit_code == 0
    assert signal_events
    assert signal_events[-1]["diff"]["filtered_out"] is False
    assert signal_events[-1]["diff"]["anomaly_candidate"] is True
    assert signal_events[-1]["diff"]["mode"] == "ANOMALY_CANDIDATE"
    assert anomaly_nodes
    assert anomaly_nodes[0].metadata["reviewed"] is False


def test_trigger_ontology_repair_preserves_runtime_step_and_writes_history(tmp_path, monkeypatch, water_market_schema) -> None:
    runtime_path = Path(tmp_path) / "runtime"
    runtime_path.mkdir(parents=True, exist_ok=True)
    kg_path = Path(tmp_path) / "kg.json"
    config_path = Path(tmp_path) / "config.yaml"
    config = {
        "agent": {
            "bootstrap": {
                "mode": "llm_synthesize",
                "domain_brief": "Base domain brief.",
            },
            "ontology_repair": {
                "gap_threshold": 2,
                "preserve_kg": True,
                "max_repairs_per_session": 3,
            },
        },
        "llm": {"provider": "ollama", "model": "fake-model", "base_url": "http://127.0.0.1:11434", "timeout_seconds": 1.0},
        "memory": {"json_path": str(kg_path)},
        "runtime": {"runtime_path": str(runtime_path), "event_log_path": str(runtime_path / "event_log.jsonl")},
    }
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

    kg = KnowledgeGraph(json_path=kg_path, auto_load=False, auto_save=True)
    forecast_registry = ForecastRegistry(json_path=runtime_path / "forecasts.json", auto_load=False, auto_save=True)
    pipeline = AnalysisPipeline(
        knowledge_graph=kg,
        forecast_registry=forecast_registry,
        sim_config=SimConfig(max_steps=1),
        config_path=config_path,
    )
    current_world = pipeline.compiler.compile(water_market_schema)
    current_world.runtime_step = 17
    pipeline.conscious_state.runtime_metadata["last_runtime_step"] = 17
    gap_trait = KGNode(
        id="sm:identity_trait:ontology_gap:test",
        label="Ontology gap",
        node_type="identity_trait",
        content="Ontology gap around methane leak",
        confidence=0.9,
        metadata={
            "payload": {
                "trait_key": "ontology_gap",
                "cluster_topics": ["methane_leak"],
                "repair_triggered": True,
                "repair_failed": False,
            }
        },
    )
    kg.add_node(gap_trait)

    package_path = runtime_path / "bootstrap_package.json"
    package_path.write_text(
        json.dumps(
            {
                "schema": water_market_schema,
                "policies": [],
                "assumptions": [],
                "bootstrap_mode": "llm_synthesize",
                "domain_brief": "Base domain brief.",
            }
        ),
        encoding="utf-8",
    )

    ctx = stream_runtime.RuntimeContext(
        config=config,
        paths=stream_runtime.RuntimePaths(
            config_path=config_path,
            config_base=config_path.parent,
            runtime_path=runtime_path,
            event_log_path=runtime_path / "event_log.jsonl",
            kg_path=kg_path,
            schema_path=None,
        ),
        pipeline=pipeline,
        current_world=current_world,
        base_world_template=current_world.clone(),
        estimator=object(),
        ingestion_engine=SignalIngestionEngine(),
        llm_client=object(),
        event_log=EventLog(runtime_path / "event_log.jsonl"),
        logged_event_ids=set(),
        cursor_store=StreamCursorStore(),
        signal_memory=stream_runtime.SignalMemory(),
        pending_signals=[],
        queued_signal_ids=set(),
        checkpoint_manager=CheckpointManager(),
        runtime_path=runtime_path,
        keywords=[],
        filter_cfg={},
        args=SimpleNamespace(
            resume=True,
            model="auto",
            ollama_base_url=None,
            bootstrap_mode="llm_synthesize",
            domain_brief_path=None,
            schema_path=None,
            hours=0.0,
            poll_seconds=60.0,
            analysis_interval_seconds=0.1,
            max_signals_per_poll=30,
            log_level="WARNING",
        ),
        package_path=package_path,
        bootstrap_mode="llm_synthesize",
        provider="ollama",
        model_name="fake-model",
        sources=[],
        poll_seconds=60.0,
        analysis_interval_seconds=0.1,
        started_at=stream_runtime._utc_now(),
        deadline=None,
        stats=stream_runtime._initial_runtime_stats(),
    )

    def _fake_bootstrap(**kwargs):  # noqa: ANN003
        repaired_brief = kwargs["domain_brief_override"]
        assert "methane_leak" in repaired_brief
        new_pipeline = AnalysisPipeline(
            knowledge_graph=kwargs["knowledge_graph"],
            forecast_registry=kwargs["forecast_registry"],
            sim_config=SimConfig(max_steps=1),
            config_path=config_path,
        )
        new_world = new_pipeline.compiler.compile(water_market_schema)
        new_world.runtime_step = 0
        kwargs["paths"].runtime_path.mkdir(parents=True, exist_ok=True)
        kwargs["paths"].runtime_path.joinpath("bootstrap_package.json").write_text(
            json.dumps(
                {
                    "schema": water_market_schema,
                    "policies": [],
                    "assumptions": [],
                    "bootstrap_mode": "llm_synthesize",
                    "domain_brief": repaired_brief,
                }
            ),
            encoding="utf-8",
        )
        return stream_runtime.BootstrapResult(
            pipeline=new_pipeline,
            current_world=new_world,
            base_world_template=new_world.clone(),
            llm_client=object(),
            estimator=object(),
            bootstrap_mode="llm_synthesize",
            provider="ollama",
            model_name="fake-model",
            package_path=kwargs["paths"].runtime_path / "bootstrap_package.json",
        )

    monkeypatch.setattr(stream_runtime, "_bootstrap", _fake_bootstrap)

    repaired = stream_runtime._trigger_ontology_repair(
        ctx,
        gap_topics=["methane_leak"],
        repair_trait_ids=["sm:identity_trait:ontology_gap:test"],
    )

    history_lines = (runtime_path / "domain_brief_history.jsonl").read_text(encoding="utf-8").splitlines()

    assert repaired is True
    assert ctx.current_world.runtime_step == 17
    assert ctx.stats["ontology_repairs_triggered"] == 1
    assert history_lines
    assert "methane_leak" in history_lines[-1]


def test_trigger_ontology_repair_marks_traits_failed_on_bootstrap_error(tmp_path, monkeypatch, water_market_schema) -> None:
    runtime_path = Path(tmp_path) / "runtime"
    runtime_path.mkdir(parents=True, exist_ok=True)
    kg_path = Path(tmp_path) / "kg.json"
    config_path = Path(tmp_path) / "config.yaml"
    config = {
        "agent": {
            "bootstrap": {
                "mode": "llm_synthesize",
                "domain_brief": "Base domain brief.",
            },
            "ontology_repair": {
                "gap_threshold": 2,
                "preserve_kg": True,
                "max_repairs_per_session": 3,
            },
        },
        "llm": {"provider": "ollama", "model": "fake-model", "base_url": "http://127.0.0.1:11434", "timeout_seconds": 1.0},
        "memory": {"json_path": str(kg_path)},
        "runtime": {"runtime_path": str(runtime_path), "event_log_path": str(runtime_path / "event_log.jsonl")},
    }
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

    kg = KnowledgeGraph(json_path=kg_path, auto_load=False, auto_save=True)
    forecast_registry = ForecastRegistry(json_path=runtime_path / "forecasts.json", auto_load=False, auto_save=True)
    pipeline = AnalysisPipeline(
        knowledge_graph=kg,
        forecast_registry=forecast_registry,
        sim_config=SimConfig(max_steps=1),
        config_path=config_path,
    )
    current_world = pipeline.compiler.compile(water_market_schema)
    gap_trait = KGNode(
        id="sm:identity_trait:ontology_gap:failed",
        label="Ontology gap",
        node_type="identity_trait",
        content="Ontology gap around methane leak",
        confidence=0.9,
        metadata={
            "payload": {
                "trait_key": "ontology_gap",
                "cluster_topics": ["methane_leak"],
                "repair_triggered": True,
                "repair_failed": False,
            }
        },
    )
    kg.add_node(gap_trait)
    package_path = runtime_path / "bootstrap_package.json"
    package_path.write_text(
        json.dumps(
            {
                "schema": water_market_schema,
                "policies": [],
                "assumptions": [],
                "bootstrap_mode": "llm_synthesize",
                "domain_brief": "Base domain brief.",
            }
        ),
        encoding="utf-8",
    )
    ctx = stream_runtime.RuntimeContext(
        config=config,
        paths=stream_runtime.RuntimePaths(
            config_path=config_path,
            config_base=config_path.parent,
            runtime_path=runtime_path,
            event_log_path=runtime_path / "event_log.jsonl",
            kg_path=kg_path,
            schema_path=None,
        ),
        pipeline=pipeline,
        current_world=current_world,
        base_world_template=current_world.clone(),
        estimator=object(),
        ingestion_engine=SignalIngestionEngine(),
        llm_client=object(),
        event_log=EventLog(runtime_path / "event_log.jsonl"),
        logged_event_ids=set(),
        cursor_store=StreamCursorStore(),
        signal_memory=stream_runtime.SignalMemory(),
        pending_signals=[],
        queued_signal_ids=set(),
        checkpoint_manager=CheckpointManager(),
        runtime_path=runtime_path,
        keywords=[],
        filter_cfg={},
        args=SimpleNamespace(
            resume=True,
            model="auto",
            ollama_base_url=None,
            bootstrap_mode="llm_synthesize",
            domain_brief_path=None,
            schema_path=None,
            hours=0.0,
            poll_seconds=60.0,
            analysis_interval_seconds=0.1,
            max_signals_per_poll=30,
            log_level="WARNING",
        ),
        package_path=package_path,
        bootstrap_mode="llm_synthesize",
        provider="ollama",
        model_name="fake-model",
        sources=[],
        poll_seconds=60.0,
        analysis_interval_seconds=0.1,
        started_at=stream_runtime._utc_now(),
        deadline=None,
        stats=stream_runtime._initial_runtime_stats(),
    )

    def _boom(**kwargs):  # noqa: ANN003
        del kwargs
        raise RuntimeError("bootstrap failed")

    monkeypatch.setattr(stream_runtime, "_bootstrap", _boom)

    repaired = stream_runtime._trigger_ontology_repair(
        ctx,
        gap_topics=["methane_leak"],
        repair_trait_ids=["sm:identity_trait:ontology_gap:failed"],
    )
    refreshed = kg.get_node("sm:identity_trait:ontology_gap:failed", lazy_embed=False)

    assert repaired is False
    assert refreshed is not None
    assert refreshed.metadata["payload"]["repair_failed"] is True


def test_trigger_ontology_repair_schema_path_writes_overlay_and_review_queue(tmp_path, water_market_schema) -> None:
    runtime_path = Path(tmp_path) / "runtime"
    runtime_path.mkdir(parents=True, exist_ok=True)
    schema_path = Path(tmp_path) / "schema.json"
    schema_path.write_text(json.dumps(water_market_schema), encoding="utf-8")
    kg_path = Path(tmp_path) / "kg.json"
    config_path = Path(tmp_path) / "config.yaml"
    config = {
        "agent": {
            "bootstrap": {
                "mode": "schema_path",
                "schema_path": str(schema_path),
            },
            "ontology_repair": {
                "gap_threshold": 2,
                "preserve_kg": True,
                "max_repairs_per_session": 3,
            },
        },
        "memory": {"json_path": str(kg_path)},
        "runtime": {"runtime_path": str(runtime_path), "event_log_path": str(runtime_path / "event_log.jsonl")},
    }
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

    kg = KnowledgeGraph(json_path=kg_path, auto_load=False, auto_save=True)
    kg.add_node(
        KGNode(
            id="gap-evidence",
            label="Gap evidence",
            node_type="claim",
            content="Methane leak shocks agriculture_output and conflict_level through infrastructure stress.",
            confidence=0.9,
            status="active",
        )
    )
    gap_trait = KGNode(
        id="sm:identity_trait:ontology_gap:schema",
        label="Ontology gap",
        node_type="identity_trait",
        content="Ontology gap around methane leak",
        confidence=0.9,
        metadata={
            "payload": {
                "trait_key": "ontology_gap",
                "cluster_topics": ["methane_leak agriculture_output conflict_level"],
                "repair_triggered": True,
                "repair_failed": False,
            }
        },
    )
    kg.add_node(gap_trait)
    forecast_registry = ForecastRegistry(json_path=runtime_path / "forecasts.json", auto_load=False, auto_save=True)
    pipeline = AnalysisPipeline(
        knowledge_graph=kg,
        forecast_registry=forecast_registry,
        sim_config=SimConfig(max_steps=1),
        config_path=config_path,
    )
    current_world = pipeline.compiler.compile(water_market_schema)
    current_world.runtime_step = 17

    ctx = stream_runtime.RuntimeContext(
        config=config,
        paths=stream_runtime.RuntimePaths(
            config_path=config_path,
            config_base=config_path.parent,
            runtime_path=runtime_path,
            event_log_path=runtime_path / "event_log.jsonl",
            kg_path=kg_path,
            schema_path=schema_path,
        ),
        pipeline=pipeline,
        current_world=current_world,
        base_world_template=current_world.clone(),
        estimator=object(),
        ingestion_engine=SignalIngestionEngine(),
        llm_client=object(),
        event_log=EventLog(runtime_path / "event_log.jsonl"),
        logged_event_ids=set(),
        cursor_store=StreamCursorStore(),
        signal_memory=stream_runtime.SignalMemory(),
        pending_signals=[],
        queued_signal_ids=set(),
        checkpoint_manager=CheckpointManager(),
        runtime_path=runtime_path,
        keywords=[],
        filter_cfg={},
        args=SimpleNamespace(
            resume=True,
            model="auto",
            ollama_base_url=None,
            bootstrap_mode="schema_path",
            domain_brief_path=None,
            schema_path=str(schema_path),
            hours=0.0,
            poll_seconds=60.0,
            analysis_interval_seconds=0.1,
            max_signals_per_poll=30,
            log_level="WARNING",
        ),
        package_path=runtime_path / "bootstrap_package.json",
        bootstrap_mode="schema_path",
        provider="none",
        model_name="",
        sources=[],
        poll_seconds=60.0,
        analysis_interval_seconds=0.1,
        started_at=stream_runtime._utc_now(),
        deadline=None,
        stats=stream_runtime._initial_runtime_stats(),
    )

    repaired = stream_runtime._trigger_ontology_repair(
        ctx,
        gap_topics=["methane_leak agriculture_output conflict_level"],
        repair_trait_ids=["sm:identity_trait:ontology_gap:schema"],
    )

    package_payload = json.loads((runtime_path / "bootstrap_package.json").read_text(encoding="utf-8"))
    queue_lines = (runtime_path / "ontology_repair_queue.jsonl").read_text(encoding="utf-8").splitlines()
    refreshed_trait = kg.get_node("sm:identity_trait:ontology_gap:schema", lazy_embed=False)
    proposal_nodes = kg.query(node_type="ontology_patch_proposal")

    assert repaired is True
    assert ctx.current_world.runtime_step == 17
    assert package_payload["schema"]["metadata"]["ontology_topics"] == ["methane_leak agriculture_output conflict_level"]
    assert package_payload["schema"]["metadata"]["ontology_aliases"]["methane_leak agriculture_output conflict_level"][:2] == [
        "agriculture_output",
        "conflict_level",
    ]
    assert queue_lines
    queue_payload = json.loads(queue_lines[-1])
    assert queue_payload["review_required"] is False
    assert queue_payload["relation_candidates"]
    assert queue_payload["applied_relation_candidates"]
    assert queue_payload["appended_causal_edges"]
    assert any(edge["source"] == "agriculture_output" and edge["target"] == "conflict_level" for edge in package_payload["schema"]["causal_dag"])
    assert any(edge.source == "agriculture_output" and edge.target == "conflict_level" for edge in ctx.current_world.causal_dag)
    assert refreshed_trait is not None
    assert refreshed_trait.metadata["payload"]["repair_applied"] is True
    assert refreshed_trait.metadata["payload"]["repair_review_required"] is False
    assert proposal_nodes


def test_process_one_signal_blocks_updates_when_budget_is_exhausted(tmp_path, water_market_schema) -> None:
    runtime_path = Path(tmp_path) / "runtime"
    runtime_path.mkdir(parents=True, exist_ok=True)
    schema_path = Path(tmp_path) / "schema.json"
    schema_path.write_text(json.dumps(water_market_schema), encoding="utf-8")
    kg_path = Path(tmp_path) / "kg.json"
    config_path = Path(tmp_path) / "config.yaml"
    config = {
        "agent": {
            "budget_usd_per_day": 0.001,
            "cost_governance": {"enabled": True, "max_compute_budget_per_session": 0.001},
            "bootstrap": {"mode": "schema_path", "schema_path": str(schema_path)},
        },
        "memory": {"json_path": str(kg_path)},
        "runtime": {"runtime_path": str(runtime_path), "event_log_path": str(runtime_path / "event_log.jsonl")},
        "sim": {"max_steps": 4},
    }
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

    knowledge_graph = KnowledgeGraph(json_path=kg_path, auto_load=False, auto_save=True)
    pipeline = AnalysisPipeline(
        knowledge_graph=knowledge_graph,
        forecast_registry=ForecastRegistry(json_path=runtime_path / "forecasts.json", auto_load=False, auto_save=True),
        sim_config=SimConfig(max_steps=4),
        config_path=config_path,
    )
    current_world = pipeline.compiler.compile(water_market_schema)
    policy = build_budget_policy(config)
    budget_ledger = BudgetLedger(runtime_path / "cost_ledger.jsonl", policy=policy, auto_load=False)

    class _FailEstimator:
        def estimate(self, *args, **kwargs):  # noqa: ANN002, ANN003
            raise AssertionError("budget gate should prevent estimator execution")

    ctx = stream_runtime.RuntimeContext(
        config=config,
        paths=stream_runtime.RuntimePaths(
            config_path=config_path,
            config_base=config_path.parent,
            runtime_path=runtime_path,
            event_log_path=runtime_path / "event_log.jsonl",
            kg_path=kg_path,
            schema_path=schema_path,
        ),
        pipeline=pipeline,
        current_world=current_world,
        base_world_template=current_world.clone(),
        estimator=_FailEstimator(),
        ingestion_engine=SignalIngestionEngine(),
        llm_client=object(),
        event_log=EventLog(runtime_path / "event_log.jsonl"),
        logged_event_ids=set(),
        cursor_store=StreamCursorStore(),
        signal_memory=stream_runtime.SignalMemory(),
        pending_signals=[],
        queued_signal_ids=set(),
        checkpoint_manager=CheckpointManager(),
        runtime_path=runtime_path,
        keywords=[],
        filter_cfg={},
        args=SimpleNamespace(
            resume=True,
            model="auto",
            ollama_base_url=None,
            bootstrap_mode="schema_path",
            domain_brief_path=None,
            schema_path=str(schema_path),
            hours=0.0,
            poll_seconds=60.0,
            analysis_interval_seconds=0.1,
            max_signals_per_poll=30,
            log_level="WARNING",
            include_watch=False,
        ),
        package_path=runtime_path / "bootstrap_package.json",
        bootstrap_mode="schema_path",
        provider="none",
        model_name="",
        sources=[],
        poll_seconds=60.0,
        analysis_interval_seconds=0.1,
        started_at=stream_runtime._utc_now(),
        deadline=None,
        budget_policy=policy,
        cost_model=CostModel(policy),
        budget_ledger=budget_ledger,
        stats=stream_runtime._initial_runtime_stats(),
    )

    signal_payload = Signal(
        signal_id="budgeted-signal",
        source_type="manual",
        text="Drought shock sharply raises water crisis risk.",
        topic="water_risk",
        timestamp="2026-04-12T12:00:00+00:00",
    )
    ctx.ingestion_engine.ingest = lambda *args, **kwargs: [  # type: ignore[method-assign]
        SignalTrigger(
            signal_id="budgeted-signal",
            mahalanobis_score=2.0,
            classification=ShockClassification(shock_type="shock", severity=0.8, semantic_gap=0.1),
            mode="ANALYZE",
            requested_mode="ANALYZE",
        )
    ]

    result = stream_runtime._process_one_signal(signal_payload, ctx=ctx)

    ledger_lines = (runtime_path / "cost_ledger.jsonl").read_text(encoding="utf-8").splitlines()
    runtime_events = [json.loads(line) for line in (runtime_path / "event_log.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    signal_event = next(event for event in runtime_events if event.get("operator") == "runtime_signal_ingest")
    ledger_entry = json.loads(ledger_lines[-1])

    assert result.updated == 0
    assert result.skipped_watch == 1
    assert signal_event["diff"]["mode"] == "WATCH"
    assert signal_event["diff"]["requested_mode"] == "ANALYZE"
    assert signal_event["diff"]["budget_reason"] == "budget_exhaustion_downgrade"
    assert ledger_entry["task_type"] == "signal_processing"
    assert ledger_entry["approved_mode"] == "WATCH"
    assert ledger_entry["actual_cost"] == 0.0


def test_runtime_answer_engine_blocks_generation_when_budget_is_exhausted(tmp_path, water_market_schema) -> None:
    runtime_path = Path(tmp_path) / "runtime"
    runtime_path.mkdir(parents=True, exist_ok=True)
    kg_path = Path(tmp_path) / "kg.json"
    config_path = Path(tmp_path) / "config.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "agent": {
                    "budget_usd_per_day": 0.001,
                    "cost_governance": {"enabled": True, "max_compute_budget_per_session": 0.001},
                },
                "memory": {"json_path": str(kg_path), "embedding_provider": "hashing"},
                "runtime": {"runtime_path": str(runtime_path), "event_log_path": str(runtime_path / "event_log.jsonl")},
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    knowledge_graph = KnowledgeGraph(json_path=kg_path, auto_load=False, auto_save=True)
    knowledge_graph.add_node(
        KGNode(
            id="analysis-1",
            label="Water crisis analysis",
            node_type="analysis_run",
            content="Drought stress raises water crisis risk and lowers cooperation.",
            confidence=0.9,
            status="active",
            metadata={"rationale": "Persistent drought stress reduces basin cooperation."},
        )
    )
    knowledge_graph.save()
    pipeline = AnalysisPipeline(
        knowledge_graph=knowledge_graph,
        forecast_registry=ForecastRegistry(json_path=runtime_path / "forecasts.json", auto_load=False, auto_save=True),
        sim_config=SimConfig(max_steps=2),
        config_path=config_path,
    )
    world = pipeline.compiler.compile(water_market_schema)
    (runtime_path / "world_state.json").write_text(json.dumps(world.snapshot()), encoding="utf-8")

    artifacts = load_runtime_artifacts(config_path)

    class _FailChatClient:
        def chat_text(self, *args, **kwargs):  # noqa: ANN002, ANN003
            raise AssertionError("budget gate should block answer generation before LLM call")

    payload = RuntimeAnswerEngine(artifacts).answer("What is the water crisis outlook?", limit=3, chat_client=_FailChatClient())
    ledger_lines = (runtime_path / "cost_ledger.jsonl").read_text(encoding="utf-8").splitlines()
    ledger_entry = json.loads(ledger_lines[-1])

    assert payload["matched"] is True
    assert payload["answer_generated"] is False
    assert "budget gate blocked answer generation" in str(payload["llm_error"])
    assert payload["budget"]["blocked_count"] >= 1
    assert ledger_entry["task_type"] == "answer_generation"
    assert ledger_entry["approved_mode"] == "WATCH"


def test_cli_status_reports_budget_ledger(tmp_path, capsys) -> None:
    runtime_path = Path(tmp_path) / "runtime"
    runtime_path.mkdir(parents=True, exist_ok=True)
    kg_path = Path(tmp_path) / "kg.json"
    KnowledgeGraph(json_path=kg_path, auto_load=False, auto_save=False).save()
    config_path = Path(tmp_path) / "config.yaml"
    config = {
        "agent": {
            "budget_usd_per_day": 10.0,
            "cost_governance": {"enabled": True, "max_compute_budget_per_session": 10.0},
        },
        "memory": {"json_path": str(kg_path)},
        "runtime": {"runtime_path": str(runtime_path), "event_log_path": str(runtime_path / "event_log.jsonl")},
    }
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    policy = build_budget_policy(config)
    ledger = BudgetLedger(runtime_path / "cost_ledger.jsonl", policy=policy, auto_load=False)
    cost_model = CostModel(policy)
    estimate = cost_model.estimate(task_id="status-check", llm_calls=1, sim_steps=1, actors=1, resources=1, domains=1)
    precheck = cost_model.precheck(requested_mode="ANALYZE", estimate=estimate, budget_spent=0.0)
    ledger.record(task_type="signal_processing", requested_mode="ANALYZE", decision=precheck, actual_cost=3.0)

    exit_code = main(["--config", str(config_path), "status"])
    payload = json.loads(capsys.readouterr().out)

    assert exit_code == 0
    assert payload["budget"]["tracking_enabled"] is True
    assert payload["budget"]["spent_usd"] == 3.0
    assert payload["budget"]["remaining_usd"] == 7.0
    assert payload["budget"]["by_task_type"] == {"signal_processing": 1}


def test_stream_runtime_query_explain_and_forecasts(tmp_path, capsys, water_market_schema) -> None:
    runtime_path = Path(tmp_path) / "runtime"
    runtime_path.mkdir(parents=True, exist_ok=True)
    kg_path = Path(tmp_path) / "kg.json"
    config_path = Path(tmp_path) / "config.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "agent": {"bootstrap": {"mode": "llm_synthesize"}},
                "memory": {"json_path": str(kg_path)},
                "runtime": {"runtime_path": str(runtime_path), "event_log_path": str(runtime_path / "event_log.jsonl")},
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    registry = ForecastRegistry(json_path=runtime_path / "forecasts.json", auto_load=False, auto_save=True)
    knowledge_graph = KnowledgeGraph(json_path=kg_path, auto_load=False, auto_save=True)
    pipeline = AnalysisPipeline(
        sim_config=SimConfig(max_steps=3, convergence_check_steps=50, convergence_epsilon=1.0e-4, seed=17),
        knowledge_graph=knowledge_graph,
        forecast_registry=registry,
        config_path=config_path,
    )
    world = pipeline.compiler.compile(water_market_schema)
    pipeline.update(
        world,
        ParameterVector(
            outcome_modifiers={"water_crisis": 0.8, "cooperation": -0.3},
            rationale="Drought shock worsens crisis risk.",
        ),
        signal_text="Drought shock escalates across the basin.",
        signal_id="signal-query",
        session_log=SessionLog(session_id="query-run"),
    )
    forecast = registry.pending()[0]
    pipeline.verify_forecast(
        forecast.forecast_id,
        actual_prob=float(forecast.predicted_prob),
        verified_at=stream_runtime._utc_now(),
        session_log=SessionLog(session_id="query-verify"),
    )
    CheckpointManager().save(pipeline.conscious_state, runtime_path / "checkpoint.json")
    knowledge_graph.save()
    registry.save()

    exit_code = stream_runtime.main(
        [
            "--config-path",
            str(config_path),
            "--query",
            "explain",
            "--forecast-id",
            str(forecast.forecast_id),
        ]
    )
    explain_output = capsys.readouterr().out

    assert exit_code == 0
    assert str(forecast.forecast_id) in explain_output
    assert "Causal chain:" in explain_output

    exit_code = stream_runtime.main(
        [
            "--config-path",
            str(config_path),
            "--query",
            "forecasts",
        ]
    )
    forecasts_output = json.loads(capsys.readouterr().out)

    assert exit_code == 0
    assert forecasts_output
    assert any(item["forecast_id"] == str(forecast.forecast_id) for item in forecasts_output)
