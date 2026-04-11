"""Tests for local runtime checkpointing and cursor semantics."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from freeman.agent.consciousness import ConsciousState
from freeman.interface.cli import main
from freeman.memory.knowledgegraph import KnowledgeGraph
from freeman.memory.selfmodel import SelfModelGraph
from freeman.runtime.agent_runtime import AgentRuntime
from freeman.runtime.checkpoint import CheckpointManager
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
