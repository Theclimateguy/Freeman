from __future__ import annotations

from datetime import datetime, timezone
import io
import json
import logging
import os
from types import SimpleNamespace

import pytest
import yaml

from freeman.agent.costmodel import BudgetLedger, CostModel, build_budget_policy
from freeman.agent.signalingestion import Signal
from freeman.logging_config import JsonFormatter, set_log_context
from freeman.llm.circuit_breaker import CircuitBreaker, CircuitOpenError
from freeman.memory.knowledgegraph import KGNode, KnowledgeGraph
from freeman.runtime import stream_runtime
from freeman.runtime.signal_loop import _run_poll
from freeman.runtime.health import health_from_config
from freeman.runtime.startup_checks import validate_config


def test_knowledge_graph_transaction_rolls_back_memory_and_disk(tmp_path) -> None:
    kg_path = tmp_path / "kg.json"
    kg = KnowledgeGraph(json_path=kg_path, auto_load=False, auto_save=True)
    kg.add_node(KGNode(id="baseline", label="Baseline", confidence=0.9))
    before = json.loads(kg_path.read_text(encoding="utf-8"))

    with pytest.raises(RuntimeError):
        with kg.transaction():
            kg.add_node(KGNode(id="partial", label="Partial", confidence=0.9))
            kg.save()
            raise RuntimeError("interrupt repair")

    assert kg.get_node("partial", lazy_embed=False) is None
    assert json.loads(kg_path.read_text(encoding="utf-8")) == before


def test_budget_ledger_fsyncs_append(tmp_path, monkeypatch) -> None:
    calls: list[int] = []
    monkeypatch.setattr(os, "fsync", lambda fd: calls.append(int(fd)))
    policy = build_budget_policy({"agent": {"cost_governance": {"enabled": True, "max_compute_budget_per_session": 1.0}}})
    model = CostModel(policy)
    estimate = model.estimate(task_id="task-1", llm_calls=1, sim_steps=1, actors=1, resources=1)
    decision = model.precheck(requested_mode="ANALYZE", estimate=estimate, budget_spent=0.0)

    ledger = BudgetLedger(tmp_path / "cost_ledger.jsonl", policy=policy, auto_load=False)
    ledger.record(task_type="signal_processing", requested_mode="ANALYZE", decision=decision, actual_cost=0.01)

    assert calls
    assert (tmp_path / "cost_ledger.jsonl").read_text(encoding="utf-8").strip()


def test_startup_validation_requires_remote_llm_key(monkeypatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("LLM_API_KEY", raising=False)

    base_config = {
        "agent": {"budget_usd_per_day": 1.0},
        "runtime": {"runtime_path": "."},
        "memory": {"json_path": "./kg.json"},
    }
    errors = validate_config({**base_config, "llm": {"provider": "openai"}})
    assert "LLM provider 'openai' requires OPENAI_API_KEY or LLM_API_KEY." in errors

    assert validate_config({**base_config, "llm": {"provider": "openai", "api_key": "test-key"}}) == []


def test_startup_validation_checks_budget_and_writable_paths(tmp_path) -> None:
    errors = validate_config(
        {
            "agent": {"budget_usd_per_day": 0.0, "cost_governance": {"max_compute_budget_per_session": 0.0}},
            "runtime": {"runtime_path": str(tmp_path / "runtime")},
            "memory": {"json_path": str(tmp_path / "kg.json")},
            "llm": {"provider": "none"},
        }
    )

    assert errors == ["agent.budget_usd_per_day and cost_governance.max_compute_budget_per_session must be > 0."]


def test_startup_validation_uses_default_budget_for_partial_config(tmp_path) -> None:
    errors = validate_config(
        {
            "runtime": {"runtime_path": str(tmp_path / "runtime")},
            "memory": {"json_path": str(tmp_path / "kg.json")},
            "llm": {"provider": "none"},
        }
    )

    assert errors == []


def test_stream_runtime_uses_freeman_config_env_default(tmp_path, monkeypatch) -> None:
    config_path = tmp_path / "container-config.yaml"
    captured: dict[str, str] = {}

    class Parser:
        def parse_args(self, _argv):
            raise RuntimeError("stop after parser construction")

    def fake_build_parser(*, default_config_path: str):
        captured["default_config_path"] = default_config_path
        return Parser()

    monkeypatch.setenv("FREEMAN_CONFIG", str(config_path))
    monkeypatch.setattr(stream_runtime, "_build_parser", fake_build_parser)

    with pytest.raises(RuntimeError, match="stop after parser construction"):
        stream_runtime.main([])

    assert captured["default_config_path"] == str(config_path)


def test_health_from_config_reports_ok_and_degraded_states(tmp_path) -> None:
    runtime_path = tmp_path / "runtime"
    runtime_path.mkdir()
    kg_path = tmp_path / "kg.json"
    KnowledgeGraph(json_path=kg_path, auto_load=False, auto_save=False).save()
    (runtime_path / "world_state.json").write_text(
        json.dumps(
            {
                "domain_id": "health",
                "t": 3,
                "runtime_step": 5,
                "actors": {},
                "resources": {},
                "outcomes": {},
                "causal_dag": [],
                "metadata": {},
            }
        ),
        encoding="utf-8",
    )
    (runtime_path / "event_log.jsonl").write_text(
        json.dumps(
            {
                "event_id": "trace:signal:s1",
                "trigger_type": "signal",
                "timestamp": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
                "diff": {"signal_id": "s1"},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "memory": {"json_path": str(kg_path)},
                "runtime": {
                    "runtime_path": str(runtime_path),
                    "event_log_path": str(runtime_path / "event_log.jsonl"),
                    "health_signal_stale_seconds": 3600,
                },
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    assert health_from_config(config_path).status == "ok"
    (runtime_path / "event_log.jsonl").write_text("", encoding="utf-8")
    degraded = health_from_config(config_path)
    assert degraded.status == "degraded"
    assert "no_signal_events" in degraded.reasons


def test_circuit_breaker_closed_open_half_open() -> None:
    breaker = CircuitBreaker(failure_threshold=2, reset_timeout=0.0)
    attempts = {"count": 0}

    def _flaky() -> str:
        attempts["count"] += 1
        if attempts["count"] <= 2:
            raise RuntimeError("provider down")
        return "ok"

    with pytest.raises(RuntimeError):
        breaker.call(_flaky)
    with pytest.raises(RuntimeError):
        breaker.call(_flaky)
    assert breaker.state == "OPEN"

    assert breaker.call(_flaky) == "ok"
    assert breaker.state == "CLOSED"

    blocker = CircuitBreaker(failure_threshold=1, reset_timeout=60.0)
    with pytest.raises(RuntimeError):
        blocker.call(lambda: (_ for _ in ()).throw(RuntimeError("down")))
    with pytest.raises(CircuitOpenError):
        blocker.call(lambda: "blocked")


def test_json_formatter_includes_run_id() -> None:
    stream = io.StringIO()
    handler = logging.StreamHandler(stream)
    handler.setFormatter(JsonFormatter())
    logger = logging.getLogger("freeman.test.json")
    logger.handlers = [handler]
    logger.propagate = False
    logger.setLevel(logging.INFO)
    set_log_context(run_id="run-test", correlation_id="corr-test")

    logger.info("structured event")

    payload = json.loads(stream.getvalue())
    assert payload["msg"] == "structured event"
    assert payload["run_id"] == "run-test"
    assert payload["correlation_id"] == "corr-test"


def test_stream_poll_backpressure_leaves_full_pending_queue_unchanged() -> None:
    class _Source:
        def fetch(self):
            return [
                Signal(
                    signal_id="new-signal",
                    source_type="manual",
                    text="Climate emissions adaptation policy update.",
                    topic="climate_news",
                )
            ]

    ctx = SimpleNamespace(
        keywords=[],
        filter_cfg={},
        config={"runtime": {"pending_queue_max_size": 1}},
        stats={"filtered_out_count": 0, "signals_seen": 0, "queue_backpressure_skipped": 0},
        pending_signals=[
            Signal(signal_id="queued", source_type="manual", text="queued", topic="queued"),
        ],
        args=SimpleNamespace(max_signals_per_poll=30),
        cursor_store=SimpleNamespace(is_committed=lambda signal_id: False),
        queued_signal_ids={"queued"},
        current_world=None,
    )

    assert _run_poll([_Source()], ctx) == 0
    assert [signal.signal_id for signal in ctx.pending_signals] == ["queued"]
    assert ctx.stats["queue_backpressure_skipped"] == 1
