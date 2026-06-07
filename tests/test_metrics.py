from __future__ import annotations

from datetime import datetime, timezone
import json

import yaml

from freeman.agent.costmodel import CostModel, BudgetLedger, build_budget_policy
from freeman.agent.forecastregistry import Forecast, ForecastRegistry
from freeman.interface.cli import main
from freeman.memory.knowledgegraph import KGEdge, KGNode, KnowledgeGraph
from freeman.runtime.metrics import collect_metrics, metrics_from_config


def _write_config(tmp_path, *, memory_backend: str = "json") -> tuple[dict, object]:
    runtime_path = tmp_path / "runtime"
    kg_json_path = tmp_path / "kg.json"
    kg_sqlite_path = tmp_path / "kg.db"
    config = {
        "agent": {
            "budget_usd_per_day": 10.0,
            "cost_governance": {"enabled": True, "max_compute_budget_per_session": 10.0},
        },
        "llm": {"provider": "ollama"},
        "memory": {
            "backend": memory_backend,
            "json_path": str(kg_json_path),
            "sqlite_path": str(kg_sqlite_path),
        },
        "runtime": {
            "runtime_path": str(runtime_path),
            "event_log_path": str(runtime_path / "event_log.jsonl"),
        },
    }
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return config, config_path


def test_metrics_from_config_exports_prometheus_runtime_metrics(tmp_path) -> None:
    config, config_path = _write_config(tmp_path)
    runtime_path = tmp_path / "runtime"
    runtime_path.mkdir()
    (runtime_path / "world_state.json").write_text(
        json.dumps(
            {
                "domain_id": "metrics",
                "t": 4,
                "runtime_step": 6,
                "actors": {},
                "resources": {},
                "outcomes": {},
                "causal_dag": [],
                "metadata": {},
            }
        ),
        encoding="utf-8",
    )
    KnowledgeGraph(json_path=tmp_path / "kg.json", auto_load=False, auto_save=True).add_node(
        KGNode(id="n1", label="Node 1", confidence=0.9)
    )
    registry = ForecastRegistry(json_path=runtime_path / "forecasts.json", auto_load=False, auto_save=True)
    registry.record(
        Forecast(
            forecast_id="f1",
            domain_id="metrics",
            outcome_id="risk",
            predicted_prob=0.4,
            session_id="s1",
            horizon_steps=2,
            created_at=datetime.now(timezone.utc),
            status="pending",
        )
    )
    policy = build_budget_policy(config)
    estimate = CostModel(policy).estimate(task_id="task-1", llm_calls=1, sim_steps=1, actors=1, resources=1)
    decision = CostModel(policy).precheck(requested_mode="ANALYZE", estimate=estimate, budget_spent=0.0)
    BudgetLedger(runtime_path / "cost_ledger.jsonl", policy=policy, auto_load=False).record(
        task_type="signal_processing",
        requested_mode="ANALYZE",
        decision=decision,
        actual_cost=3.0,
    )
    events = [
        {
            "event_id": "trace:signal:s1",
            "operator": "runtime_signal_ingest",
            "trigger_type": "signal",
            "timestamp": "2026-06-07T00:00:00+00:00",
            "diff": {
                "signal_id": "s1",
                "filtered_out": False,
                "llm_used": True,
                "signal_processing_seconds": 0.2,
                "analysis_pipeline_seconds": 0.4,
                "llm_call_seconds": 0.15,
            },
        },
        {
            "event_id": "trace:signal:s2",
            "operator": "runtime_signal_ingest",
            "trigger_type": "signal",
            "timestamp": "2026-06-07T00:00:01+00:00",
            "diff": {"signal_id": "s2", "filtered_out": True, "llm_used": False},
        },
        {
            "event_id": "repair:1",
            "operator": "ontology_repair_request",
            "trigger_type": "internal",
            "timestamp": "2026-06-07T00:00:02+00:00",
            "diff": {},
        },
    ]
    (runtime_path / "event_log.jsonl").write_text(
        "\n".join(json.dumps(event, sort_keys=True) for event in events) + "\n",
        encoding="utf-8",
    )

    output = metrics_from_config(config_path)

    assert "# TYPE signals_ingested_total counter" in output
    assert "signals_ingested_total 2" in output
    assert "signals_filtered_total 1" in output
    assert "ontology_repairs_total 1" in output
    assert 'llm_calls_total{provider="ollama",task_type="signal_processing"} 1' in output
    assert "world_t 4" in output
    assert "kg_node_count 1" in output
    assert "budget_spent_usd 3" in output
    assert "budget_remaining_usd 7" in output
    assert "active_forecasts 1" in output
    assert "# TYPE signal_processing_seconds histogram" in output
    assert 'signal_processing_seconds_bucket{le="0.25"} 1' in output
    assert "signal_processing_seconds_sum 0.2" in output
    assert "signal_processing_seconds_count 1" in output
    assert 'llm_call_seconds_bucket{le="0.25",provider="ollama"} 1' in output


def test_metrics_counts_sqlite_kg_without_json_export(tmp_path) -> None:
    _config, config_path = _write_config(tmp_path, memory_backend="sqlite")
    graph = KnowledgeGraph(
        json_path=tmp_path / "kg.json",
        sqlite_path=tmp_path / "kg.db",
        backend_name="sqlite",
        auto_load=False,
        auto_save=True,
    )
    graph.add_node(KGNode(id="a", label="A", confidence=0.9))
    graph.add_node(KGNode(id="b", label="B", confidence=0.8))
    graph.add_edge(KGEdge(source="a", target="b", relation_type="causes"))

    samples = collect_metrics(config_path)
    by_name = {sample.name: sample.value for sample in samples if not sample.labels}

    assert by_name["kg_node_count"] == 2
    assert not (tmp_path / "kg.json").exists()


def test_metrics_cli_is_read_only_for_missing_runtime(tmp_path, capsys) -> None:
    _config, config_path = _write_config(tmp_path)

    exit_code = main(["--config", str(config_path), "metrics"])
    output = capsys.readouterr().out

    assert exit_code == 0
    assert "signals_ingested_total 0" in output
    assert "world_t -1" in output
    assert not (tmp_path / "runtime").exists()
    assert not (tmp_path / "kg.json").exists()
