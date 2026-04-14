from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path

from freeman.agent.forecastregistry import Forecast, ForecastRegistry
from freeman.api import tool_api
from freeman.domain.compiler import DomainCompiler
from freeman.memory.knowledgegraph import KGEdge, KGNode, KnowledgeGraph
from freeman.runtime.kgsnapshot import KGSnapshotManager
import yaml


def _write_runtime_fixture(tmp_path: Path, water_market_schema: dict) -> Path:
    runtime_path = tmp_path / "runtime"
    snapshot_dir = runtime_path / "kg_snapshots"
    kg_path = tmp_path / "kg.json"
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "memory": {"json_path": str(kg_path)},
                "runtime": {
                    "runtime_path": str(runtime_path),
                    "event_log_path": str(runtime_path / "event_log.jsonl"),
                    "kg_snapshots": {"enabled": True, "path": str(snapshot_dir)},
                },
                "sim": {"max_steps": 2, "seed": 7},
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    kg = KnowledgeGraph(json_path=kg_path, auto_load=False, auto_save=True)
    kg.add_node(KGNode(id="ghg", label="Greenhouse Gases", node_type="concept", confidence=0.95))
    kg.add_node(KGNode(id="warming", label="Global Warming", node_type="concept", confidence=0.92))
    kg.add_node(
        KGNode(
            id="anomaly_candidate:sig-42",
            label="Unexpected methane leak",
            node_type="anomaly_candidate",
            content="Methane leak outside the seeded ontology.",
            confidence=0.88,
            metadata={"signal_id": "sig-42", "runtime_step": 6, "reviewed": False},
        )
    )
    kg.add_node(
        KGNode(
            id="sm:identity_trait:ontology_gap:methane",
            label="Ontology gap",
            node_type="identity_trait",
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
    )
    kg.add_edge(
        KGEdge(
            source="ghg",
            target="warming",
            relation_type="causes",
            confidence=0.55,
            metadata={"runtime_step": 4, "world_step": 1},
        )
    )
    edge = next(item for item in kg.edges() if item.source == "ghg" and item.target == "warming")

    snapshot_manager = KGSnapshotManager(snapshot_dir=snapshot_dir, enabled=True)
    snapshot_manager.write_snapshot(
        kg,
        runtime_step=4,
        reason="seed_loaded",
        domain_id="water_market",
        signal_id="seed",
        world_t=1,
    )

    kg.add_edge(
        KGEdge(
            source="ghg",
            target="warming",
            relation_type="causes",
            confidence=0.85,
            id=edge.id,
            metadata={"runtime_step": 6, "world_step": 2},
        )
    )
    kg.save()
    snapshot_manager.write_snapshot(
        kg,
        runtime_step=6,
        reason="signal_processed",
        domain_id="water_market",
        signal_id="sig-42",
        world_t=2,
    )

    runtime_path.mkdir(parents=True, exist_ok=True)
    registry = ForecastRegistry(json_path=runtime_path / "forecasts.json", auto_load=False, auto_save=True)
    registry.record(
        Forecast(
            forecast_id="water_market:6:warming",
            domain_id="water_market",
            outcome_id="warming",
            predicted_prob=0.8,
            session_id="session:test",
            horizon_steps=3,
            created_at=datetime(2026, 4, 14, tzinfo=timezone.utc),
            created_step=2,
            created_runtime_step=6,
            status="pending",
            causal_path=[edge.id],
        )
    )

    world = DomainCompiler().compile(water_market_schema)
    world.runtime_step = 6
    world.t = 2
    (runtime_path / "world_state.json").write_text(
        json.dumps(world.snapshot(), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    (runtime_path / "pending_signals.json").write_text(
        json.dumps({"signals": [{"signal_id": "queued-1"}]}, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    (runtime_path / "event_log.jsonl").write_text(
        json.dumps({"event_id": "trace:signal:seed"}) + "\n" + json.dumps({"event_id": "trace:signal:sig-42"}) + "\n",
        encoding="utf-8",
    )
    return config_path


def test_runtime_query_tools_read_persisted_runtime(tmp_path, water_market_schema) -> None:
    config_path = _write_runtime_fixture(tmp_path, water_market_schema)

    summary = tool_api.freeman_get_runtime_summary(str(config_path))
    forecasts = tool_api.freeman_query_forecasts(str(config_path), status="pending")
    explanation = tool_api.freeman_explain_forecast(str(config_path), forecast_id="water_market:6:warming")
    anomalies = tool_api.freeman_query_anomalies(str(config_path))
    causal = tool_api.freeman_query_causal_edges(str(config_path), source="Greenhouse", target="warming")
    relation = tool_api.freeman_trace_relation_learning(
        str(config_path),
        source="ghg",
        target="warming",
        relation_type="causes",
        last_n_steps=3,
    )

    assert summary["runtime_step"] == 6
    assert summary["snapshot_count"] == 2
    assert summary["forecast_counts"] == {"pending": 1}
    assert summary["pending_signal_count"] == 1
    assert forecasts["count"] == 1
    assert forecasts["items"][0]["forecast_id"] == "water_market:6:warming"
    assert explanation["status"] == "pending"
    assert explanation["causal_chain"][0]["edge_id"]
    assert "pending ex-post verification" in explanation["text"]
    assert len(anomalies["anomaly_candidates"]) == 1
    assert anomalies["ontology_gap_traits"][0]["cluster_topics"] == ["methane_leak"]
    assert causal["count"] == 1
    assert causal["items"][0]["source_label"] == "Greenhouse Gases"
    assert len(relation["timeline"]) == 2
    assert relation["current_matches"][0]["confidence"] == 0.85


def test_invoke_tool_dispatches_runtime_query(tmp_path, water_market_schema) -> None:
    config_path = _write_runtime_fixture(tmp_path, water_market_schema)

    payload = tool_api.invoke_tool(
        "freeman_query_forecasts",
        {"config_path": str(config_path), "status": "pending", "limit": 5},
    )

    assert payload["count"] == 1
    assert payload["items"][0]["status"] == "pending"
