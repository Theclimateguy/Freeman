from __future__ import annotations

import json

from freeman.interface.kgevolution import KnowledgeGraphEvolutionExporter
from freeman.memory.knowledgegraph import KGEdge, KGNode, KnowledgeGraph
from freeman.runtime.kgsnapshot import KGSnapshotManager


def test_snapshot_manager_writes_enriched_snapshot(tmp_path) -> None:
    kg = KnowledgeGraph(json_path=tmp_path / "kg.json", auto_load=False, auto_save=False)
    kg.add_node(KGNode(id="a", label="A", content="alpha", confidence=0.9))
    kg.add_node(KGNode(id="b", label="B", content="beta", confidence=0.8))
    kg.add_edge(KGEdge(source="a", target="b", relation_type="supports"))

    manager = KGSnapshotManager(snapshot_dir=tmp_path / "kg_snapshots", enabled=True)
    snapshot_path = manager.write_snapshot(
        kg,
        runtime_step=3,
        reason="signal_processed",
        domain_id="demo",
        signal_id="signal-123",
        trigger_mode="ANALYZE",
        world_t=2,
        metadata={"world_updated": True},
    )

    assert snapshot_path is not None
    payload = json.loads(snapshot_path.read_text(encoding="utf-8"))
    assert payload["snapshot_meta"]["runtime_step"] == 3
    assert payload["snapshot_meta"]["signal_id"] == "signal-123"
    assert payload["snapshot_meta"]["node_count"] == 2
    assert payload["snapshot_meta"]["edge_count"] == 1
    assert (tmp_path / "kg_snapshots" / "manifest.jsonl").exists()


def test_evolution_exporter_builds_timeline_html_from_snapshot_dir(tmp_path) -> None:
    snapshot_dir = tmp_path / "kg_snapshots"
    snapshot_dir.mkdir()

    first_payload = {
        "backend": "networkx-json",
        "nodes": [
            {"id": "a", "label": "A", "node_type": "concept", "content": "alpha", "confidence": 0.9, "status": "active", "evidence": [], "sources": [], "metadata": {"bucket": "forcing"}, "embedding": [], "created_at": "2026-01-01T00:00:00+00:00", "updated_at": "2026-01-01T00:00:00+00:00", "archived_at": None}
        ],
        "edges": [],
        "snapshot_meta": {"runtime_step": 0, "timestamp": "2026-01-01T00:00:00+00:00", "reason": "bootstrap", "domain_id": "demo"},
    }
    second_payload = {
        "backend": "networkx-json",
        "nodes": [
            {"id": "a", "label": "A", "node_type": "concept", "content": "alpha", "confidence": 0.9, "status": "active", "evidence": [], "sources": [], "metadata": {"bucket": "forcing"}, "embedding": [], "created_at": "2026-01-01T00:00:00+00:00", "updated_at": "2026-01-01T00:00:00+00:00", "archived_at": None},
            {"id": "b", "label": "B", "node_type": "concept", "content": "beta", "confidence": 0.8, "status": "active", "evidence": [], "sources": [], "metadata": {"bucket": "policy"}, "embedding": [], "created_at": "2026-01-01T00:01:00+00:00", "updated_at": "2026-01-01T00:01:00+00:00", "archived_at": None},
        ],
        "edges": [
            {"id": "a:supports:b", "source": "a", "target": "b", "relation_type": "supports", "confidence": 0.8, "weight": 1.0, "metadata": {}, "created_at": "2026-01-01T00:01:00+00:00", "updated_at": "2026-01-01T00:01:00+00:00"}
        ],
        "snapshot_meta": {"runtime_step": 1, "timestamp": "2026-01-01T00:01:00+00:00", "reason": "signal_processed", "domain_id": "demo", "signal_id": "sig-1"},
    }
    (snapshot_dir / "000000__bootstrap.json").write_text(json.dumps(first_payload), encoding="utf-8")
    (snapshot_dir / "000001__signal_processed.json").write_text(json.dumps(second_payload), encoding="utf-8")

    exporter = KnowledgeGraphEvolutionExporter()
    output_path = exporter.export_html(snapshot_dir, tmp_path / "kg_evolution.html")

    html = output_path.read_text(encoding="utf-8")
    assert output_path.exists()
    assert "Freeman Graph Evolution" in html
    assert "signal_processed" in html
    assert "added nodes highlighted" in html
