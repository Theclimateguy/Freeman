"""Tests for KG reconciliation and persistence."""

from __future__ import annotations

from freeman.memory.knowledgegraph import KGNode, KnowledgeGraph
from freeman.memory.reconciler import Reconciler
from freeman.memory.sessionlog import KGDelta, SessionLog


def test_reconciler_archives_low_confidence_nodes(tmp_path) -> None:
    graph_path = tmp_path / "kg.json"
    kg = KnowledgeGraph(json_path=graph_path, auto_load=False, auto_save=True)
    existing = KGNode(
        id="claim_water",
        label="Water Stress",
        content="Water stress is rising.",
        confidence=0.9,
        metadata={"claim_key": "water_stress"},
    )
    kg.add_node(existing)

    session = SessionLog(session_id="s1")
    session.add_kg_delta(
        KGDelta(
            operation="update_node",
            payload={"node": existing.snapshot()},
            support=1,
            contradiction=9,
        )
    )

    result = Reconciler().reconcile(kg, session)
    archived = kg.get_node("claim_water")

    assert archived is not None
    assert archived.status == "archived"
    assert archived.confidence < 0.15
    assert "claim_water" in result.archived_node_ids
    assert graph_path.exists()


def test_reconciler_splits_conflicting_claims_and_exports_graph(tmp_path) -> None:
    kg = KnowledgeGraph(json_path=tmp_path / "kg.json", auto_load=False, auto_save=True)
    base = KGNode(
        id="claim_inflation",
        label="Inflation Outlook",
        content="Inflation will rise.",
        confidence=0.8,
        metadata={"claim_key": "inflation"},
    )
    kg.add_node(base)

    conflicting = KGNode(
        id="claim_inflation_down",
        label="Inflation Outlook",
        content="Inflation will fall.",
        confidence=0.8,
        metadata={"claim_key": "inflation"},
    )
    session = SessionLog(session_id="s2")
    session.add_kg_delta(KGDelta(operation="add_node", payload={"node": conflicting.snapshot()}))

    result = Reconciler().reconcile(kg, session)
    archived = kg.get_node("claim_inflation")
    matches = kg.query(text="inflation")
    dot = kg.export_dot()
    html_path = kg.export_html(tmp_path / "kg.html")
    json_path = kg.export_json(tmp_path / "kg-export.json")

    assert archived is not None
    assert archived.status == "archived"
    assert result.split_nodes["claim_inflation"] == ["claim_inflation__split_1", "claim_inflation_down"]
    assert {"claim_inflation__split_1", "claim_inflation_down"} <= {node.id for node in matches}
    assert "split_into" in dot
    assert html_path.exists()
    assert json_path.exists()
