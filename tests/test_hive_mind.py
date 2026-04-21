"""Tests for multi-agent hive-mind primitives."""

from __future__ import annotations

import math

import pytest

from freeman.agent.attentionscheduler import AttentionScheduler, AttentionTask
from freeman.memory.knowledgegraph import KGEdge, KGNode, KnowledgeGraph
from freeman.memory.reconciler import Reconciler
from freeman.memory.sessionlog import SessionLog


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
