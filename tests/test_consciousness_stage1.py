"""Stage 1 tests for the Freeman consciousness layer."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from freeman.agent.consciousness import ConsciousState, ENGINE_TOKEN, TraceEvent
from freeman.memory.knowledgegraph import KnowledgeGraph
from freeman.memory.selfmodel import (
    SelfModelAccessError,
    SelfModelEdge,
    SelfModelGraph,
    SelfModelNode,
)


def _now() -> datetime:
    return datetime(2026, 4, 12, 12, 0, tzinfo=timezone.utc)


def _node() -> SelfModelNode:
    now = _now()
    return SelfModelNode(
        node_id="sm:self_capability:water",
        node_type="self_capability",
        domain="water",
        payload={"mae": 0.12, "capability": 0.71},
        confidence=0.71,
        created_at=now,
        updated_at=now,
        source_trace_id="trace:capability",
    )


def _edge() -> SelfModelEdge:
    now = _now()
    return SelfModelEdge(
        edge_id="sm-edge:1",
        edge_type="supports",
        source_id="sm:self_capability:water",
        target_id="sm:goal_state:water-risk",
        weight=0.8,
        created_at=now,
        trace_id="trace:edge",
    )


def _trace() -> TraceEvent:
    return TraceEvent(
        event_id="trace:1234",
        timestamp=_now(),
        transition_type="external",
        trigger_type="signal",
        operator="capability_review",
        pre_state_ref="state:0",
        post_state_ref="state:1",
        input_refs=["signal:water-1"],
        diff={"updated_nodes": ["sm:self_capability:water"]},
        rationale="updated from rolling MAE",
    )


def test_selfmodel_node_round_trip() -> None:
    node = _node()
    restored = SelfModelNode.from_dict(node.to_dict())

    assert restored == node


def test_selfmodel_edge_round_trip() -> None:
    edge = _edge()
    restored = SelfModelEdge.from_dict(edge.to_dict())

    assert restored == edge


def test_write_guard_raises(tmp_path) -> None:
    kg = KnowledgeGraph(json_path=tmp_path / "kg.json", auto_load=False, auto_save=False)
    graph = SelfModelGraph(kg)

    with pytest.raises(SelfModelAccessError):
        graph.write(_node(), caller_token="not-the-engine")


def test_write_guard_passes(tmp_path) -> None:
    kg = KnowledgeGraph(json_path=tmp_path / "kg.json", auto_load=False, auto_save=False)
    graph = SelfModelGraph(kg)
    node = _node()

    graph.write(node, caller_token=ENGINE_TOKEN)

    stored = graph.get_nodes_by_type("self_capability")
    assert stored == [node]


def test_conscious_state_round_trip(tmp_path) -> None:
    kg = KnowledgeGraph(json_path=tmp_path / "kg.json", auto_load=False, auto_save=False)
    graph = SelfModelGraph(kg)
    graph.write(_node(), caller_token=ENGINE_TOKEN)
    graph.write(
        SelfModelNode(
            node_id="sm:goal_state:water-risk",
            node_type="goal_state",
            domain="water",
            payload={"urgency": 0.9},
            confidence=0.9,
            created_at=_now(),
            updated_at=_now(),
            source_trace_id="trace:goal",
        ),
        caller_token=ENGINE_TOKEN,
    )
    graph.write(_edge(), caller_token=ENGINE_TOKEN)
    state = ConsciousState(
        world_ref="world:water:1",
        self_model_ref=graph,
        goal_state=["sm:goal_state:water-risk"],
        attention_state={"water": 0.7, "macro": 0.3},
        trace_state=[_trace()],
        runtime_metadata={"schema_version": 1},
    )

    restored = ConsciousState.from_dict(
        state.to_dict(),
        KnowledgeGraph(json_path=tmp_path / "kg-restored.json", auto_load=False, auto_save=False),
    )

    assert restored == state


def test_trace_event_round_trip() -> None:
    event = _trace()
    restored = TraceEvent.from_dict(event.to_dict())

    assert restored == event
