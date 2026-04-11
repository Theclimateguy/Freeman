"""Identity-state helpers for CLI and interface layers."""

from __future__ import annotations

from typing import Any

from freeman.agent.consciousness import ConsciousState, TraceEvent
from freeman.memory.knowledgegraph import KnowledgeGraph
from freeman.memory.selfmodel import SelfModelGraph


def build_identity_state(
    knowledge_graph: KnowledgeGraph,
    *,
    trace_state: list[TraceEvent] | None = None,
    runtime_metadata: dict[str, Any] | None = None,
) -> ConsciousState:
    """Build a read-only consciousness snapshot from persisted state."""

    self_model = SelfModelGraph(knowledge_graph)
    attention_nodes = self_model.get_nodes_by_type("attention_focus")
    goal_nodes = self_model.get_nodes_by_type("goal_state")
    attention_state = {
        str(node.domain or node.payload.get("domain", node.node_id)): float(node.payload.get("weight", node.confidence))
        for node in attention_nodes
    }
    return ConsciousState(
        world_ref=str((runtime_metadata or {}).get("world_ref", "world:unknown")),
        self_model_ref=self_model,
        goal_state=[node.node_id for node in goal_nodes],
        attention_state=attention_state,
        trace_state=list(trace_state or []),
        runtime_metadata=dict(runtime_metadata or {}),
    )


__all__ = ["build_identity_state"]
