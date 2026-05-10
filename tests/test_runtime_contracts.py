"""Runtime interface contracts across world, memory, and conscious state layers."""

from __future__ import annotations

from freeman.agent.consciousness import ConsciousState
from freeman.memory.knowledgegraph import KnowledgeGraph
from freeman.memory.selfmodel import SelfModelGraph
from freeman.runtime.contracts import ConsciousStateContract, KnowledgeGraphContract, WorldStateContract


def test_world_state_satisfies_runtime_contract(water_market_world) -> None:
    assert isinstance(water_market_world, WorldStateContract)
    assert water_market_world.clone().snapshot()["domain_id"] == water_market_world.domain_id


def test_knowledge_graph_satisfies_runtime_contract(tmp_path) -> None:
    graph = KnowledgeGraph(json_path=tmp_path / "kg.json", auto_load=False, auto_save=False)

    assert isinstance(graph, KnowledgeGraphContract)
    assert graph.semantic_search("water", top_k=1).trace() == []


def test_conscious_state_satisfies_runtime_contract(tmp_path) -> None:
    graph = KnowledgeGraph(json_path=tmp_path / "kg.json", auto_load=False, auto_save=False)
    state = ConsciousState(world_ref="world:test:0", self_model_ref=SelfModelGraph(graph), agent_role="planner")

    assert isinstance(state, ConsciousStateContract)
    assert state.to_dict()["agent_role"] == "planner"
