"""Stage 4 tests for read-only consciousness narration."""

from __future__ import annotations

import json
from pathlib import Path

import yaml

from freeman.agent.consciousness import ConsciousState, TraceEvent
from freeman.interface.cli import main
from freeman.interface.identity import build_identity_state
from freeman.llm.explanation_renderer import ExplanationRenderer
from freeman.llm.identity_narrator import IdentityNarrator
from freeman.memory.knowledgegraph import KnowledgeGraph
from freeman.memory.selfmodel import SelfModelGraph, SelfModelNode

from tests.test_consciousness_stage1 import _now


class DummyLLM:
    def chat_text(self, messages, *, temperature=0.2, max_tokens=None):  # noqa: ANN001
        del temperature, max_tokens
        return f"dummy:{len(messages)}"


def _state(tmp_path) -> ConsciousState:
    kg = KnowledgeGraph(json_path=tmp_path / "kg.json", auto_load=False, auto_save=True)
    graph = SelfModelGraph(kg)
    graph.write(
        SelfModelNode(
            node_id="sm:self_capability:water",
            node_type="self_capability",
            domain="water",
            payload={"capability": 0.7, "mean_abs_error": 0.2},
            confidence=0.7,
            created_at=_now(),
            updated_at=_now(),
            source_trace_id="trace:1",
        ),
        caller_token="freeman_consciousness_engine",
    )
    state = ConsciousState(
        world_ref="world:water:1",
        self_model_ref=graph,
        goal_state=[],
        attention_state={"water": 1.0},
        trace_state=[
            TraceEvent(
                event_id="trace:1",
                timestamp=_now(),
                transition_type="external",
                trigger_type="manual",
                operator="capability_review",
                pre_state_ref="state:0",
                post_state_ref="state:1",
                input_refs=["sm:self_capability:water"],
                diff={"nodes": [graph.get_nodes_by_type("self_capability")[0].to_dict()]},
                rationale="updated capability",
            )
        ],
        runtime_metadata={"world_ref": "world:water:1"},
    )
    kg.save()
    return state


def test_narrator_structured_snapshot_no_llm(tmp_path) -> None:
    state = _state(tmp_path)
    narrator = IdentityNarrator(None)

    payload = narrator.structured_snapshot(state)

    assert payload
    assert payload["node_counts"]["self_capability"] == 1


def test_narrator_does_not_mutate_state(tmp_path) -> None:
    state = _state(tmp_path)
    narrator = IdentityNarrator(DummyLLM())
    before = state.to_dict()

    narrator.render(state)

    assert state.to_dict() == before


def test_renderer_does_not_mutate_state(tmp_path) -> None:
    state = _state(tmp_path)
    renderer = ExplanationRenderer(DummyLLM())
    before = state.to_dict()

    renderer.explain_trace(state.trace_state)

    assert state.to_dict() == before


def test_cli_identity_returns_json(tmp_path, capsys) -> None:
    state = _state(tmp_path)
    config_path = Path(tmp_path) / "config.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "memory": {
                    "json_path": str(state.self_model_ref.knowledge_graph.json_path),
                    "session_log_path": str(Path(tmp_path) / "sessions"),
                }
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    exit_code = main(["--config", str(config_path), "identity"])
    captured = capsys.readouterr()
    payload = json.loads(captured.out)

    assert exit_code == 0
    assert "nodes_by_type" in payload
    assert "self_capability" in payload["nodes_by_type"]
