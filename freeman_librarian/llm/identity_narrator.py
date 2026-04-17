"""Read-only narration over the Freeman consciousness state."""

from __future__ import annotations

import json
from typing import Any

from freeman_librarian.agent.consciousness import ConsciousState


class IdentityNarrator:
    """Project structured consciousness state into machine and prose views."""

    def __init__(self, llm_client: Any | None) -> None:
        self.llm_client = llm_client

    def structured_snapshot(self, state: ConsciousState) -> dict[str, Any]:
        """Return the primary machine-readable summary without calling an LLM."""

        snapshot = state.to_dict()
        self_model = snapshot.get("self_model", {})
        nodes = self_model.get("nodes", [])
        by_type: dict[str, list[dict[str, Any]]] = {}
        for node in nodes:
            by_type.setdefault(str(node.get("node_type", "unknown")), []).append(node)
        return {
            "world_ref": snapshot["world_ref"],
            "goal_state": snapshot["goal_state"],
            "attention_state": snapshot["attention_state"],
            "trace_count": len(snapshot["trace_state"]),
            "runtime_metadata": snapshot["runtime_metadata"],
            "node_counts": {node_type: len(items) for node_type, items in sorted(by_type.items())},
            "nodes_by_type": {node_type: items for node_type, items in sorted(by_type.items())},
            "edge_count": len(self_model.get("edges", [])),
        }

    def render(self, state: ConsciousState) -> str:
        """Return a human-readable identity narrative without mutating state."""

        structured = self.structured_snapshot(state)
        if self.llm_client is None:
            return json.dumps(structured, indent=2, sort_keys=True)
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a read-only narrator for a deterministic consciousness state. "
                    "Summarize what the agent currently knows, where it is uncertain, and what it is focused on. "
                    "Do not invent state not present in the payload."
                ),
            },
            {"role": "user", "content": json.dumps(structured, ensure_ascii=False, sort_keys=True)},
        ]
        return str(self.llm_client.chat_text(messages, temperature=0.1, max_tokens=600)).strip()


__all__ = ["IdentityNarrator"]
