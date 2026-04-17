"""Read-only explanation rendering for consciousness trace events."""

from __future__ import annotations

import json
from typing import Any

from freeman.agent.consciousness import TraceEvent


class ExplanationRenderer:
    """Render trace events into human-readable explanations."""

    def __init__(self, llm_client: Any | None) -> None:
        self.llm_client = llm_client

    def explain_trace(self, trace_slice: list[TraceEvent]) -> str:
        """Convert a trace slice into a readable explanation."""

        payload = [event.to_dict() for event in trace_slice]
        if self.llm_client is None:
            return json.dumps(payload, indent=2, sort_keys=True)
        messages = [
            {
                "role": "system",
                "content": (
                    "You explain deterministic state transitions from a structured trace. "
                    "Describe what changed, why it changed, and which operator acted. "
                    "Do not speculate beyond the trace."
                ),
            },
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False, sort_keys=True)},
        ]
        try:
            return str(self.llm_client.chat_text(messages, temperature=0.1, max_tokens=700)).strip()
        except Exception:  # noqa: BLE001
            return json.dumps(payload, indent=2, sort_keys=True)

    def explain_node_update(self, event: TraceEvent) -> str:
        """Explain one trace event in plain language."""

        if self.llm_client is None:
            return json.dumps(event.to_dict(), indent=2, sort_keys=True)
        messages = [
            {
                "role": "system",
                "content": "Explain one deterministic state transition in plain technical language.",
            },
            {"role": "user", "content": json.dumps(event.to_dict(), ensure_ascii=False, sort_keys=True)},
        ]
        try:
            return str(self.llm_client.chat_text(messages, temperature=0.1, max_tokens=400)).strip()
        except Exception:  # noqa: BLE001
            return json.dumps(event.to_dict(), indent=2, sort_keys=True)


__all__ = ["ExplanationRenderer"]
