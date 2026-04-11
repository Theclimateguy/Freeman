"""Append-only event log for Freeman runtime trace events."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from freeman.agent.consciousness import ConsciousState, TraceEvent
from freeman.utils import deep_copy_jsonable


class EventLog:
    """Durable JSONL log of trace events."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path).resolve()

    def append(self, event: TraceEvent) -> None:
        """Append one trace event to the JSONL log."""

        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(event.to_dict(), sort_keys=True))
            handle.write("\n")

    def lookup(self, event_id: str) -> TraceEvent | None:
        """Return one trace event by id via linear scan."""

        for event in self._read_all():
            if event.event_id == str(event_id):
                return event
        return None

    def slice_from(self, after_event_id: str) -> list[TraceEvent]:
        """Return all events after the specified event id."""

        if not self.path.exists():
            return []
        if not after_event_id:
            return self._read_all()
        seen = False
        result: list[TraceEvent] = []
        for event in self._read_all():
            if seen:
                result.append(event)
                continue
            if event.event_id == str(after_event_id):
                seen = True
        return result if seen else self._read_all()

    def replay(self, checkpoint_state: ConsciousState) -> ConsciousState:
        """Replay events from the checkpoint boundary to reconstruct the final state."""

        state = ConsciousState.from_dict(
            checkpoint_state.to_dict(),
            checkpoint_state.self_model_ref.knowledge_graph,
        )
        last_event_id = state.trace_state[-1].event_id if state.trace_state else ""
        for event in self.slice_from(last_event_id):
            self._apply_event(state, event)
        return state

    def _read_all(self) -> list[TraceEvent]:
        if not self.path.exists():
            return []
        events: list[TraceEvent] = []
        for line in self.path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            events.append(TraceEvent.from_dict(json.loads(line)))
        return events

    def _apply_event(self, state: ConsciousState, event: TraceEvent) -> None:
        if any(existing.event_id == event.event_id for existing in state.trace_state):
            return
        diff: dict[str, Any] = deep_copy_jsonable(event.diff)
        signal_id = diff.get("signal_id")
        if signal_id is not None:
            processed = list(state.runtime_metadata.get("processed_signals", []))
            if signal_id not in processed:
                processed.append(str(signal_id))
            state.runtime_metadata["processed_signals"] = processed
        state.trace_state.append(event)


__all__ = ["EventLog"]
