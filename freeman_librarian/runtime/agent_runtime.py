"""Foreground runtime loop for local Freeman agent execution."""

from __future__ import annotations

from datetime import datetime, timezone
import signal
import time
from pathlib import Path
from typing import Any, Callable, Iterable

from freeman_librarian.agent.consciousness import ConsciousState, TraceEvent
from freeman_librarian.runtime.checkpoint import CheckpointManager
from freeman_librarian.runtime.event_log import EventLog
from freeman_librarian.runtime.stream import StreamCursorStore


def _now() -> datetime:
    return datetime.now(timezone.utc).replace(microsecond=0)


class AgentRuntime:
    """Minimal foreground runtime with checkpointing and cursor dedup."""

    def __init__(
        self,
        *,
        state: ConsciousState,
        signals: Iterable[dict[str, Any]] = (),
        cursor_store: StreamCursorStore | None = None,
        checkpoint_manager: CheckpointManager | None = None,
        event_log: EventLog | None = None,
        runtime_path: Path | None = None,
        process_signal: Callable[[ConsciousState, dict[str, Any]], None] | None = None,
    ) -> None:
        self.state = state
        self.signals = list(signals)
        self.cursor_store = cursor_store or StreamCursorStore()
        self.checkpoint_manager = checkpoint_manager or CheckpointManager()
        self.runtime_path = Path(runtime_path or ".").resolve()
        self._event_log = event_log or EventLog(self.runtime_path / "event_log.jsonl")
        self.process_signal = process_signal or self._default_process_signal
        self._stop_requested = False

    def run_oneshot(self) -> None:
        for signal_payload in self.signals:
            if self._stop_requested:
                break
            self._process_one(signal_payload)

    def run_follow(self) -> None:
        previous_handlers = {
            signal.SIGINT: signal.getsignal(signal.SIGINT),
            signal.SIGTERM: signal.getsignal(signal.SIGTERM),
        }

        def _request_stop(signum, frame):  # noqa: ANN001
            del signum, frame
            self._stop_requested = True

        signal.signal(signal.SIGINT, _request_stop)
        signal.signal(signal.SIGTERM, _request_stop)
        try:
            while not self._stop_requested:
                if not self.signals:
                    time.sleep(0.01)
                    continue
                self.run_oneshot()
                self._stop_requested = True
        finally:
            self._graceful_shutdown()
            signal.signal(signal.SIGINT, previous_handlers[signal.SIGINT])
            signal.signal(signal.SIGTERM, previous_handlers[signal.SIGTERM])

    def resume(self) -> None:
        checkpoint_path = self.runtime_path / "checkpoint.json"
        cursor_path = self.runtime_path / "cursors.json"
        if checkpoint_path.exists():
            self.state = self.checkpoint_manager.load(checkpoint_path)
        self.cursor_store.load(cursor_path)
        self.run_follow()

    def _process_one(self, signal_payload: dict[str, Any]) -> None:
        signal_id = str(signal_payload["signal_id"])
        if self.cursor_store.is_committed(signal_id):
            return
        self.process_signal(self.state, signal_payload)
        event = TraceEvent(
            event_id=f"trace:signal:{signal_id}",
            timestamp=signal_payload.get("timestamp", _now()),
            transition_type="external",
            trigger_type="signal",
            operator="runtime_signal_ingest",
            pre_state_ref=f"state:{len(self.state.trace_state)}",
            post_state_ref=f"state:{len(self.state.trace_state) + 1}",
            input_refs=[f"signal:{signal_id}"],
            diff={"signal_id": signal_id},
            rationale=f"processed signal {signal_id}",
        )
        self.state.trace_state.append(event)
        if self._event_log is not None:
            self._event_log.append(event)
        self.cursor_store.commit(signal_id)

    def _default_process_signal(self, state: ConsciousState, signal_payload: dict[str, Any]) -> None:
        signal_id = str(signal_payload["signal_id"])
        processed = list(state.runtime_metadata.get("processed_signals", []))
        processed.append(signal_id)
        state.runtime_metadata["processed_signals"] = processed
        if "last_signal_id" in signal_payload:
            state.runtime_metadata["last_signal_id"] = signal_payload["last_signal_id"]

    def _graceful_shutdown(self) -> None:
        self.runtime_path.mkdir(parents=True, exist_ok=True)
        self.checkpoint_manager.save(self.state, self.runtime_path / "checkpoint.json")
        self.cursor_store.save(self.runtime_path / "cursors.json")


__all__ = ["AgentRuntime"]
