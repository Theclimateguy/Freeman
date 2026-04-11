"""Stream cursor persistence for local Freeman runtime."""

from __future__ import annotations

import json
from pathlib import Path


class StreamCursorStore:
    """Track committed signal ids for at-least-once delivery with idempotent mutation."""

    def __init__(self) -> None:
        self._committed: set[str] = set()

    def is_committed(self, signal_id: str) -> bool:
        return str(signal_id) in self._committed

    def commit(self, signal_id: str) -> None:
        self._committed.add(str(signal_id))

    def save(self, path: Path) -> None:
        target = Path(path).resolve()
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps({"committed_signal_ids": sorted(self._committed)}, indent=2), encoding="utf-8")

    def load(self, path: Path) -> None:
        source = Path(path).resolve()
        if not source.exists():
            self._committed = set()
            return
        payload = json.loads(source.read_text(encoding="utf-8"))
        self._committed = {str(value) for value in payload.get("committed_signal_ids", [])}


__all__ = ["StreamCursorStore"]
