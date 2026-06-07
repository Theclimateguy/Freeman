"""JSON KnowledgeGraph backend."""

from __future__ import annotations

from contextlib import contextmanager
import json
from pathlib import Path
from typing import Any, Iterator


class JsonKGBackend:
    """Persist KnowledgeGraph payloads as one JSON file."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path).expanduser().resolve()

    def load(self) -> dict[str, Any]:
        return json.loads(self.path.read_text(encoding="utf-8"))

    def save(self, data: dict[str, Any]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")

    @contextmanager
    def transaction(self) -> Iterator[None]:
        yield

    def node_count(self) -> int:
        if not self.path.exists():
            return 0
        return len(self.load().get("nodes", []))


__all__ = ["JsonKGBackend"]
