"""KnowledgeGraph persistence backend protocol."""

from __future__ import annotations

from contextlib import AbstractContextManager
from typing import Any, Protocol


class KGBackend(Protocol):
    """Persistence contract for KnowledgeGraph payloads."""

    def load(self) -> dict[str, Any]:
        """Load a serialized graph payload."""

    def save(self, data: dict[str, Any]) -> None:
        """Persist a serialized graph payload."""

    def transaction(self) -> AbstractContextManager[None]:
        """Return a backend transaction context manager."""

    def node_count(self) -> int:
        """Return persisted node count."""


__all__ = ["KGBackend"]
