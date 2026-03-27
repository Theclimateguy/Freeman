"""Embedding adapter primitives."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable, List


@runtime_checkable
class EmbeddingAdapter(Protocol):
    """Minimal embedding interface used by semantic memory."""

    def embed(self, text: str) -> List[float]:
        """Return an embedding vector for the provided text."""


@dataclass
class DeterministicEmbeddingAdapter:
    """Offline-safe embedding stub used in tests and local reindexing."""

    dimension: int = 1536
    fill_value: float = 0.0

    def embed(self, text: str) -> List[float]:
        del text
        return [float(self.fill_value)] * self.dimension


__all__ = ["DeterministicEmbeddingAdapter", "EmbeddingAdapter"]
