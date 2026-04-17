"""Embedding adapter primitives."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
import re
from typing import List, Protocol, runtime_checkable


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


@dataclass
class HashingEmbeddingAdapter:
    """Deterministic lexical-semantic embedding adapter without external APIs.

    This adapter is intended for local semantic retrieval when a production
    embedding endpoint is unavailable. It uses hashed token n-grams and L2
    normalization so that related prompts share a stable sparse vector space.
    """

    dimension: int = 384
    min_n: int = 1
    max_n: int = 2
    lowercase: bool = True

    def embed(self, text: str) -> List[float]:
        content = text.lower() if self.lowercase else text
        tokens = re.findall(r"[A-Za-zА-Яа-я0-9_]+", content)
        vector = [0.0] * max(self.dimension, 1)
        if not tokens:
            return vector

        upper_n = max(self.min_n, self.max_n)
        for ngram_size in range(max(1, self.min_n), upper_n + 1):
            if len(tokens) < ngram_size:
                continue
            for start in range(len(tokens) - ngram_size + 1):
                gram = " ".join(tokens[start : start + ngram_size])
                digest = hashlib.blake2b(gram.encode("utf-8"), digest_size=8).digest()
                index = int.from_bytes(digest[:4], "little") % len(vector)
                sign = 1.0 if digest[4] % 2 == 0 else -1.0
                vector[index] += sign

        norm = math.sqrt(sum(value * value for value in vector))
        if norm > 0.0:
            vector = [value / norm for value in vector]
        return vector


__all__ = ["DeterministicEmbeddingAdapter", "EmbeddingAdapter", "HashingEmbeddingAdapter"]
