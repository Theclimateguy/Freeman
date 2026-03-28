"""Unit tests for the local hashing embedding adapter."""

from __future__ import annotations

from freeman.llm import HashingEmbeddingAdapter


def test_hashing_embedding_is_deterministic() -> None:
    adapter = HashingEmbeddingAdapter(dimension=64)

    first = adapter.embed("shipping costs and inflation risk")
    second = adapter.embed("shipping costs and inflation risk")

    assert first == second


def test_hashing_embedding_normalizes_non_empty_text() -> None:
    adapter = HashingEmbeddingAdapter(dimension=64)

    vector = adapter.embed("relationship stress and jealousy")

    norm = sum(value * value for value in vector) ** 0.5
    assert 0.99 <= norm <= 1.01


def test_hashing_embedding_returns_zero_vector_for_empty_text() -> None:
    adapter = HashingEmbeddingAdapter(dimension=32)

    vector = adapter.embed("")

    assert len(vector) == 32
    assert all(value == 0.0 for value in vector)
