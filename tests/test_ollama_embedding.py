"""Tests for the local Ollama embedding client."""

from __future__ import annotations

import json
from urllib.error import HTTPError

import pytest

from freeman.llm.ollama import OllamaEmbeddingClient


class _FakeResponse:
    def __init__(self, payload: dict) -> None:
        self.payload = payload

    def __enter__(self) -> "_FakeResponse":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        return None

    def read(self) -> bytes:
        return json.dumps(self.payload).encode("utf-8")


def test_ollama_client_parses_embed_endpoint(monkeypatch) -> None:
    requests: list[tuple[str, dict]] = []

    def fake_urlopen(request, timeout):  # noqa: ANN001
        requests.append((request.full_url, json.loads(request.data.decode("utf-8"))))
        return _FakeResponse({"embeddings": [[0.1, 0.2, 0.3]]})

    monkeypatch.setattr("freeman.llm.ollama.urlopen", fake_urlopen)
    client = OllamaEmbeddingClient(model="nomic-embed-text", base_url="http://localhost:11434")

    embedding = client.embed("water stress")

    assert embedding == [0.1, 0.2, 0.3]
    assert requests[0][0].endswith("/api/embed")
    assert requests[0][1]["input"] == "water stress"


def test_ollama_client_falls_back_to_legacy_embeddings_endpoint(monkeypatch) -> None:
    calls: list[str] = []

    def fake_urlopen(request, timeout):  # noqa: ANN001
        calls.append(request.full_url)
        if request.full_url.endswith("/api/embed"):
            raise HTTPError(request.full_url, 404, "not found", hdrs=None, fp=None)
        return _FakeResponse({"embedding": [0.4, 0.5]})

    monkeypatch.setattr("freeman.llm.ollama.urlopen", fake_urlopen)
    client = OllamaEmbeddingClient(model="nomic-embed-text", base_url="http://localhost:11434", max_retries=0)

    embedding = client.embed("trust repair")

    assert embedding == [0.4, 0.5]
    assert calls == [
        "http://localhost:11434/api/embed",
        "http://localhost:11434/api/embeddings",
    ]


def test_ollama_client_batches_with_embed_many(monkeypatch) -> None:
    payloads: list[dict] = []

    def fake_urlopen(request, timeout):  # noqa: ANN001
        payloads.append(json.loads(request.data.decode("utf-8")))
        return _FakeResponse({"embeddings": [[1.0, 0.0], [0.0, 1.0]]})

    monkeypatch.setattr("freeman.llm.ollama.urlopen", fake_urlopen)
    client = OllamaEmbeddingClient(model="mxbai-embed-large", base_url="http://localhost:11434")

    embeddings = client.embed_many(["alpha", "beta"])

    assert embeddings == [[1.0, 0.0], [0.0, 1.0]]
    assert payloads[0]["input"] == [
        "Represent this sentence for searching relevant passages: alpha",
        "Represent this sentence for searching relevant passages: beta",
    ]
