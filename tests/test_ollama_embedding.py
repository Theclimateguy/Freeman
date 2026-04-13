"""Tests for the local Ollama embedding client."""

from __future__ import annotations

import json
from urllib.error import HTTPError

import pytest

from freeman.llm.ollama import OllamaChatClient, OllamaEmbeddingClient


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


def test_ollama_chat_client_parses_json_mode(monkeypatch) -> None:
    payloads: list[dict] = []

    def fake_urlopen(request, timeout):  # noqa: ANN001
        payloads.append(json.loads(request.data.decode("utf-8")))
        return _FakeResponse({"message": {"content": '{"ok": true, "mode": "json"}'}})

    monkeypatch.setattr("freeman.llm.ollama.urlopen", fake_urlopen)
    client = OllamaChatClient(model="qwen2.5-coder:14b", base_url="http://localhost:11434")

    response = client.chat_json([{"role": "user", "content": "return json"}], temperature=0.0, max_tokens=32)

    assert response == {"ok": True, "mode": "json"}
    assert payloads[0]["format"] == "json"
    assert payloads[0]["options"]["num_predict"] == 32


def test_ollama_chat_client_repair_schema_uses_structured_prompt(monkeypatch) -> None:
    requests: list[dict] = []

    def fake_chat_json(self, messages, *, temperature=0.2, max_tokens=None):  # noqa: ANN001
        requests.append(
            {
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
            }
        )
        return {"schema": {"domain_id": "demo", "actors": [], "resources": [], "relations": [], "outcomes": [], "causal_dag": []}, "policies": [], "assumptions": ["patched"]}

    monkeypatch.setattr(OllamaChatClient, "chat_json", fake_chat_json)
    client = OllamaChatClient(model="qwen2.5-coder:14b", base_url="http://localhost:11434")

    repaired = client.repair_schema(
        {"schema": {"domain_id": "broken"}},
        [{"check_name": "compile_error", "details": {"field": "schema"}}],
        domain_description="demo brief",
        error_history=[{"attempt": 1, "verifier_error": "compile_error: schema"}],
        verifier_schema_spec={"required_top_level_keys": ["domain_id"]},
        repair_stage="accumulated",
        max_tokens=123,
    )

    assert repaired["assumptions"] == ["patched"]
    assert requests[0]["messages"][0]["content"].startswith("You repair Freeman simulation packages")
    assert "error_history" in requests[0]["messages"][1]["content"]
    assert "verifier_schema_spec" in requests[0]["messages"][1]["content"]
    assert "\"repair_stage\": \"accumulated\"" in requests[0]["messages"][1]["content"]
    assert requests[0]["temperature"] == pytest.approx(0.1)
    assert requests[0]["max_tokens"] == 123
