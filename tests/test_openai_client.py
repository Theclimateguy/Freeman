from __future__ import annotations

import json

import pytest

from freeman.llm.deepseek import DeepSeekChatClient
from freeman.llm.openai import OpenAIChatClient


class _FakeResponse:
    def __init__(self, payload: dict) -> None:
        self.payload = payload

    def __enter__(self) -> "_FakeResponse":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        return None

    def read(self) -> bytes:
        return json.dumps(self.payload).encode("utf-8")


def test_openai_chat_client_parses_json_object_mode(monkeypatch) -> None:
    payloads: list[dict] = []

    def fake_urlopen(request, timeout):  # noqa: ANN001
        del timeout
        payloads.append(json.loads(request.data.decode("utf-8")))
        return _FakeResponse({"choices": [{"message": {"content": '{"ok": true, "provider": "openai"}'}}]})

    monkeypatch.setattr("freeman.llm.openai.urlopen", fake_urlopen)
    client = OpenAIChatClient(api_key="test-key", base_url="https://example.test/v1")

    response = client.chat_json([{"role": "user", "content": "return json"}], temperature=0.0, max_tokens=64)

    assert response == {"ok": True, "provider": "openai"}
    assert payloads[0]["response_format"] == {"type": "json_object"}
    assert payloads[0]["max_tokens"] == 64


def test_openai_chat_client_repair_schema_uses_structured_prompt(monkeypatch) -> None:
    calls: list[dict] = []

    def fake_chat_json(self, messages, *, temperature=0.2, max_tokens=None):  # noqa: ANN001
        calls.append(
            {
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
            }
        )
        return {"schema": {"domain_id": "demo"}, "policies": [], "assumptions": ["patched"]}

    monkeypatch.setattr(OpenAIChatClient, "chat_json", fake_chat_json)
    client = OpenAIChatClient(api_key="test-key")

    repaired = client.repair_schema(
        {"schema": {"domain_id": "broken"}},
        [{"check_name": "compile_error", "details": {"field": "schema"}}],
        domain_description="demo brief",
        error_history=[{"attempt": 1, "verifier_error": "compile_error"}],
        verifier_schema_spec={"required_top_level_keys": ["domain_id"]},
        repair_stage="accumulated",
        max_tokens=321,
    )

    assert repaired["assumptions"] == ["patched"]
    assert calls[0]["messages"][0]["content"].startswith("You repair Freeman simulation packages")
    assert "\"repair_stage\": \"accumulated\"" in calls[0]["messages"][1]["content"]
    assert "verifier_schema_spec" in calls[0]["messages"][1]["content"]
    assert calls[0]["temperature"] == pytest.approx(0.1)
    assert calls[0]["max_tokens"] == 321


def test_deepseek_embed_fails_loudly() -> None:
    client = DeepSeekChatClient(api_key="test-key")

    with pytest.raises(RuntimeError, match="does not provide embeddings"):
        client.embed("water stress")
