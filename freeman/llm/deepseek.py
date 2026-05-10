"""Minimal DeepSeek API client."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from freeman.llm.structured import build_repair_messages, extract_json_object


@dataclass
class DeepSeekChatClient:
    """Small OpenAI-compatible client for DeepSeek chat completions."""

    api_key: str
    model: str = "deepseek-chat"
    base_url: str = "https://api.deepseek.com"
    timeout_seconds: float = 90.0
    max_retries: int = 2
    retry_backoff_seconds: float = 1.5

    def create_chat_completion(
        self,
        messages: List[Dict[str, str]],
        *,
        temperature: float = 0.2,
        max_tokens: Optional[int] = None,
        response_format: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        """Call the DeepSeek chat completion endpoint and return parsed JSON."""

        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
        }
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens
        if response_format is not None:
            payload["response_format"] = response_format

        request = Request(
            url=f"{self.base_url}/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )

        retryable_http = {408, 409, 429, 500, 502, 503, 504}
        attempts = max(int(self.max_retries), 0) + 1
        for attempt in range(1, attempts + 1):
            try:
                with urlopen(request, timeout=self.timeout_seconds) as response:
                    return json.loads(response.read().decode("utf-8"))
            except HTTPError as exc:  # pragma: no cover - exercised only against live API
                body = exc.read().decode("utf-8", errors="replace")
                if exc.code in retryable_http and attempt < attempts:
                    time.sleep(self.retry_backoff_seconds * attempt)
                    continue
                raise RuntimeError(f"DeepSeek HTTP {exc.code}: {body}") from exc
            except (URLError, TimeoutError, ConnectionResetError, OSError) as exc:  # pragma: no cover - live API only
                if attempt < attempts:
                    time.sleep(self.retry_backoff_seconds * attempt)
                    continue
                raise RuntimeError(f"DeepSeek connection error: {exc}") from exc
        raise RuntimeError("DeepSeek request failed after retries.")

    def chat_json(
        self,
        messages: List[Dict[str, str]],
        *,
        temperature: float = 0.2,
        max_tokens: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Request a JSON object from DeepSeek and parse the response content."""

        response = self.create_chat_completion(
            messages,
            temperature=temperature,
            max_tokens=max_tokens,
            response_format={"type": "json_object"},
        )
        content = response["choices"][0]["message"]["content"]
        return extract_json_object(content, provider_name="DeepSeek")

    def chat_text(
        self,
        messages: List[Dict[str, str]],
        *,
        temperature: float = 0.2,
        max_tokens: Optional[int] = None,
    ) -> str:
        """Request free-form text from DeepSeek."""

        response = self.create_chat_completion(messages, temperature=temperature, max_tokens=max_tokens)
        return response["choices"][0]["message"]["content"].strip()

    def embed(self, text: str) -> List[float]:
        """Fail loudly because this adapter is chat-only."""

        del text
        raise RuntimeError(
            "DeepSeekChatClient does not provide embeddings. "
            "Configure memory.embedding_provider to openai, ollama, or hashing."
        )

    def repair_schema(
        self,
        package: Dict[str, Any],
        violations: List[Dict[str, Any]],
        *,
        domain_description: str = "",
        error_history: List[Dict[str, Any]] | None = None,
        verifier_schema_spec: Dict[str, Any] | None = None,
        repair_stage: str = "standard",
        max_tokens: int = 4000,
    ) -> Dict[str, Any]:
        """Repair a Freeman package using structured verifier feedback."""

        messages = build_repair_messages(
            package,
            violations,
            domain_description=domain_description,
            error_history=error_history,
            verifier_schema_spec=verifier_schema_spec,
            repair_stage=repair_stage,
        )
        return self.chat_json(messages, temperature=0.1, max_tokens=max_tokens)
