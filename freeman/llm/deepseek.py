"""Minimal DeepSeek API client."""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

REPAIR_SYSTEM_PROMPT = """You repair Freeman simulation packages using structured verifier feedback.

Return exactly one JSON object with:
- schema: corrected Freeman domain schema
- policies: corrected Freeman Policy snapshots
- assumptions: short list of assumptions

Rules:
- Preserve the domain semantics and actor/resource naming unless the feedback requires otherwise.
- Repair only the fields implicated by the feedback when possible.
- Use the violation details, including field names, observed values, expected bounds, and repair_targets.
- Keep the package compact and numerically stable.
- Use only Freeman-supported evolution types: linear, stock_flow, logistic, threshold, coupled.
- Use the verifier schema spec when provided. A verifier-clean package matters more than preserving a broken field.
- If error history is provided, avoid repeating earlier rejected structures.
- Return JSON only.
"""


def _strip_code_fences(text: str) -> str:
    """Remove surrounding Markdown code fences from a model response."""

    stripped = text.strip()
    match = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", stripped, flags=re.DOTALL)
    return match.group(1) if match else stripped


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
        return json.loads(_strip_code_fences(content))

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
        """Offline-safe embedding stub for adapters that only support chat."""

        del text
        return [0.0] * 1536

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

        prompt_payload = {
            "domain_description": domain_description,
            "package": package,
            "violations": violations,
            "repair_stage": repair_stage,
            "error_history": error_history or [],
            "verifier_schema_spec": verifier_schema_spec or {},
        }
        messages = [
            {"role": "system", "content": REPAIR_SYSTEM_PROMPT},
            {"role": "user", "content": json.dumps(prompt_payload, ensure_ascii=False)},
        ]
        return self.chat_json(messages, temperature=0.1, max_tokens=max_tokens)
