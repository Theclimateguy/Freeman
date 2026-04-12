"""Local Ollama clients (chat + embeddings)."""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List
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


def _coerce_embedding_list(values: Iterable[Any]) -> List[float]:
    return [float(value) for value in values]


def _strip_code_fences(text: str) -> str:
    stripped = text.strip()
    match = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", stripped, flags=re.DOTALL)
    return match.group(1) if match else stripped


def _extract_json_object(text: str) -> dict[str, Any]:
    """Parse a JSON object from model output, with fenced and substring fallback."""

    cleaned = _strip_code_fences(text)
    try:
        parsed = json.loads(cleaned)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start >= 0 and end > start:
        parsed = json.loads(cleaned[start : end + 1])
        if isinstance(parsed, dict):
            return parsed
    raise ValueError("Ollama response did not contain a valid JSON object.")


@dataclass
class OllamaChatClient:
    """OpenAI-style chat interface over a local Ollama server."""

    model: str = "qwen2.5-coder:14b"
    base_url: str = "http://127.0.0.1:11434"
    timeout_seconds: float = 120.0
    max_retries: int = 2
    retry_backoff_seconds: float = 1.5

    def create_chat_completion(
        self,
        messages: List[Dict[str, str]],
        *,
        temperature: float = 0.2,
        max_tokens: int | None = None,
        json_mode: bool = False,
    ) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": float(temperature),
            },
        }
        if max_tokens is not None:
            payload["options"]["num_predict"] = int(max_tokens)
        if json_mode:
            payload["format"] = "json"
        return self._request("/api/chat", payload)

    def chat_json(
        self,
        messages: List[Dict[str, str]],
        *,
        temperature: float = 0.2,
        max_tokens: int | None = None,
    ) -> Dict[str, Any]:
        response = self.create_chat_completion(
            messages,
            temperature=temperature,
            max_tokens=max_tokens,
            json_mode=True,
        )
        content = str(response.get("message", {}).get("content", "")).strip()
        return _extract_json_object(content)

    def chat_text(
        self,
        messages: List[Dict[str, str]],
        *,
        temperature: float = 0.2,
        max_tokens: int | None = None,
    ) -> str:
        response = self.create_chat_completion(
            messages,
            temperature=temperature,
            max_tokens=max_tokens,
            json_mode=False,
        )
        return str(response.get("message", {}).get("content", "")).strip()

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

    def _request(self, path: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        request = Request(
            url=f"{self.base_url.rstrip('/')}{path}",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        retryable_http = {404, 408, 409, 429, 500, 502, 503, 504}
        attempts = max(int(self.max_retries), 0) + 1
        for attempt in range(1, attempts + 1):
            try:
                with urlopen(request, timeout=self.timeout_seconds) as response:
                    return json.loads(response.read().decode("utf-8"))
            except HTTPError as exc:  # pragma: no cover - live server only
                body = exc.read().decode("utf-8", errors="replace")
                if exc.code in retryable_http and attempt < attempts:
                    time.sleep(self.retry_backoff_seconds * attempt)
                    continue
                raise RuntimeError(f"Ollama HTTP {exc.code} on {path}: {body}") from exc
            except (URLError, TimeoutError, ConnectionResetError, OSError) as exc:  # pragma: no cover - live server only
                if attempt < attempts:
                    time.sleep(self.retry_backoff_seconds * attempt)
                    continue
                raise RuntimeError(f"Ollama connection error on {path}: {exc}") from exc
        raise RuntimeError(f"Ollama request failed for {path}.")


@dataclass
class OllamaEmbeddingClient:
    """Embedding client for a local Ollama server.

    The Ollama API has two embedding variants in the wild:
    - ``POST /api/embed`` with ``{"input": ...}``
    - ``POST /api/embeddings`` with ``{"prompt": ...}``

    This client prefers ``/api/embed`` for batched embeddings and falls back
    to ``/api/embeddings`` for compatibility with older model endpoints.
    """

    model: str = "nomic-embed-text"
    base_url: str = "http://127.0.0.1:11434"
    timeout_seconds: float = 120.0
    max_retries: int = 2
    retry_backoff_seconds: float = 1.5
    prompt_prefix: str = ""

    def embed(self, text: str) -> List[float]:
        """Embed one text item."""

        prepared = self._prepare_text(text)
        try:
            result = self._request("/api/embed", {"model": self.model, "input": prepared})
            return self._parse_single_embedding(result)
        except RuntimeError as exc:
            if "/api/embed" not in str(exc):
                raise
        result = self._request("/api/embeddings", {"model": self.model, "prompt": prepared})
        return self._parse_single_embedding(result)

    def embed_many(self, texts: Iterable[str]) -> List[List[float]]:
        """Embed multiple texts, batching when ``/api/embed`` is available."""

        prepared = [self._prepare_text(text) for text in texts]
        if not prepared:
            return []
        try:
            result = self._request("/api/embed", {"model": self.model, "input": prepared})
            return self._parse_many_embeddings(result)
        except RuntimeError as exc:
            if "/api/embed" not in str(exc):
                raise
        return [self.embed(text) for text in prepared]

    def _prepare_text(self, text: str) -> str:
        stripped = text.strip()
        if self.prompt_prefix:
            return f"{self.prompt_prefix}{stripped}"
        if self.model.startswith("mxbai-embed-large"):
            return f"Represent this sentence for searching relevant passages: {stripped}"
        return stripped

    def _request(self, path: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        request = Request(
            url=f"{self.base_url.rstrip('/')}{path}",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        retryable_http = {404, 408, 409, 429, 500, 502, 503, 504}
        attempts = max(int(self.max_retries), 0) + 1
        for attempt in range(1, attempts + 1):
            try:
                with urlopen(request, timeout=self.timeout_seconds) as response:
                    return json.loads(response.read().decode("utf-8"))
            except HTTPError as exc:  # pragma: no cover - live server only
                body = exc.read().decode("utf-8", errors="replace")
                if exc.code in retryable_http and attempt < attempts:
                    time.sleep(self.retry_backoff_seconds * attempt)
                    continue
                raise RuntimeError(f"Ollama HTTP {exc.code} on {path}: {body}") from exc
            except (URLError, TimeoutError, ConnectionResetError, OSError) as exc:  # pragma: no cover - live server only
                if attempt < attempts:
                    time.sleep(self.retry_backoff_seconds * attempt)
                    continue
                raise RuntimeError(f"Ollama connection error on {path}: {exc}") from exc
        raise RuntimeError(f"Ollama request failed for {path}.")

    def _parse_single_embedding(self, payload: Dict[str, Any]) -> List[float]:
        if "embedding" in payload:
            return _coerce_embedding_list(payload["embedding"])
        if "embeddings" in payload and payload["embeddings"]:
            first = payload["embeddings"][0]
            return _coerce_embedding_list(first["embedding"] if isinstance(first, dict) and "embedding" in first else first)
        raise RuntimeError("Ollama embedding response did not contain vectors.")

    def _parse_many_embeddings(self, payload: Dict[str, Any]) -> List[List[float]]:
        if "embeddings" in payload:
            embeddings = payload["embeddings"]
            if embeddings and isinstance(embeddings[0], dict) and "embedding" in embeddings[0]:
                return [_coerce_embedding_list(item["embedding"]) for item in embeddings]
            return [_coerce_embedding_list(item) for item in embeddings]
        if "embedding" in payload:
            return [_coerce_embedding_list(payload["embedding"])]
        raise RuntimeError("Ollama embedding response did not contain vectors.")


__all__ = ["OllamaChatClient", "OllamaEmbeddingClient"]
