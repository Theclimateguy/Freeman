"""Local Ollama embedding client."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


def _coerce_embedding_list(values: Iterable[Any]) -> List[float]:
    return [float(value) for value in values]


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


__all__ = ["OllamaEmbeddingClient"]
