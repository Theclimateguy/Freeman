"""Minimal OpenAI-compatible embedding client."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict, List
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


@dataclass
class OpenAIEmbeddingClient:
    """Small client for OpenAI embeddings."""

    api_key: str
    model: str = "text-embedding-3-small"
    base_url: str = "https://api.openai.com/v1"
    timeout_seconds: float = 90.0

    def embed(self, text: str) -> List[float]:
        """Embed text with the configured OpenAI model."""

        payload: Dict[str, Any] = {
            "model": self.model,
            "input": text,
        }
        request = Request(
            url=f"{self.base_url}/embeddings",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )

        try:
            with urlopen(request, timeout=self.timeout_seconds) as response:
                body = json.loads(response.read().decode("utf-8"))
        except HTTPError as exc:  # pragma: no cover - exercised only against live API
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"OpenAI HTTP {exc.code}: {detail}") from exc
        except URLError as exc:  # pragma: no cover - exercised only against live API
            raise RuntimeError(f"OpenAI connection error: {exc}") from exc

        data = body.get("data", [])
        if not data:
            raise RuntimeError("OpenAI embedding response did not contain vectors.")
        return [float(value) for value in data[0]["embedding"]]


__all__ = ["OpenAIEmbeddingClient"]
