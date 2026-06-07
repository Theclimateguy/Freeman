"""Small synchronous circuit breaker for LLM provider calls."""

from __future__ import annotations

from dataclasses import dataclass, field
import time
from typing import Any, Callable, Literal

CircuitState = Literal["CLOSED", "OPEN", "HALF_OPEN"]


class CircuitOpenError(RuntimeError):
    """Raised when a provider call is blocked by an open circuit."""


@dataclass
class CircuitBreaker:
    """Track provider failures and stop repeated calls during an outage."""

    failure_threshold: int = 3
    reset_timeout: float = 60.0
    state: CircuitState = "CLOSED"
    failure_count: int = 0
    opened_at: float | None = None

    def call(self, fn: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
        if self.state == "OPEN":
            elapsed = time.monotonic() - float(self.opened_at or 0.0)
            if elapsed < float(self.reset_timeout):
                raise CircuitOpenError("LLM circuit is OPEN; provider calls are temporarily blocked.")
            self.state = "HALF_OPEN"
        try:
            result = fn(*args, **kwargs)
        except Exception:
            self.failure_count += 1
            if self.failure_count >= max(int(self.failure_threshold), 1):
                self.state = "OPEN"
                self.opened_at = time.monotonic()
            raise
        self.failure_count = 0
        self.opened_at = None
        self.state = "CLOSED"
        return result


@dataclass
class CircuitBreakerChatClient:
    """Proxy that applies one circuit breaker to chat-client methods."""

    client: Any
    breaker: CircuitBreaker = field(default_factory=CircuitBreaker)

    def __getattr__(self, name: str) -> Any:
        return getattr(self.client, name)

    def create_chat_completion(self, *args: Any, **kwargs: Any) -> Any:
        return self.breaker.call(self.client.create_chat_completion, *args, **kwargs)

    def chat_json(self, *args: Any, **kwargs: Any) -> Any:
        return self.breaker.call(self.client.chat_json, *args, **kwargs)

    def chat_text(self, *args: Any, **kwargs: Any) -> Any:
        return self.breaker.call(self.client.chat_text, *args, **kwargs)

    def repair_schema(self, *args: Any, **kwargs: Any) -> Any:
        return self.breaker.call(self.client.repair_schema, *args, **kwargs)


def wrap_chat_client(
    client: Any,
    *,
    enabled: bool = True,
    failure_threshold: int = 3,
    reset_timeout: float = 60.0,
) -> Any:
    """Wrap a chat client with a circuit breaker when enabled."""

    if client is None or not enabled:
        return client
    return CircuitBreakerChatClient(
        client,
        breaker=CircuitBreaker(failure_threshold=failure_threshold, reset_timeout=reset_timeout),
    )


__all__ = ["CircuitBreaker", "CircuitBreakerChatClient", "CircuitOpenError", "wrap_chat_client"]
