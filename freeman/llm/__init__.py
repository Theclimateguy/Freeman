"""LLM orchestration helpers for Freeman."""

from freeman.llm.adapter import DeterministicEmbeddingAdapter, EmbeddingAdapter, HashingEmbeddingAdapter
from freeman.llm.circuit_breaker import CircuitBreaker, CircuitBreakerChatClient, CircuitOpenError, wrap_chat_client
from freeman.llm.deepseek import DeepSeekChatClient
from freeman.llm.explanation_renderer import ExplanationRenderer
from freeman.llm.identity_narrator import IdentityNarrator
from freeman.llm.ollama import OllamaChatClient, OllamaEmbeddingClient
from freeman.llm.openai import OpenAIChatClient, OpenAIEmbeddingClient

__all__ = [
    "DeepSeekChatClient",
    "CircuitBreaker",
    "CircuitBreakerChatClient",
    "CircuitOpenError",
    "DeterministicEmbeddingAdapter",
    "EmbeddingAdapter",
    "ExplanationRenderer",
    "FreemanOrchestrator",
    "HashingEmbeddingAdapter",
    "IdentityNarrator",
    "LLMDrivenSimulationRun",
    "OpenAIChatClient",
    "OllamaChatClient",
    "OllamaEmbeddingClient",
    "OpenAIEmbeddingClient",
    "DeepSeekFreemanOrchestrator",
    "wrap_chat_client",
]


def __getattr__(name: str):
    if name in {"FreemanOrchestrator", "DeepSeekFreemanOrchestrator", "LLMDrivenSimulationRun"}:
        from freeman.llm.orchestrator import DeepSeekFreemanOrchestrator, FreemanOrchestrator, LLMDrivenSimulationRun

        exports = {
            "FreemanOrchestrator": FreemanOrchestrator,
            "DeepSeekFreemanOrchestrator": DeepSeekFreemanOrchestrator,
            "LLMDrivenSimulationRun": LLMDrivenSimulationRun,
        }
        return exports[name]
    raise AttributeError(name)
