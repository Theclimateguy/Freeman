"""LLM orchestration helpers for Freeman."""

from freeman_librarian.llm.adapter import DeterministicEmbeddingAdapter, EmbeddingAdapter, HashingEmbeddingAdapter
from freeman_librarian.llm.deepseek import DeepSeekChatClient
from freeman_librarian.llm.explanation_renderer import ExplanationRenderer
from freeman_librarian.llm.identity_narrator import IdentityNarrator
from freeman_librarian.llm.ollama import OllamaChatClient, OllamaEmbeddingClient
from freeman_librarian.llm.openai import OpenAIChatClient, OpenAIEmbeddingClient

__all__ = [
    "DeepSeekChatClient",
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
]


def __getattr__(name: str):
    if name in {"FreemanOrchestrator", "DeepSeekFreemanOrchestrator", "LLMDrivenSimulationRun"}:
        from freeman_librarian.llm.orchestrator import DeepSeekFreemanOrchestrator, FreemanOrchestrator, LLMDrivenSimulationRun

        exports = {
            "FreemanOrchestrator": FreemanOrchestrator,
            "DeepSeekFreemanOrchestrator": DeepSeekFreemanOrchestrator,
            "LLMDrivenSimulationRun": LLMDrivenSimulationRun,
        }
        return exports[name]
    raise AttributeError(name)
