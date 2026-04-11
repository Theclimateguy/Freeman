"""LLM orchestration helpers for Freeman."""

from freeman.llm.adapter import DeterministicEmbeddingAdapter, EmbeddingAdapter, HashingEmbeddingAdapter
from freeman.llm.deepseek import DeepSeekChatClient
from freeman.llm.explanation_renderer import ExplanationRenderer
from freeman.llm.identity_narrator import IdentityNarrator
from freeman.llm.orchestrator import DeepSeekFreemanOrchestrator, LLMDrivenSimulationRun
from freeman.llm.ollama import OllamaChatClient, OllamaEmbeddingClient
from freeman.llm.openai import OpenAIChatClient, OpenAIEmbeddingClient

__all__ = [
    "DeepSeekChatClient",
    "DeepSeekFreemanOrchestrator",
    "DeterministicEmbeddingAdapter",
    "EmbeddingAdapter",
    "ExplanationRenderer",
    "HashingEmbeddingAdapter",
    "IdentityNarrator",
    "LLMDrivenSimulationRun",
    "OpenAIChatClient",
    "OllamaChatClient",
    "OllamaEmbeddingClient",
    "OpenAIEmbeddingClient",
]
