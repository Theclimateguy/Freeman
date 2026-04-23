"""Shared builders for the Freeman lite interface."""

from __future__ import annotations

import os
from typing import Any

from freeman.llm.deepseek import DeepSeekChatClient
from freeman.llm.ollama import OllamaChatClient, OllamaEmbeddingClient
from freeman.llm.openai import OpenAIChatClient, OpenAIEmbeddingClient


def _section(config: Any, name: str) -> Any:
    if hasattr(config, name):
        return getattr(config, name)
    if isinstance(config, dict):
        return config.get(name, {})
    return {}


def _cfg_get(section: Any, name: str, default: Any) -> Any:
    if hasattr(section, name):
        return getattr(section, name)
    if isinstance(section, dict):
        return section.get(name, default)
    return default


def build_embedding_adapter(config: dict[str, Any], *, use_stub: bool = False) -> tuple[Any, str]:
    """Build an embedding adapter only when explicitly requested."""

    if use_stub:
        return OllamaEmbeddingClient(model="nomic-embed-text"), "ollama:nomic-embed-text"
    llm_cfg = _section(config, "llm")
    provider = str(_cfg_get(llm_cfg, "provider", "ollama")).strip().lower()
    model = str(_cfg_get(llm_cfg, "model", "nomic-embed-text")).strip()
    base_url = str(_cfg_get(llm_cfg, "base_url", "http://127.0.0.1:11434")).strip()
    timeout_seconds = float(_cfg_get(llm_cfg, "timeout_seconds", 90.0))
    if provider == "openai":
        api_key = os.getenv("OPENAI_API_KEY", "").strip()
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY is required when llm.provider=openai")
        return OpenAIEmbeddingClient(api_key=api_key, model=model), f"openai:{model}"
    return (
        OllamaEmbeddingClient(
            model=model or "nomic-embed-text",
            base_url=base_url or os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434"),
            timeout_seconds=timeout_seconds,
        ),
        f"ollama:{model or 'nomic-embed-text'}",
    )


def build_chat_client(config: dict[str, Any]) -> tuple[Any | None, str | None]:
    """Build the single chat client used by the lite runtime."""

    llm_cfg = _section(config, "llm")
    provider = str(_cfg_get(llm_cfg, "provider", "")).strip().lower()
    model = str(_cfg_get(llm_cfg, "model", "")).strip()
    base_url = str(_cfg_get(llm_cfg, "base_url", "")).strip()
    timeout_seconds = float(_cfg_get(llm_cfg, "timeout_seconds", 90.0))
    if provider in {"", "none"}:
        return None, "llm provider is not configured"
    if provider == "openai":
        api_key = os.getenv("OPENAI_API_KEY", "").strip() or os.getenv("LLM_API_KEY", "").strip()
        if not api_key:
            return None, "OPENAI_API_KEY or LLM_API_KEY is not set"
        return (
            OpenAIChatClient(
                api_key=api_key,
                model=model or "gpt-4o-mini",
                base_url=base_url or "https://api.openai.com/v1",
                timeout_seconds=timeout_seconds,
            ),
            None,
        )
    if provider == "deepseek":
        api_key = os.getenv("DEEPSEEK_API_KEY", "").strip() or os.getenv("LLM_API_KEY", "").strip()
        if not api_key:
            return None, "DEEPSEEK_API_KEY or LLM_API_KEY is not set"
        return (
            DeepSeekChatClient(
                api_key=api_key,
                model=model or "deepseek-chat",
                base_url=base_url or "https://api.deepseek.com",
                timeout_seconds=timeout_seconds,
            ),
            None,
        )
    if provider == "ollama":
        return (
            OllamaChatClient(
                model=model or "qwen2.5-coder:14b",
                base_url=base_url or os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434"),
                timeout_seconds=timeout_seconds,
            ),
            None,
        )
    return None, f"unsupported llm provider: {provider}"


__all__ = [
    "build_chat_client",
    "build_embedding_adapter",
]
