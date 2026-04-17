"""Shared config-driven builders for Freeman interface layers."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from freeman.llm.adapter import DeterministicEmbeddingAdapter, HashingEmbeddingAdapter
from freeman.llm.deepseek import DeepSeekChatClient
from freeman.llm.ollama import OllamaChatClient, OllamaEmbeddingClient
from freeman.llm.openai import OpenAIChatClient, OpenAIEmbeddingClient
from freeman.memory.knowledgegraph import KnowledgeGraph
from freeman.memory.vectorstore import KGVectorStore


def resolve_path(base: Path, candidate: str | None, default: str) -> Path:
    """Resolve a possibly-relative path against ``base``."""

    target = Path(candidate or default).expanduser()
    return target if target.is_absolute() else (base / target).resolve()


def resolve_memory_json_path(config: dict[str, Any], *, config_path: Path) -> Path:
    memory_cfg = config.get("memory", {})
    return resolve_path(config_path.parent, memory_cfg.get("json_path"), "./data/kg_state.json")


def resolve_runtime_path(config: dict[str, Any], *, config_path: Path) -> Path:
    runtime_cfg = config.get("runtime", {})
    return resolve_path(config_path.parent, runtime_cfg.get("runtime_path"), "./data/runtime")


def resolve_event_log_path(config: dict[str, Any], *, config_path: Path) -> Path:
    runtime_cfg = config.get("runtime", {})
    return resolve_path(config_path.parent, runtime_cfg.get("event_log_path"), "./data/runtime/event_log.jsonl")


def resolve_semantic_min_score(config: dict[str, Any]) -> float:
    """Return the retrieval acceptance floor used by semantic query layers."""

    memory_cfg = config.get("memory", {})
    return float(memory_cfg.get("semantic_min_score", 0.05))


def build_vectorstore(config: dict[str, Any], *, config_path: Path) -> KGVectorStore | None:
    """Build the configured KG vector store, if enabled."""

    memory_cfg = config.get("memory", {})
    vector_cfg = memory_cfg.get("vector_store", {})
    if not vector_cfg.get("enabled", False):
        return None
    backend = str(vector_cfg.get("backend", "chroma")).lower()
    if backend != "chroma":
        raise ValueError(f"Unsupported vector store backend: {backend}")
    path = resolve_path(config_path.parent, vector_cfg.get("path"), "./data/chroma_db")
    collection = str(vector_cfg.get("collection", "kg_nodes"))
    return KGVectorStore(path=path, collection_name=collection)


def build_embedding_adapter(config: dict[str, Any], *, use_stub: bool = False) -> tuple[Any, str]:
    """Build the configured embedding adapter."""

    memory_cfg = config.get("memory", {})
    provider = str(memory_cfg.get("embedding_provider", "")).strip().lower()
    model = str(memory_cfg.get("embedding_model", "nomic-embed-text"))
    timeout_seconds = float(memory_cfg.get("embedding_timeout_seconds", 120.0))
    prompt_prefix = str(memory_cfg.get("embedding_prompt_prefix", ""))
    if use_stub:
        return DeterministicEmbeddingAdapter(), "deterministic_stub"
    if provider in {"deterministic", "stub"}:
        return DeterministicEmbeddingAdapter(), "deterministic_stub"
    if provider in {"hash", "hashing"}:
        dimension = int(memory_cfg.get("hashing_embedding_dimension", 384))
        return HashingEmbeddingAdapter(dimension=dimension), f"hashing:{dimension}"
    if provider == "openai":
        api_key = os.getenv("OPENAI_API_KEY", "").strip()
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY is required when memory.embedding_provider=openai")
        return OpenAIEmbeddingClient(api_key=api_key, model=model), f"openai:{model}"
    if provider in {"", "ollama"}:
        base_url = str(memory_cfg.get("embedding_base_url", os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434")))
        return (
            OllamaEmbeddingClient(
                model=model,
                base_url=base_url,
                timeout_seconds=timeout_seconds,
                prompt_prefix=prompt_prefix,
            ),
            f"ollama:{model}",
        )
    raise ValueError(f"Unsupported embedding provider: {provider}")


def build_chat_client(config: dict[str, Any]) -> tuple[Any | None, str | None]:
    """Build the configured chat client used by answer-generation layers."""

    llm_cfg = config.get("llm", {})
    provider = str(llm_cfg.get("provider", "")).strip().lower()
    model = str(llm_cfg.get("model", "")).strip()
    base_url = str(llm_cfg.get("base_url", "")).strip()
    timeout_seconds = float(llm_cfg.get("timeout_seconds", 90.0))
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


def build_knowledge_graph(
    config: dict[str, Any],
    *,
    config_path: Path,
    embedding_adapter: Any | None = None,
    vectorstore: KGVectorStore | None = None,
    auto_load: bool = True,
    auto_save: bool = True,
) -> KnowledgeGraph:
    """Instantiate a config-backed knowledge graph."""

    return KnowledgeGraph(
        json_path=resolve_memory_json_path(config, config_path=config_path),
        auto_load=auto_load,
        auto_save=auto_save,
        llm_adapter=embedding_adapter,
        vectorstore=vectorstore,
    )


__all__ = [
    "build_chat_client",
    "build_embedding_adapter",
    "build_knowledge_graph",
    "build_vectorstore",
    "resolve_event_log_path",
    "resolve_memory_json_path",
    "resolve_path",
    "resolve_runtime_path",
    "resolve_semantic_min_score",
]
