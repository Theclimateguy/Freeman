"""Bootstrap assembly for Freeman stream runtime."""

from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path
from typing import Any
from urllib.error import URLError
from urllib.request import urlopen

from freeman.agent.analysispipeline import AnalysisPipeline
from freeman.agent.consciousness import ConsciousState
from freeman.agent.forecastregistry import ForecastRegistry
from freeman.agent.parameterestimator import ParameterEstimator
from freeman.interface.factory import build_embedding_adapter, build_knowledge_graph, build_vectorstore
from freeman.llm import DeepSeekChatClient, FreemanOrchestrator, OllamaChatClient, OpenAIChatClient
from freeman.llm.circuit_breaker import wrap_chat_client
from freeman.memory.knowledgegraph import KnowledgeGraph
from freeman.runtime.bootstrap_contracts import build_bootstrap_contract
from freeman.runtime.lifecycle import (
    BootstrapResult,
    RuntimePaths,
    RuntimeStorage,
    _append_unlogged_trace_events,
    _atomic_write_json,
    _build_sim_config,
    _persist_runtime_state,
    _read_optional_text,
    _resolve_path,
)
from freeman.core.world import WorldState

LOGGER = logging.getLogger("stream_runtime")

def _load_model_tags(base_url: str) -> list[str]:
    url = f"{base_url.rstrip('/')}/api/tags"
    with urlopen(url, timeout=5.0) as response:  # noqa: S310 - local Ollama endpoint
        payload = json.loads(response.read().decode("utf-8"))
    models = payload.get("models", [])
    return [str(model.get("name", "")).strip() for model in models if str(model.get("name", "")).strip()]


def _select_ollama_model(base_url: str, requested: str) -> str:
    if requested and requested.lower() != "auto":
        return requested
    try:
        available = _load_model_tags(base_url)
    except (URLError, OSError, TimeoutError) as exc:
        LOGGER.warning("Could not query Ollama tags (%s). Falling back to qwen2.5-coder:14b.", exc)
        return "qwen2.5-coder:14b"
    if not available:
        return "qwen2.5-coder:14b"
    preferred = [
        "qwen2.5-coder:14b",
        "qwen2.5:14b-instruct",
        "qwen2.5:14b",
        "qwen2.5-coder:7b",
        "qwen2.5:7b-instruct",
        "qwen2.5:7b",
    ]
    lowered = {name.lower(): name for name in available}
    for candidate in preferred:
        if candidate.lower() in lowered:
            return lowered[candidate.lower()]
    qwen_models = [name for name in available if "qwen" in name.lower()]
    if qwen_models:
        return sorted(qwen_models, reverse=True)[0]
    return available[0]


def _build_chat_client(
    *,
    provider: str,
    model: str,
    base_url: str,
    timeout_seconds: float,
    api_key: str = "",
    circuit_breaker_enabled: bool = True,
    circuit_failure_threshold: int = 3,
    circuit_reset_timeout: float = 60.0,
) -> Any:
    provider_name = str(provider).strip().lower()
    if provider_name in {"", "ollama"}:
        return wrap_chat_client(
            OllamaChatClient(
                model=model or "qwen2.5-coder:14b",
                base_url=base_url or os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434"),
                timeout_seconds=timeout_seconds,
            ),
            enabled=circuit_breaker_enabled,
            failure_threshold=circuit_failure_threshold,
            reset_timeout=circuit_reset_timeout,
        )
    if provider_name == "deepseek":
        resolved_api_key = str(api_key or "").strip() or os.getenv("DEEPSEEK_API_KEY", "").strip() or os.getenv("LLM_API_KEY", "").strip()
        if not resolved_api_key:
            raise RuntimeError("DEEPSEEK_API_KEY or LLM_API_KEY is required for deepseek bootstrap mode.")
        return wrap_chat_client(
            DeepSeekChatClient(
                api_key=resolved_api_key,
                model=model or "deepseek-chat",
                base_url=base_url or "https://api.deepseek.com",
                timeout_seconds=timeout_seconds,
            ),
            enabled=circuit_breaker_enabled,
            failure_threshold=circuit_failure_threshold,
            reset_timeout=circuit_reset_timeout,
        )
    if provider_name in {"openai", "openai-compatible", "openai_compatible"}:
        resolved_api_key = str(api_key or "").strip() or os.getenv("OPENAI_API_KEY", "").strip() or os.getenv("LLM_API_KEY", "").strip()
        if not resolved_api_key:
            raise RuntimeError("OPENAI_API_KEY or LLM_API_KEY is required for openai bootstrap mode.")
        return wrap_chat_client(
            OpenAIChatClient(
                api_key=resolved_api_key,
                model=model or "gpt-4o-mini",
                base_url=base_url or "https://api.openai.com/v1",
                timeout_seconds=timeout_seconds,
            ),
            enabled=circuit_breaker_enabled,
            failure_threshold=circuit_failure_threshold,
            reset_timeout=circuit_reset_timeout,
        )
    raise ValueError(f"Unsupported llm provider for runtime: {provider}")

def _bootstrap(
    *,
    args: argparse.Namespace,
    config: dict[str, Any],
    paths: RuntimePaths,
    storage: RuntimeStorage,
    domain_brief_override: str | None = None,
    force_rebuild: bool = False,
    load_resume_state: bool = True,
    load_resume_world: bool = True,
    knowledge_graph: KnowledgeGraph | None = None,
    forecast_registry: ForecastRegistry | None = None,
) -> BootstrapResult:
    llm_cfg = config.get("llm", {})
    agent_cfg = config.get("agent", {})
    bootstrap_cfg = agent_cfg.get("bootstrap", {})
    provider = str(llm_cfg.get("provider", "ollama") or "ollama")
    requested_model = str(args.model or llm_cfg.get("model", "auto"))
    ollama_base_url = str(
        args.ollama_base_url
        or llm_cfg.get("base_url")
        or os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434")
    )
    model_name = (
        _select_ollama_model(ollama_base_url, requested_model)
        if provider.strip().lower() in {"", "ollama"}
        else requested_model
    )
    llm_client = _build_chat_client(
        provider=provider,
        model=model_name,
        base_url=ollama_base_url,
        timeout_seconds=float(llm_cfg.get("timeout_seconds", 120.0)),
        api_key=str(llm_cfg.get("api_key", "") or ""),
        circuit_breaker_enabled=bool(llm_cfg.get("circuit_breaker", {}).get("enabled", True)),
        circuit_failure_threshold=int(llm_cfg.get("circuit_breaker", {}).get("failure_threshold", 3)),
        circuit_reset_timeout=float(llm_cfg.get("circuit_breaker", {}).get("reset_timeout_seconds", 60.0)),
    )
    LOGGER.info("Using llm provider=%s model=%s base_url=%s", provider, model_name, ollama_base_url)
    LOGGER.info("Knowledge graph path=%s runtime path=%s", paths.kg_path, paths.runtime_path)

    forecast_registry = forecast_registry or ForecastRegistry(
        json_path=paths.runtime_path / "forecasts.json",
        auto_load=bool(args.resume),
        auto_save=True,
    )
    vectorstore = build_vectorstore(config, config_path=paths.config_path)
    embedding_adapter = None
    if vectorstore is not None or str(config.get("memory", {}).get("embedding_provider", "")).strip():
        embedding_adapter, _embedding_backend = build_embedding_adapter(config)
    knowledge_graph = knowledge_graph or build_knowledge_graph(
        config,
        config_path=paths.config_path,
        embedding_adapter=embedding_adapter,
        vectorstore=vectorstore,
        auto_load=True,
        auto_save=True,
    )
    pipeline = AnalysisPipeline(
        knowledge_graph=knowledge_graph,
        sim_config=_build_sim_config(config),
        forecast_registry=forecast_registry,
        config_path=paths.config_path,
    )

    if load_resume_state and args.resume and (paths.runtime_path / "checkpoint.json").exists():
        loaded_state = storage.checkpoint_manager.load(paths.runtime_path / "checkpoint.json")
        pipeline.conscious_state = ConsciousState.from_dict(loaded_state.to_dict(), knowledge_graph)
        LOGGER.info("Loaded consciousness checkpoint.")
    if storage.pending_signals:
        LOGGER.info("Loaded pending queue length=%d", len(storage.pending_signals))

    current_world: WorldState | None = None
    world_state_path = paths.runtime_path / "world_state.json"
    if load_resume_world and args.resume and world_state_path.exists():
        current_world = WorldState.from_snapshot(json.loads(world_state_path.read_text(encoding="utf-8")))
        LOGGER.info(
            "Loaded world checkpoint domain_id=%s step=%s runtime_step=%s",
            current_world.domain_id,
            current_world.t,
            current_world.runtime_step,
        )

    configured_bootstrap_mode = str(args.bootstrap_mode or bootstrap_cfg.get("mode") or "").strip().lower()
    package_path = paths.runtime_path / "bootstrap_package.json"
    domain_brief_path = (
        _resolve_path(paths.config_base, args.domain_brief_path, args.domain_brief_path)
        if args.domain_brief_path
        else (
            _resolve_path(paths.config_base, bootstrap_cfg.get("domain_brief_path"), bootstrap_cfg.get("domain_brief_path"))
            if bootstrap_cfg.get("domain_brief_path")
            else None
        )
    )
    domain_brief_inline = str(bootstrap_cfg.get("domain_brief", "")).strip()
    if configured_bootstrap_mode:
        bootstrap_mode = configured_bootstrap_mode
    elif domain_brief_inline or domain_brief_path is not None:
        bootstrap_mode = "llm_synthesize"
    elif paths.schema_path is not None:
        bootstrap_mode = "schema_path"
    else:
        bootstrap_mode = "llm_synthesize"
    synthesized_package = None if force_rebuild else (json.loads(package_path.read_text(encoding="utf-8")) if package_path.exists() else None)

    fallback_candidate = bootstrap_cfg.get("fallback_schema_path")

    if synthesized_package is None and bootstrap_mode == "schema_path":
        schema_path = paths.schema_path
        if schema_path is None:
            candidate = bootstrap_cfg.get("schema_path")
            if candidate:
                schema_path = _resolve_path(paths.config_base, candidate, candidate)
        if schema_path is None or not schema_path.exists():
            raise FileNotFoundError("bootstrap.mode=schema_path requires an existing schema_path.")
        synthesized_package = {
            "schema": json.loads(schema_path.read_text(encoding="utf-8")),
            "policies": [],
            "assumptions": [],
            "bootstrap_mode": "schema_path",
            "bootstrap_contract": build_bootstrap_contract(
                bootstrap_mode="schema_path",
                llm_provider=provider,
                model_name=model_name,
                has_fallback_schema=bool(fallback_candidate),
                actual_bootstrap_path="schema_seed",
                schema_path=str(schema_path),
                fallback_schema_path=str(_resolve_path(paths.config_base, fallback_candidate, fallback_candidate))
                if fallback_candidate
                else None,
                domain_brief_supplied=bool(domain_brief_inline or domain_brief_path is not None),
            ),
        }
    elif synthesized_package is None and bootstrap_mode == "llm_synthesize":
        domain_brief = str(domain_brief_override or domain_brief_inline or _read_optional_text(domain_brief_path)).strip()
        if not domain_brief:
            raise RuntimeError("bootstrap.mode=llm_synthesize requires agent.bootstrap.domain_brief or domain_brief_path.")
        if not hasattr(llm_client, "repair_schema"):
            raise RuntimeError(
                f"Configured llm provider '{provider}' does not expose repair_schema(); "
                "use a local or remote chat client that implements repair_schema()."
            )
        LOGGER.info("Synthesizing bootstrap schema from domain brief.")
        orchestrator = FreemanOrchestrator(
            llm_client,
            package_normalization=bootstrap_cfg.get("package_normalization", "auto"),
        )
        try:
            package, world_id, attempts, repair_history = orchestrator.compile_and_repair(
                domain_brief,
                max_retries=int(bootstrap_cfg.get("max_retries", 10)),
                trial_steps=int(bootstrap_cfg.get("trial_steps", 3)),
                config=_build_sim_config(config),
                etl_bootstrap=True,
            )
            synthesized_package = {
                **package,
                "bootstrap_mode": "llm_synthesize",
                "world_id": world_id,
                "synthesis_attempts": attempts,
                "bootstrap_attempts": json.loads(
                    json.dumps(orchestrator.last_bootstrap_attempts or repair_history, ensure_ascii=False)
                ),
                "repair_history": repair_history,
                "domain_brief": domain_brief,
                "bootstrap_contract": build_bootstrap_contract(
                    bootstrap_mode="llm_synthesize",
                    llm_provider=provider,
                    model_name=model_name,
                    has_fallback_schema=bool(fallback_candidate),
                    actual_bootstrap_path="etl_from_brief",
                    schema_path=None,
                    fallback_schema_path=str(_resolve_path(paths.config_base, fallback_candidate, fallback_candidate))
                    if fallback_candidate
                    else None,
                    domain_brief_supplied=bool(domain_brief),
                ),
            }
            _atomic_write_json(package_path, synthesized_package)
            LOGGER.info("Synthesized bootstrap package attempts=%d saved=%s", attempts, package_path)
        except Exception as exc:  # noqa: BLE001
            if not fallback_candidate:
                raise
            LOGGER.warning("LLM bootstrap failed: %s. Falling back to schema_path=%s", exc, fallback_candidate)
            schema_path = _resolve_path(paths.config_base, fallback_candidate, fallback_candidate)
            synthesized_package = {
                "schema": json.loads(schema_path.read_text(encoding="utf-8")),
                "policies": [],
                "assumptions": [],
                "bootstrap_mode": "llm_synthesize_fallback",
                "domain_brief": domain_brief,
                "bootstrap_error": str(exc),
                "bootstrap_attempts": json.loads(
                    json.dumps(getattr(orchestrator, "last_bootstrap_attempts", []), ensure_ascii=False)
                ),
                "fallback_schema_path": str(schema_path),
                "bootstrap_contract": build_bootstrap_contract(
                    bootstrap_mode="llm_synthesize",
                    llm_provider=provider,
                    model_name=model_name,
                    has_fallback_schema=True,
                    actual_bootstrap_path="fallback_schema_seed",
                    schema_path=None,
                    fallback_schema_path=str(schema_path),
                    domain_brief_supplied=bool(domain_brief),
                ),
            }
            _atomic_write_json(package_path, synthesized_package)
    elif synthesized_package is None:
        raise ValueError(f"Unsupported bootstrap mode: {bootstrap_mode}")

    if "bootstrap_contract" not in synthesized_package:
        effective_mode = str(synthesized_package.get("bootstrap_mode") or bootstrap_mode or "").strip().lower()
        actual_bootstrap_path = "etl_from_brief"
        schema_path_for_contract: str | None = None
        if effective_mode == "schema_path":
            actual_bootstrap_path = "schema_seed"
            schema_candidate = paths.schema_path or bootstrap_cfg.get("schema_path")
            if schema_candidate:
                schema_path_for_contract = str(_resolve_path(paths.config_base, schema_candidate, schema_candidate))
        elif effective_mode == "llm_synthesize_fallback":
            actual_bootstrap_path = "fallback_schema_seed"
        synthesized_package["bootstrap_contract"] = build_bootstrap_contract(
            bootstrap_mode="schema_path" if effective_mode == "schema_path" else "llm_synthesize",
            llm_provider=provider,
            model_name=model_name,
            has_fallback_schema=bool(fallback_candidate or synthesized_package.get("fallback_schema_path")),
            actual_bootstrap_path=actual_bootstrap_path,
            schema_path=schema_path_for_contract,
            fallback_schema_path=str(synthesized_package.get("fallback_schema_path") or "")
            or (
                str(_resolve_path(paths.config_base, fallback_candidate, fallback_candidate))
                if fallback_candidate
                else None
            ),
            domain_brief_supplied=bool(
                synthesized_package.get("domain_brief") or domain_brief_inline or domain_brief_path is not None
            ),
        )
        _atomic_write_json(package_path, synthesized_package)

    schema_payload = dict(synthesized_package["schema"])
    base_world_template = pipeline.compiler.compile(schema_payload)
    if current_world is None:
        bootstrap_result = pipeline.run(schema_payload, policies=list(synthesized_package.get("policies", [])))
        current_world = bootstrap_result.world.clone()
        current_world.runtime_step = 0
        _append_unlogged_trace_events(pipeline.conscious_state, storage.event_log, storage.logged_event_ids)
        _persist_runtime_state(
            pipeline=pipeline,
            world_state=current_world,
            runtime_path=paths.runtime_path,
            checkpoint_manager=storage.checkpoint_manager,
            cursor_store=storage.cursor_store,
            signal_memory=storage.signal_memory,
            pending_signals=storage.pending_signals,
        )
        LOGGER.info(
            "Bootstrap completed mode=%s domain_id=%s dominant_outcome=%s",
            bootstrap_mode,
            bootstrap_result.world.domain_id,
            bootstrap_result.dominant_outcome,
        )

    return BootstrapResult(
        pipeline=pipeline,
        current_world=current_world,
        base_world_template=base_world_template,
        llm_client=llm_client,
        estimator=ParameterEstimator(
            llm_client,
            epistemic_log=pipeline.epistemic_log,
            belief_conflict_log=pipeline.belief_conflict_log,
        ),
        bootstrap_mode=bootstrap_mode,
        provider=provider,
        model_name=model_name,
        package_path=package_path,
    )



__all__ = [
    "FreemanOrchestrator",
    "_bootstrap",
    "_build_chat_client",
    "_select_ollama_model",
]
