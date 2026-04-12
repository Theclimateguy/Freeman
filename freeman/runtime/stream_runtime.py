"""Run Freeman against configurable signal streams with checkpoint/resume."""

from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
import json
import logging
import os
from pathlib import Path
import signal
import sys
import time
from typing import Any
from urllib.error import URLError
from urllib.request import urlopen

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    from freeman_connectors import build_signal_source
except ImportError:
    connectors_root = REPO_ROOT / "packages" / "freeman-connectors"
    if str(connectors_root) not in sys.path:
        sys.path.insert(0, str(connectors_root))
    from freeman_connectors import build_signal_source

from freeman.agent.analysispipeline import AnalysisPipeline
from freeman.agent.consciousness import ConsciousState, ConsciousnessEngine, TraceEvent
from freeman.agent.forecastregistry import ForecastRegistry
from freeman.agent.parameterestimator import ParameterEstimator
from freeman.agent.signalingestion import ManualSignalSource, Signal, SignalIngestionEngine, SignalMemory
from freeman.game.runner import SimConfig
from freeman.llm import DeepSeekChatClient, OllamaChatClient, OpenAIChatClient
from freeman.llm.orchestrator import DeepSeekFreemanOrchestrator
from freeman.memory.knowledgegraph import KnowledgeGraph
from freeman.core.world import WorldState
from freeman.runtime.checkpoint import CheckpointManager
from freeman.runtime.event_log import EventLog
from freeman.runtime.stream import StreamCursorStore

LOGGER = logging.getLogger("stream_runtime")


def _utc_now() -> datetime:
    return datetime.now(timezone.utc).replace(microsecond=0)


def _to_datetime(value: str | datetime) -> datetime:
    if isinstance(value, datetime):
        dt = value
    else:
        dt = datetime.fromisoformat(str(value))
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc).replace(microsecond=0)


def _resolve_path(base: Path, candidate: str | None, default: str) -> Path:
    path = Path(candidate or default)
    return path if path.is_absolute() else (base / path).resolve()


def _atomic_write_json(path: Path, payload: Any) -> None:
    target = Path(path).resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(f"{target.suffix}.tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    tmp.replace(target)


def _load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def _read_optional_text(path: str | Path | None) -> str:
    if path is None:
        return ""
    target = Path(path).resolve()
    if not target.exists():
        return ""
    return target.read_text(encoding="utf-8").strip()


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
) -> Any:
    provider_name = str(provider).strip().lower()
    if provider_name in {"", "ollama"}:
        return OllamaChatClient(
            model=model or "qwen2.5-coder:14b",
            base_url=base_url or os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434"),
            timeout_seconds=timeout_seconds,
        )
    if provider_name == "deepseek":
        api_key = os.getenv("DEEPSEEK_API_KEY", "").strip() or os.getenv("LLM_API_KEY", "").strip()
        if not api_key:
            raise RuntimeError("DEEPSEEK_API_KEY or LLM_API_KEY is required for deepseek bootstrap mode.")
        return DeepSeekChatClient(
            api_key=api_key,
            model=model or "deepseek-chat",
            base_url=base_url or "https://api.deepseek.com",
            timeout_seconds=timeout_seconds,
        )
    if provider_name == "openai":
        api_key = os.getenv("OPENAI_API_KEY", "").strip() or os.getenv("LLM_API_KEY", "").strip()
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY or LLM_API_KEY is required for openai bootstrap mode.")
        return OpenAIChatClient(
            api_key=api_key,
            model=model or "gpt-4o-mini",
            base_url=base_url or "https://api.openai.com/v1",
            timeout_seconds=timeout_seconds,
        )
    raise ValueError(f"Unsupported llm provider for runtime: {provider}")


def _build_sim_config(config: dict[str, Any]) -> SimConfig:
    sim_cfg = config.get("sim", {})
    return SimConfig(
        max_steps=int(sim_cfg.get("max_steps", 50)),
        dt=float(sim_cfg.get("dt", 1.0)),
        level2_check_every=int(sim_cfg.get("level2_check_every", 5)),
        level2_shock_delta=float(sim_cfg.get("level2_shock_delta", 0.01)),
        stop_on_hard_level2=bool(sim_cfg.get("stop_on_hard_level2", True)),
        convergence_check_steps=int(sim_cfg.get("convergence_check_steps", 20)),
        convergence_epsilon=float(sim_cfg.get("convergence_epsilon", 1.0e-4)),
        fixed_point_max_iter=int(sim_cfg.get("fixed_point_max_iter", 20)),
        fixed_point_alpha=float(sim_cfg.get("fixed_point_alpha", 0.1)),
        seed=int(sim_cfg.get("seed", 42)),
    )


def _keywords_from_config(config: dict[str, Any], default_keywords: list[str] | None = None) -> list[str]:
    agent_cfg = config.get("agent", {})
    raw = (
        agent_cfg.get("stream_keywords")
        or agent_cfg.get("keywords")
        or agent_cfg.get("climate_keywords")
        or default_keywords
        or []
    )
    return [str(item).strip().lower() for item in raw if str(item).strip()]


def _signal_matches_keywords(signal_payload: Signal, keywords: list[str]) -> bool:
    if not keywords:
        return True
    haystack = " ".join(
        [
            signal_payload.topic,
            signal_payload.text,
            " ".join(signal_payload.entities),
            json.dumps(signal_payload.metadata, ensure_ascii=False),
        ]
    ).lower()
    return any(keyword in haystack for keyword in keywords)


def _source_configs(config: dict[str, Any], default_sources: list[dict[str, Any]] | None = None) -> list[dict[str, Any]]:
    configured = config.get("agent", {}).get("sources", [])
    if isinstance(configured, list) and configured:
        return [dict(item) for item in configured if isinstance(item, dict)]
    return [dict(item) for item in (default_sources or [])]


def _append_unlogged_trace_events(
    state: ConsciousState,
    event_log: EventLog,
    logged_event_ids: set[str],
) -> int:
    appended = 0
    for event in state.trace_state:
        if event.event_id in logged_event_ids:
            continue
        event_log.append(event)
        logged_event_ids.add(event.event_id)
        appended += 1
    return appended


def _load_logged_ids_and_backfill_cursor(event_log: EventLog, cursor_store: StreamCursorStore) -> set[str]:
    logged: set[str] = set()
    for event in event_log.slice_from(""):
        logged.add(event.event_id)
        signal_id = event.diff.get("signal_id") if isinstance(event.diff, dict) else None
        if signal_id is not None:
            cursor_store.commit(str(signal_id))
    return logged


def _load_signal_memory(path: Path) -> SignalMemory:
    memory = SignalMemory()
    source = Path(path).resolve()
    if source.exists():
        payload = json.loads(source.read_text(encoding="utf-8"))
        memory.load_snapshot(payload.get("records", []))
    return memory


def _save_signal_memory(memory: SignalMemory, path: Path) -> None:
    _atomic_write_json(path, {"records": memory.snapshot()})


def _signal_to_dict(signal_payload: Signal) -> dict[str, Any]:
    return {
        "signal_id": str(signal_payload.signal_id),
        "source_type": str(signal_payload.source_type),
        "text": str(signal_payload.text),
        "topic": str(signal_payload.topic),
        "entities": list(signal_payload.entities),
        "sentiment": float(signal_payload.sentiment),
        "timestamp": str(signal_payload.timestamp),
        "metadata": dict(signal_payload.metadata),
    }


def _signal_from_dict(payload: dict[str, Any]) -> Signal:
    return Signal(
        signal_id=str(payload["signal_id"]),
        source_type=str(payload["source_type"]),
        text=str(payload["text"]),
        topic=str(payload["topic"]),
        entities=[str(value) for value in payload.get("entities", [])],
        sentiment=float(payload.get("sentiment", 0.0)),
        timestamp=str(payload.get("timestamp", _utc_now().isoformat())),
        metadata=dict(payload.get("metadata", {})),
    )


def _load_pending_queue(path: Path) -> list[Signal]:
    source = Path(path).resolve()
    if not source.exists():
        return []
    payload = json.loads(source.read_text(encoding="utf-8"))
    return [_signal_from_dict(item) for item in payload.get("signals", [])]


def _save_pending_queue(signals: list[Signal], path: Path) -> None:
    _atomic_write_json(path, {"signals": [_signal_to_dict(item) for item in signals]})


def _persist_runtime_state(
    *,
    pipeline: AnalysisPipeline,
    world_state: WorldState,
    runtime_path: Path,
    checkpoint_manager: CheckpointManager,
    cursor_store: StreamCursorStore,
    signal_memory: SignalMemory,
    pending_signals: list[Signal],
) -> None:
    runtime_path.mkdir(parents=True, exist_ok=True)
    pipeline.knowledge_graph.save()
    _atomic_write_json(runtime_path / "world_state.json", world_state.snapshot())
    _save_signal_memory(signal_memory, runtime_path / "signal_memory.json")
    _save_pending_queue(pending_signals, runtime_path / "pending_signals.json")
    checkpoint_manager.save(pipeline.conscious_state, runtime_path / "checkpoint.json")
    cursor_store.save(runtime_path / "cursors.json")


def _runtime_trace_for_signal(
    *,
    state: ConsciousState,
    signal_payload: Signal,
    trigger_mode: str,
    llm_used: bool,
    updated_world: bool,
    update_error: str | None = None,
) -> TraceEvent:
    timestamp = _to_datetime(signal_payload.timestamp)
    index = len(state.trace_state)
    signal_id = str(signal_payload.signal_id)
    diff = {
        "signal_id": signal_id,
        "source_type": str(signal_payload.source_type),
        "topic": str(signal_payload.topic),
        "mode": trigger_mode,
        "llm_used": bool(llm_used),
        "world_updated": bool(updated_world),
    }
    if update_error:
        diff["update_error"] = str(update_error)
    return TraceEvent(
        event_id=f"trace:signal:{signal_id}",
        timestamp=timestamp,
        transition_type="external",
        trigger_type="signal",
        operator="runtime_signal_ingest",
        pre_state_ref=f"state:{index}",
        post_state_ref=f"state:{index + 1}",
        input_refs=[f"signal:{signal_id}"],
        diff=diff,
        rationale=f"signal processed in mode={trigger_mode}",
    )


def _runtime_trace_for_verification(
    *,
    state: ConsciousState,
    verified_count: int,
    verified_ids: list[str],
    mean_abs_error: float | None,
) -> TraceEvent:
    index = len(state.trace_state)
    diff: dict[str, Any] = {
        "verified_count": int(verified_count),
        "verified_ids": list(verified_ids),
    }
    if mean_abs_error is not None:
        diff["mean_abs_error"] = float(mean_abs_error)
    return TraceEvent(
        event_id=f"trace:verify:{_utc_now().isoformat()}",
        timestamp=_utc_now(),
        transition_type="external",
        trigger_type="manual",
        operator="runtime_forecast_verify",
        pre_state_ref=f"state:{index}",
        post_state_ref=f"state:{index + 1}",
        input_refs=[f"forecast:{item}" for item in verified_ids],
        diff=diff,
        rationale=f"verified {verified_count} due forecasts",
    )


def _build_parser(*, default_config_path: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run Freeman on configurable signal streams.")
    parser.add_argument("--config-path", default=default_config_path)
    parser.add_argument("--schema-path", default=None)
    parser.add_argument("--bootstrap-mode", choices=["schema_path", "llm_synthesize"], default=None)
    parser.add_argument("--domain-brief-path", default=None)
    parser.add_argument("--hours", type=float, default=8.0)
    parser.add_argument("--poll-seconds", type=float, default=None)
    parser.add_argument("--analysis-interval-seconds", type=float, default=1.0)
    parser.add_argument("--model", default="auto")
    parser.add_argument("--ollama-base-url", default=None)
    parser.add_argument("--max-signals-per-poll", type=int, default=30)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--include-watch", action="store_true")
    parser.add_argument("--keyword", action="append", default=None)
    parser.add_argument("--log-level", default="INFO")
    return parser


def main(
    argv: list[str] | None = None,
    *,
    default_config_path: str = "config.yaml",
    default_sources: list[dict[str, Any]] | None = None,
    default_keywords: list[str] | None = None,
) -> int:
    args = _build_parser(default_config_path=default_config_path).parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, str(args.log_level).upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    config_path = Path(args.config_path).resolve()
    config = _load_yaml(config_path)
    config_base = config_path.parent
    runtime_cfg = config.get("runtime", {})
    memory_cfg = config.get("memory", {})
    llm_cfg = config.get("llm", {})
    agent_cfg = config.get("agent", {})
    bootstrap_cfg = agent_cfg.get("bootstrap", {})

    runtime_path = _resolve_path(config_base, runtime_cfg.get("runtime_path"), "./data/runtime")
    event_log_path = _resolve_path(config_base, runtime_cfg.get("event_log_path"), str(runtime_path / "event_log.jsonl"))
    kg_path = _resolve_path(config_base, memory_cfg.get("json_path"), "./data/kg_state.json")
    schema_path_arg = getattr(args, "schema_path", None)
    schema_path = _resolve_path(config_base, schema_path_arg, schema_path_arg) if schema_path_arg else None

    poll_seconds = float(
        args.poll_seconds
        if args.poll_seconds is not None
        else agent_cfg.get("source_refresh_seconds", runtime_cfg.get("poll_interval_seconds", 300))
    )
    keywords = (
        [str(item).strip().lower() for item in args.keyword if str(item).strip()]
        if args.keyword
        else _keywords_from_config(config, default_keywords)
    )

    ollama_base_url = str(
        args.ollama_base_url
        or llm_cfg.get("base_url")
        or os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434")
    )
    provider = str(llm_cfg.get("provider", "ollama") or "ollama")
    requested_model = str(args.model or llm_cfg.get("model", "auto"))
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
    )

    LOGGER.info("Using llm provider=%s model=%s base_url=%s", provider, model_name, ollama_base_url)
    LOGGER.info("Knowledge graph path=%s runtime path=%s", kg_path, runtime_path)

    runtime_path.mkdir(parents=True, exist_ok=True)
    forecast_registry = ForecastRegistry(
        json_path=runtime_path / "forecasts.json",
        auto_load=bool(args.resume),
        auto_save=True,
    )
    knowledge_graph = KnowledgeGraph(json_path=kg_path, auto_load=True, auto_save=True)
    pipeline = AnalysisPipeline(
        knowledge_graph=knowledge_graph,
        sim_config=_build_sim_config(config),
        forecast_registry=forecast_registry,
        config_path=config_path,
    )

    checkpoint_manager = CheckpointManager()
    cursor_store = StreamCursorStore()
    if args.resume:
        cursor_store.load(runtime_path / "cursors.json")
    event_log = EventLog(event_log_path)
    logged_event_ids = _load_logged_ids_and_backfill_cursor(event_log, cursor_store)
    signal_memory = _load_signal_memory(runtime_path / "signal_memory.json") if args.resume else SignalMemory()
    pending_signals = _load_pending_queue(runtime_path / "pending_signals.json") if args.resume else []
    queued_signal_ids = {str(item.signal_id) for item in pending_signals}

    if args.resume and (runtime_path / "checkpoint.json").exists():
        loaded_state = checkpoint_manager.load(runtime_path / "checkpoint.json")
        pipeline.conscious_state = ConsciousState.from_dict(loaded_state.to_dict(), knowledge_graph)
        LOGGER.info("Loaded consciousness checkpoint.")
    if pending_signals:
        LOGGER.info("Loaded pending queue length=%d", len(pending_signals))

    current_world: WorldState | None = None
    world_state_path = runtime_path / "world_state.json"
    if args.resume and world_state_path.exists():
        current_world = WorldState.from_snapshot(json.loads(world_state_path.read_text(encoding="utf-8")))
        LOGGER.info("Loaded world checkpoint domain_id=%s step=%s", current_world.domain_id, current_world.t)
    bootstrap_mode = str(
        args.bootstrap_mode
        or bootstrap_cfg.get("mode")
        or ("schema_path" if schema_path is not None else "llm_synthesize")
    ).strip().lower()
    package_path = runtime_path / "bootstrap_package.json"
    domain_brief_path = (
        _resolve_path(config_base, args.domain_brief_path, args.domain_brief_path)
        if args.domain_brief_path
        else (
            _resolve_path(config_base, bootstrap_cfg.get("domain_brief_path"), bootstrap_cfg.get("domain_brief_path"))
            if bootstrap_cfg.get("domain_brief_path")
            else None
        )
    )
    domain_brief_inline = str(bootstrap_cfg.get("domain_brief", "")).strip()
    synthesized_package: dict[str, Any] | None = None

    if package_path.exists():
        synthesized_package = json.loads(package_path.read_text(encoding="utf-8"))

    if synthesized_package is None and bootstrap_mode == "schema_path":
        if schema_path is None:
            candidate = bootstrap_cfg.get("schema_path")
            if candidate:
                schema_path = _resolve_path(config_base, candidate, candidate)
        if schema_path is None or not schema_path.exists():
            raise FileNotFoundError("bootstrap.mode=schema_path requires an existing schema_path.")
        schema_payload = json.loads(schema_path.read_text(encoding="utf-8"))
        synthesized_package = {
            "schema": schema_payload,
            "policies": [],
            "assumptions": [],
            "bootstrap_mode": "schema_path",
        }
    elif synthesized_package is None and bootstrap_mode == "llm_synthesize":
        domain_brief = domain_brief_inline or _read_optional_text(domain_brief_path)
        if not domain_brief:
            raise RuntimeError("bootstrap.mode=llm_synthesize requires agent.bootstrap.domain_brief or domain_brief_path.")
        if not hasattr(llm_client, "repair_schema"):
            raise RuntimeError(
                f"Configured llm provider '{provider}' does not expose repair_schema(); "
                "use ollama or deepseek for llm_synthesize bootstrap."
        )
        LOGGER.info("Synthesizing bootstrap schema from domain brief.")
        orchestrator = DeepSeekFreemanOrchestrator(llm_client)
        try:
            package, world_id, attempts, repair_history = orchestrator.compile_and_repair(
                domain_brief,
                max_retries=int(bootstrap_cfg.get("max_retries", 5)),
                trial_steps=int(bootstrap_cfg.get("trial_steps", 3)),
                config=_build_sim_config(config),
            )
            synthesized_package = {
                **package,
                "bootstrap_mode": "llm_synthesize",
                "world_id": world_id,
                "synthesis_attempts": attempts,
                "repair_history": repair_history,
                "domain_brief": domain_brief,
            }
            _atomic_write_json(package_path, synthesized_package)
            LOGGER.info("Synthesized bootstrap package attempts=%d saved=%s", attempts, package_path)
        except Exception as exc:  # noqa: BLE001
            fallback_candidate = bootstrap_cfg.get("fallback_schema_path")
            if not fallback_candidate:
                raise
            LOGGER.warning("LLM bootstrap failed: %s. Falling back to schema_path=%s", exc, fallback_candidate)
            schema_path = _resolve_path(config_base, fallback_candidate, fallback_candidate)
            schema_payload = json.loads(schema_path.read_text(encoding="utf-8"))
            synthesized_package = {
                "schema": schema_payload,
                "policies": [],
                "assumptions": [],
                "bootstrap_mode": "llm_synthesize_fallback",
                "domain_brief": domain_brief,
                "bootstrap_error": str(exc),
                "fallback_schema_path": str(schema_path),
            }
            _atomic_write_json(package_path, synthesized_package)
    elif synthesized_package is None:
        raise ValueError(f"Unsupported bootstrap mode: {bootstrap_mode}")

    schema_payload = dict(synthesized_package["schema"])
    bootstrap_policies = list(synthesized_package.get("policies", []))
    base_world_template = pipeline.compiler.compile(schema_payload)

    if current_world is None:
        bootstrap_result = pipeline.run(schema_payload, policies=bootstrap_policies)
        current_world = bootstrap_result.world.clone()
        _append_unlogged_trace_events(pipeline.conscious_state, event_log, logged_event_ids)
        _persist_runtime_state(
            pipeline=pipeline,
            world_state=current_world,
            runtime_path=runtime_path,
            checkpoint_manager=checkpoint_manager,
            cursor_store=cursor_store,
            signal_memory=signal_memory,
            pending_signals=pending_signals,
        )
        LOGGER.info(
            "Bootstrap completed mode=%s domain_id=%s dominant_outcome=%s",
            bootstrap_mode,
            bootstrap_result.world.domain_id,
            bootstrap_result.dominant_outcome,
        )

    sources = [build_signal_source(cfg) for cfg in _source_configs(config, default_sources)]
    LOGGER.info("Configured source count=%d", len(sources))

    estimator = ParameterEstimator(
        llm_client,
        epistemic_log=pipeline.epistemic_log,
        belief_conflict_log=pipeline.belief_conflict_log,
    )
    ingestion_engine = SignalIngestionEngine()

    stop_requested = False

    def _request_stop(signum, frame):  # noqa: ANN001
        del signum, frame
        nonlocal stop_requested
        stop_requested = True

    previous_sigint = signal.getsignal(signal.SIGINT)
    previous_sigterm = signal.getsignal(signal.SIGTERM)
    signal.signal(signal.SIGINT, _request_stop)
    signal.signal(signal.SIGTERM, _request_stop)

    started_at = _utc_now()
    deadline = None if float(args.hours) <= 0.0 else started_at + timedelta(hours=float(args.hours))
    next_poll_at = started_at
    analysis_interval_seconds = max(float(args.analysis_interval_seconds), 0.1)
    processed_count = 0
    updated_count = 0
    update_failures = 0
    verified_forecasts_count = 0
    idle_deliberations = 0
    skipped_watch_count = 0
    seen_count = 0

    try:
        while not stop_requested:
            now = _utc_now()
            if deadline is not None and now >= deadline:
                LOGGER.info("Reached duration limit (hours=%s).", args.hours)
                break

            if now >= next_poll_at:
                fetched: list[Signal] = []
                for source in sources:
                    try:
                        source_signals = source.fetch()
                        fetched.extend(source_signals)
                    except Exception as exc:  # noqa: BLE001
                        LOGGER.warning("Source fetch failed (%s): %s", getattr(source, "url", "unknown"), exc)
                if keywords:
                    fetched = [item for item in fetched if _signal_matches_keywords(item, keywords)]
                fetched.sort(key=lambda item: item.timestamp)

                enqueued = 0
                for signal_payload in fetched:
                    if enqueued >= int(args.max_signals_per_poll):
                        break
                    signal_id = str(signal_payload.signal_id)
                    if cursor_store.is_committed(signal_id):
                        continue
                    if signal_id in queued_signal_ids:
                        continue
                    pending_signals.append(signal_payload)
                    queued_signal_ids.add(signal_id)
                    seen_count += 1
                    enqueued += 1
                next_poll_at = now + timedelta(seconds=float(poll_seconds))
                if enqueued > 0:
                    LOGGER.info("Enqueued %d new signals. queue_len=%d", enqueued, len(pending_signals))
                    _persist_runtime_state(
                        pipeline=pipeline,
                        world_state=current_world,
                        runtime_path=runtime_path,
                        checkpoint_manager=checkpoint_manager,
                        cursor_store=cursor_store,
                        signal_memory=signal_memory,
                        pending_signals=pending_signals,
                    )
                else:
                    LOGGER.info("No new eligible signals this poll.")

            if pending_signals:
                signal_payload = pending_signals.pop(0)
                signal_id = str(signal_payload.signal_id)
                queued_signal_ids.discard(signal_id)
                if cursor_store.is_committed(signal_id):
                    continue
                triggers = ingestion_engine.ingest(
                    ManualSignalSource([signal_payload]),
                    classifier=llm_client,
                    signal_memory=signal_memory,
                    skip_duplicates_within_hours=1.0,
                )
                if not triggers:
                    runtime_event = _runtime_trace_for_signal(
                        state=pipeline.conscious_state,
                        signal_payload=signal_payload,
                        trigger_mode="WATCH",
                        llm_used=False,
                        updated_world=False,
                        update_error=None,
                    )
                    pipeline.conscious_state.trace_state.append(runtime_event)
                    event_log.append(runtime_event)
                    logged_event_ids.add(runtime_event.event_id)
                    cursor_store.commit(signal_id)
                    processed_count += 1
                    _persist_runtime_state(
                        pipeline=pipeline,
                        world_state=current_world,
                        runtime_path=runtime_path,
                        checkpoint_manager=checkpoint_manager,
                        cursor_store=cursor_store,
                        signal_memory=signal_memory,
                        pending_signals=pending_signals,
                    )
                    continue

                trigger = triggers[0]
                should_update = trigger.mode in {"ANALYZE", "DEEP_DIVE"}
                llm_update_attempted = should_update
                update_error: str | None = None
                result = None
                if should_update:
                    try:
                        parameter_vector = estimator.estimate(current_world, signal_payload.text)
                        try:
                            result = pipeline.update(
                                current_world,
                                parameter_vector,
                                signal_text=signal_payload.text,
                            )
                            current_world = result.world.clone()
                            updated_count += 1
                        except Exception as primary_exc:  # noqa: BLE001
                            LOGGER.warning(
                                "Primary update failed for signal_id=%s: %s; retrying from base world.",
                                signal_id,
                                primary_exc,
                            )
                            result = pipeline.update(
                                base_world_template.clone(),
                                parameter_vector,
                                signal_text=signal_payload.text,
                            )
                            current_world = result.world.clone()
                            updated_count += 1
                            update_error = f"primary_update_failed: {primary_exc}; fallback=base_world"

                        if pipeline.forecast_registry is not None and result is not None:
                            due = pipeline.forecast_registry.due(current_world.t)
                            if due:
                                verified_ids: list[str] = []
                                verification_errors: list[float] = []
                                current_probs = {
                                    key: float(value)
                                    for key, value in result.simulation.get("final_outcome_probs", {}).items()
                                }
                                for forecast in due:
                                    verified = pipeline.verify_forecast(
                                        forecast.forecast_id,
                                        actual_prob=float(current_probs.get(forecast.outcome_id, 0.0)),
                                        verified_at=_utc_now(),
                                    )
                                    verified_ids.append(verified.forecast_id)
                                    if verified.error is not None:
                                        verification_errors.append(float(verified.error))
                                if verified_ids:
                                    verified_forecasts_count += len(verified_ids)
                                    verify_trace = _runtime_trace_for_verification(
                                        state=pipeline.conscious_state,
                                        verified_count=len(verified_ids),
                                        verified_ids=verified_ids,
                                        mean_abs_error=(
                                            sum(verification_errors) / len(verification_errors)
                                            if verification_errors
                                            else None
                                        ),
                                    )
                                    pipeline.conscious_state.trace_state.append(verify_trace)
                                    event_log.append(verify_trace)
                                    logged_event_ids.add(verify_trace.event_id)
                                    engine = ConsciousnessEngine(pipeline.conscious_state, pipeline.consciousness_config)
                                    engine.refresh_after_epistemic_update(
                                        world_ref=f"world:{current_world.domain_id}:{current_world.t}",
                                        runtime_metadata={
                                            "last_domain_id": str(current_world.domain_id),
                                            "last_world_step": int(current_world.t),
                                        },
                                    )
                                    pipeline.conscious_state = engine.state

                        engine = ConsciousnessEngine(pipeline.conscious_state, pipeline.consciousness_config)
                        engine.maybe_deliberate(_utc_now())
                        pipeline.conscious_state = engine.state
                    except Exception as exc:  # noqa: BLE001
                        update_failures += 1
                        update_error = str(exc)
                        should_update = False
                        LOGGER.warning("World update failed for signal_id=%s: %s", signal_id, exc)
                elif not args.include_watch:
                    skipped_watch_count += 1

                _append_unlogged_trace_events(pipeline.conscious_state, event_log, logged_event_ids)
                runtime_event = _runtime_trace_for_signal(
                    state=pipeline.conscious_state,
                    signal_payload=signal_payload,
                    trigger_mode=trigger.mode,
                    llm_used=llm_update_attempted,
                    updated_world=should_update,
                    update_error=update_error,
                )
                pipeline.conscious_state.trace_state.append(runtime_event)
                event_log.append(runtime_event)
                logged_event_ids.add(runtime_event.event_id)
                cursor_store.commit(signal_id)
                _persist_runtime_state(
                    pipeline=pipeline,
                    world_state=current_world,
                    runtime_path=runtime_path,
                    checkpoint_manager=checkpoint_manager,
                    cursor_store=cursor_store,
                    signal_memory=signal_memory,
                    pending_signals=pending_signals,
                )
                processed_count += 1
                LOGGER.info(
                    "Processed signal_id=%s mode=%s world_t=%s queue_len=%d",
                    signal_id,
                    trigger.mode,
                    current_world.t,
                    len(pending_signals),
                )
                continue

            engine = ConsciousnessEngine(pipeline.conscious_state, pipeline.consciousness_config)
            if engine.maybe_deliberate(now) is not None:
                idle_deliberations += 1
                pipeline.conscious_state = engine.state
                _append_unlogged_trace_events(pipeline.conscious_state, event_log, logged_event_ids)
                _persist_runtime_state(
                    pipeline=pipeline,
                    world_state=current_world,
                    runtime_path=runtime_path,
                    checkpoint_manager=checkpoint_manager,
                    cursor_store=cursor_store,
                    signal_memory=signal_memory,
                    pending_signals=pending_signals,
                )
                continue

            sleep_seconds = analysis_interval_seconds
            if deadline is not None:
                sleep_seconds = min(sleep_seconds, max((deadline - _utc_now()).total_seconds(), 0.0))
            if next_poll_at is not None:
                sleep_seconds = min(sleep_seconds, max((next_poll_at - _utc_now()).total_seconds(), 0.0))
            if sleep_seconds > 0.0:
                time.sleep(sleep_seconds)
    finally:
        signal.signal(signal.SIGINT, previous_sigint)
        signal.signal(signal.SIGTERM, previous_sigterm)
        _append_unlogged_trace_events(pipeline.conscious_state, event_log, logged_event_ids)
        if current_world is not None:
            _persist_runtime_state(
                pipeline=pipeline,
                world_state=current_world,
                runtime_path=runtime_path,
                checkpoint_manager=checkpoint_manager,
                cursor_store=cursor_store,
                signal_memory=signal_memory,
                pending_signals=pending_signals,
            )

    summary = {
        "status": "stopped",
        "started_at": started_at.isoformat(),
        "ended_at": _utc_now().isoformat(),
        "hours_requested": float(args.hours),
        "bootstrap_mode": bootstrap_mode,
        "bootstrap_package_path": str(package_path) if package_path.exists() else None,
        "model": model_name,
        "llm_provider": provider,
        "runtime_path": str(runtime_path),
        "event_log_path": str(event_log_path),
        "signals_seen": seen_count,
        "signals_committed": processed_count,
        "world_updates": updated_count,
        "world_update_failures": update_failures,
        "verified_forecasts": verified_forecasts_count,
        "idle_deliberations": idle_deliberations,
        "queue_len": len(pending_signals),
        "watch_skipped": skipped_watch_count,
        "trace_events": len(pipeline.conscious_state.trace_state),
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
