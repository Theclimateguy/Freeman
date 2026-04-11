"""Run Freeman against climate-news RSS streams with local Ollama chat models.

This script is designed for long local runs with checkpoint/resume:
- fetch RSS signals on a polling interval
- classify/analyze with Ollama via chat_json
- update Freeman world + consciousness state
- persist world, trace, cursor, and signal memory after each committed signal
"""

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
from freeman.agent.parameterestimator import ParameterEstimator
from freeman.agent.signalingestion import ManualSignalSource, Signal, SignalIngestionEngine, SignalMemory
from freeman.game.runner import SimConfig
from freeman.llm import OllamaChatClient
from freeman.memory.knowledgegraph import KnowledgeGraph
from freeman.core.world import WorldState
from freeman.runtime.checkpoint import CheckpointManager
from freeman.runtime.event_log import EventLog
from freeman.runtime.stream import StreamCursorStore

LOGGER = logging.getLogger("climate_stream")

DEFAULT_CLIMATE_SOURCES: list[dict[str, Any]] = [
    {
        "type": "rss",
        "url": "https://www.carbonbrief.org/feed/",
        "default_topic": "climate_news",
        "max_entries": 20,
        "source_type": "rss_carbonbrief",
    },
    {
        "type": "rss",
        "url": "https://www.theguardian.com/environment/rss",
        "default_topic": "climate_news",
        "max_entries": 20,
        "source_type": "rss_guardian_env",
    },
    {
        "type": "rss",
        "url": "https://www.noaa.gov/rss.xml",
        "default_topic": "climate_news",
        "max_entries": 20,
        "source_type": "rss_noaa",
    },
]

DEFAULT_CLIMATE_KEYWORDS = [
    "climate",
    "warming",
    "heatwave",
    "drought",
    "flood",
    "wildfire",
    "emissions",
    "carbon",
    "co2",
    "extreme weather",
    "sea level",
    "adaptation",
    "mitigation",
]


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


def _keywords_from_config(config: dict[str, Any]) -> list[str]:
    raw = config.get("agent", {}).get("climate_keywords", DEFAULT_CLIMATE_KEYWORDS)
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


def _source_configs(config: dict[str, Any]) -> list[dict[str, Any]]:
    configured = config.get("agent", {}).get("sources", [])
    if isinstance(configured, list) and configured:
        return [dict(item) for item in configured if isinstance(item, dict)]
    return [dict(item) for item in DEFAULT_CLIMATE_SOURCES]


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


def _persist_runtime_state(
    *,
    pipeline: AnalysisPipeline,
    world_state: WorldState,
    runtime_path: Path,
    checkpoint_manager: CheckpointManager,
    cursor_store: StreamCursorStore,
    signal_memory: SignalMemory,
) -> None:
    runtime_path.mkdir(parents=True, exist_ok=True)
    pipeline.knowledge_graph.save()
    _atomic_write_json(runtime_path / "world_state.json", world_state.snapshot())
    _save_signal_memory(signal_memory, runtime_path / "signal_memory.json")
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


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run Freeman on climate RSS streams with local Ollama.")
    parser.add_argument("--config-path", default="config.climate.yaml")
    parser.add_argument("--schema-path", default="freeman/domain/profiles/gim15.json")
    parser.add_argument("--hours", type=float, default=8.0)
    parser.add_argument("--poll-seconds", type=float, default=None)
    parser.add_argument("--model", default="auto")
    parser.add_argument("--ollama-base-url", default=None)
    parser.add_argument("--max-signals-per-poll", type=int, default=30)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--include-watch", action="store_true")
    parser.add_argument("--keyword", action="append", default=None)
    parser.add_argument("--log-level", default="INFO")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
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

    runtime_path = _resolve_path(config_base, runtime_cfg.get("runtime_path"), "./data/runtime_climate")
    event_log_path = _resolve_path(config_base, runtime_cfg.get("event_log_path"), str(runtime_path / "event_log.jsonl"))
    kg_path = _resolve_path(config_base, memory_cfg.get("json_path"), "./data/kg_state.json")
    schema_path = Path(args.schema_path).resolve()
    if not schema_path.exists():
        raise FileNotFoundError(f"Schema path not found: {schema_path}")

    poll_seconds = float(
        args.poll_seconds
        if args.poll_seconds is not None
        else agent_cfg.get("source_refresh_seconds", runtime_cfg.get("poll_interval_seconds", 300))
    )
    keywords = (
        [str(item).strip().lower() for item in args.keyword if str(item).strip()]
        if args.keyword
        else _keywords_from_config(config)
    )

    ollama_base_url = str(
        args.ollama_base_url
        or llm_cfg.get("base_url")
        or os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434")
    )
    model_name = _select_ollama_model(ollama_base_url, str(args.model or llm_cfg.get("model", "auto")))
    llm_client = OllamaChatClient(
        model=model_name,
        base_url=ollama_base_url,
        timeout_seconds=float(llm_cfg.get("timeout_seconds", 120.0)),
    )

    LOGGER.info("Using Ollama model=%s base_url=%s", model_name, ollama_base_url)
    LOGGER.info("Knowledge graph path=%s runtime path=%s", kg_path, runtime_path)

    knowledge_graph = KnowledgeGraph(json_path=kg_path, auto_load=True, auto_save=True)
    pipeline = AnalysisPipeline(
        knowledge_graph=knowledge_graph,
        sim_config=_build_sim_config(config),
        config_path=config_path,
    )

    runtime_path.mkdir(parents=True, exist_ok=True)
    checkpoint_manager = CheckpointManager()
    cursor_store = StreamCursorStore()
    if args.resume:
        cursor_store.load(runtime_path / "cursors.json")
    event_log = EventLog(event_log_path)
    logged_event_ids = _load_logged_ids_and_backfill_cursor(event_log, cursor_store)
    signal_memory = _load_signal_memory(runtime_path / "signal_memory.json") if args.resume else SignalMemory()

    if args.resume and (runtime_path / "checkpoint.json").exists():
        loaded_state = checkpoint_manager.load(runtime_path / "checkpoint.json")
        pipeline.conscious_state = ConsciousState.from_dict(loaded_state.to_dict(), knowledge_graph)
        LOGGER.info("Loaded consciousness checkpoint.")

    current_world: WorldState | None = None
    world_state_path = runtime_path / "world_state.json"
    if args.resume and world_state_path.exists():
        current_world = WorldState.from_snapshot(json.loads(world_state_path.read_text(encoding="utf-8")))
        LOGGER.info("Loaded world checkpoint domain_id=%s step=%s", current_world.domain_id, current_world.t)
    schema_payload = json.loads(schema_path.read_text(encoding="utf-8"))
    base_world_template = pipeline.compiler.compile(schema_payload)

    if current_world is None:
        bootstrap_result = pipeline.run(schema_payload, policies=[])
        current_world = bootstrap_result.world.clone()
        _append_unlogged_trace_events(pipeline.conscious_state, event_log, logged_event_ids)
        _persist_runtime_state(
            pipeline=pipeline,
            world_state=current_world,
            runtime_path=runtime_path,
            checkpoint_manager=checkpoint_manager,
            cursor_store=cursor_store,
            signal_memory=signal_memory,
        )
        LOGGER.info(
            "Bootstrap completed domain_id=%s dominant_outcome=%s",
            bootstrap_result.world.domain_id,
            bootstrap_result.dominant_outcome,
        )

    sources = [build_signal_source(cfg) for cfg in _source_configs(config)]
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
    processed_count = 0
    updated_count = 0
    update_failures = 0
    skipped_watch_count = 0
    seen_count = 0

    try:
        while not stop_requested:
            now = _utc_now()
            if deadline is not None and now >= deadline:
                LOGGER.info("Reached duration limit (hours=%s).", args.hours)
                break

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

            processed_this_poll = 0
            for signal_payload in fetched:
                if stop_requested:
                    break
                if processed_this_poll >= int(args.max_signals_per_poll):
                    break
                signal_id = str(signal_payload.signal_id)
                if cursor_store.is_committed(signal_id):
                    continue
                seen_count += 1
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
                    processed_this_poll += 1
                    continue

                trigger = triggers[0]
                should_update = trigger.mode in {"ANALYZE", "DEEP_DIVE"}
                llm_update_attempted = should_update
                update_error: str | None = None
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
                )
                processed_count += 1
                processed_this_poll += 1
                LOGGER.info(
                    "Processed signal_id=%s mode=%s world_t=%s dominant=%s",
                    signal_id,
                    trigger.mode,
                    current_world.t,
                    current_world.metadata.get("dominant_outcome"),
                )

            if processed_this_poll == 0:
                LOGGER.info("No new eligible climate signals this poll.")

            if stop_requested:
                break

            if deadline is not None:
                remaining = (deadline - _utc_now()).total_seconds()
                if remaining <= 0.0:
                    break
                sleep_seconds = min(float(poll_seconds), remaining)
            else:
                sleep_seconds = float(poll_seconds)

            wait_until = _utc_now() + timedelta(seconds=sleep_seconds)
            while not stop_requested and _utc_now() < wait_until:
                time.sleep(0.25)
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
            )

    summary = {
        "status": "stopped",
        "started_at": started_at.isoformat(),
        "ended_at": _utc_now().isoformat(),
        "hours_requested": float(args.hours),
        "model": model_name,
        "runtime_path": str(runtime_path),
        "event_log_path": str(event_log_path),
        "signals_seen": seen_count,
        "signals_committed": processed_count,
        "world_updates": updated_count,
        "world_update_failures": update_failures,
        "watch_skipped": skipped_watch_count,
        "trace_events": len(pipeline.conscious_state.trace_state),
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
