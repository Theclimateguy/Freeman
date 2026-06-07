"""Runtime lifecycle, storage, and persistence helpers for Freeman streams."""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
import json
import logging
from pathlib import Path
import signal
import sys
from typing import Any

import yaml

from freeman.agent.analysispipeline import AnalysisPipeline
from freeman.agent.consciousness import ConsciousState, TraceEvent
from freeman.agent.costmodel import (
    BudgetLedger,
    BudgetPolicy,
    CostModel,
    build_budget_policy,
    budget_tracking_enabled,
    resolve_budget_decision,
)
from freeman.agent.parameterestimator import ParameterEstimator
from freeman.agent.signalingestion import Signal, SignalIngestionEngine, SignalMemory
from freeman.core.world import WorldState
from freeman.game.runner import SimConfig
from freeman.runtime.checkpoint import CheckpointManager
from freeman.runtime.event_log import EventLog
from freeman.runtime.kgsnapshot import KGSnapshotManager, snapshot_manager_from_config
from freeman.runtime.stream import StreamCursorStore
from freeman.utils import deep_copy_jsonable

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

LOGGER = logging.getLogger("stream_runtime")

DEFAULT_RUNTIME_CONFIG: dict[str, Any] = {
    "agent": {
        "budget_usd_per_day": 0.50,
        "cost_governance": {},
    },
    "memory": {
        "json_path": "./data/kg_state.json",
    },
    "runtime": {
        "runtime_path": "./data/runtime",
        "event_log_path": "./data/runtime/event_log.jsonl",
    },
}

def _utc_now() -> datetime:
    return datetime.now(timezone.utc).replace(microsecond=0)


def _merge_dicts(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _merge_dicts(merged[key], value)
        else:
            merged[key] = value
    return merged


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
        return dict(DEFAULT_RUNTIME_CONFIG)
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return _merge_dicts(DEFAULT_RUNTIME_CONFIG, payload)


def _read_optional_text(path: str | Path | None) -> str:
    if path is None:
        return ""
    target = Path(path).resolve()
    if not target.exists():
        return ""
    return target.read_text(encoding="utf-8").strip()


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


def _stream_filter_config(config: dict[str, Any]) -> dict[str, Any]:
    agent_cfg = config.get("agent", {})
    runtime_cfg = config.get("runtime", {})
    filter_cfg = dict(agent_cfg.get("stream_filter", {}) or {})
    if "min_relevance_score" not in filter_cfg:
        filter_cfg["min_relevance_score"] = float(agent_cfg.get("min_relevance_score", runtime_cfg.get("min_relevance_score", 0.0)) or 0.0)
    if "min_keyword_matches" not in filter_cfg:
        filter_cfg["min_keyword_matches"] = int(agent_cfg.get("min_keyword_matches", runtime_cfg.get("min_keyword_matches", 0)) or 0)
    if "agent_min_relevance_score" not in filter_cfg:
        filter_cfg["agent_min_relevance_score"] = float(filter_cfg.get("min_relevance_score", 0.0))
    return filter_cfg


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



@dataclass
class RuntimePaths:
    config_path: Path
    config_base: Path
    runtime_path: Path
    event_log_path: Path
    kg_path: Path
    schema_path: Path | None


@dataclass
class RuntimeStorage:
    checkpoint_manager: CheckpointManager
    cursor_store: StreamCursorStore
    event_log: EventLog
    logged_event_ids: set[str]
    signal_memory: SignalMemory
    pending_signals: list[Signal]
    queued_signal_ids: set[str]
    snapshot_manager: KGSnapshotManager = field(
        default_factory=lambda: KGSnapshotManager(snapshot_dir=Path("."), enabled=False)
    )


@dataclass
class BootstrapResult:
    pipeline: AnalysisPipeline
    current_world: WorldState
    base_world_template: WorldState
    llm_client: Any
    estimator: ParameterEstimator
    bootstrap_mode: str
    provider: str
    model_name: str
    package_path: Path


@dataclass
class RuntimeContext:
    config: dict[str, Any]
    paths: RuntimePaths
    pipeline: AnalysisPipeline
    current_world: WorldState
    base_world_template: WorldState
    estimator: ParameterEstimator
    ingestion_engine: SignalIngestionEngine
    llm_client: Any
    event_log: EventLog
    logged_event_ids: set[str]
    cursor_store: StreamCursorStore
    signal_memory: SignalMemory
    pending_signals: list[Signal]
    queued_signal_ids: set[str]
    checkpoint_manager: CheckpointManager
    runtime_path: Path
    keywords: list[str]
    filter_cfg: dict[str, Any]
    args: argparse.Namespace
    package_path: Path
    bootstrap_mode: str
    provider: str
    model_name: str
    sources: list[Any]
    poll_seconds: float
    analysis_interval_seconds: float
    started_at: datetime
    deadline: datetime | None
    budget_policy: BudgetPolicy = field(default_factory=BudgetPolicy)
    cost_model: CostModel = field(default_factory=CostModel)
    budget_ledger: BudgetLedger | None = None
    snapshot_manager: KGSnapshotManager = field(
        default_factory=lambda: KGSnapshotManager(snapshot_dir=Path("."), enabled=False)
    )
    stats: dict[str, int] = field(default_factory=dict)
    stop_requested: bool = False
    run_id: str = ""


@dataclass
class SignalResult:
    processed: int = 0
    updated: int = 0
    update_failures: int = 0
    verified_forecasts: int = 0
    skipped_watch: int = 0
    filtered_out: int = 0


@dataclass
class LoopSummary:
    started_at: datetime
    ended_at: datetime
    hours_requested: float
    bootstrap_mode: str
    bootstrap_package_path: str | None
    model: str
    llm_provider: str
    runtime_path: str
    event_log_path: str
    signals_seen: int
    signals_committed: int
    world_updates: int
    world_update_failures: int
    verified_forecasts: int
    runtime_step: int
    idle_deliberations: int
    queue_len: int
    watch_skipped: int
    filtered_out_count: int
    trace_events: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": "stopped",
            "started_at": self.started_at.isoformat(),
            "ended_at": self.ended_at.isoformat(),
            "hours_requested": self.hours_requested,
            "bootstrap_mode": self.bootstrap_mode,
            "bootstrap_package_path": self.bootstrap_package_path,
            "model": self.model,
            "llm_provider": self.llm_provider,
            "runtime_path": self.runtime_path,
            "event_log_path": self.event_log_path,
            "signals_seen": self.signals_seen,
            "signals_committed": self.signals_committed,
            "world_updates": self.world_updates,
            "world_update_failures": self.world_update_failures,
            "verified_forecasts": self.verified_forecasts,
            "runtime_step": self.runtime_step,
            "idle_deliberations": self.idle_deliberations,
            "queue_len": self.queue_len,
            "watch_skipped": self.watch_skipped,
            "filtered_out_count": self.filtered_out_count,
            "trace_events": self.trace_events,
        }



def _resolve_runtime_paths(args: argparse.Namespace, config: dict[str, Any], config_path: Path) -> RuntimePaths:
    config_base = config_path.parent
    runtime_cfg = config.get("runtime", {})
    memory_cfg = config.get("memory", {})
    schema_path_arg = getattr(args, "schema_path", None)
    return RuntimePaths(
        config_path=config_path,
        config_base=config_base,
        runtime_path=_resolve_path(config_base, runtime_cfg.get("runtime_path"), "./data/runtime"),
        event_log_path=_resolve_path(
            config_base,
            runtime_cfg.get("event_log_path"),
            str(_resolve_path(config_base, runtime_cfg.get("runtime_path"), "./data/runtime") / "event_log.jsonl"),
        ),
        kg_path=_resolve_path(config_base, memory_cfg.get("json_path"), "./data/kg_state.json"),
        schema_path=_resolve_path(config_base, schema_path_arg, schema_path_arg) if schema_path_arg else None,
    )


def _initialize_runtime_storage(args: argparse.Namespace, runtime_path: Path, event_log_path: Path) -> RuntimeStorage:
    runtime_path.mkdir(parents=True, exist_ok=True)
    checkpoint_manager = CheckpointManager()
    cursor_store = StreamCursorStore()
    if args.resume:
        cursor_store.load(runtime_path / "cursors.json")
    event_log = EventLog(event_log_path)
    logged_event_ids = _load_logged_ids_and_backfill_cursor(event_log, cursor_store)
    signal_memory = _load_signal_memory(runtime_path / "signal_memory.json") if args.resume else SignalMemory()
    pending_signals = _load_pending_queue(runtime_path / "pending_signals.json") if args.resume else []
    return RuntimeStorage(
        checkpoint_manager=checkpoint_manager,
        cursor_store=cursor_store,
        event_log=event_log,
        snapshot_manager=KGSnapshotManager(snapshot_dir=runtime_path / "kg_snapshots", enabled=False),
        logged_event_ids=logged_event_ids,
        signal_memory=signal_memory,
        pending_signals=pending_signals,
        queued_signal_ids={str(item.signal_id) for item in pending_signals},
    )


def _persist_context(ctx: RuntimeContext) -> None:
    _append_unlogged_trace_events(ctx.pipeline.conscious_state, ctx.event_log, ctx.logged_event_ids)
    _persist_runtime_state(
        pipeline=ctx.pipeline,
        world_state=ctx.current_world,
        runtime_path=ctx.runtime_path,
        checkpoint_manager=ctx.checkpoint_manager,
        cursor_store=ctx.cursor_store,
        signal_memory=ctx.signal_memory,
        pending_signals=ctx.pending_signals,
    )


def _write_kg_snapshot(
    ctx: RuntimeContext,
    *,
    reason: str,
    signal_payload: Signal | None = None,
    trigger_mode: str | None = None,
    extra_metadata: dict[str, Any] | None = None,
) -> Path | None:
    return ctx.snapshot_manager.write_snapshot(
        ctx.pipeline.knowledge_graph,
        runtime_step=int(ctx.current_world.runtime_step),
        reason=reason,
        domain_id=str(ctx.current_world.domain_id),
        signal_id=str(signal_payload.signal_id) if signal_payload is not None else None,
        trigger_mode=trigger_mode,
        world_t=int(ctx.current_world.t),
        metadata=extra_metadata,
    )


def _poll_seconds(args: argparse.Namespace, config: dict[str, Any]) -> float:
    agent_cfg = config.get("agent", {})
    runtime_cfg = config.get("runtime", {})
    return float(
        args.poll_seconds
        if args.poll_seconds is not None
        else agent_cfg.get("source_refresh_seconds", runtime_cfg.get("poll_interval_seconds", 300))
    )


def _runtime_keywords(
    args: argparse.Namespace,
    config: dict[str, Any],
    default_keywords: list[str] | None,
) -> list[str]:
    if args.keyword:
        return [str(item).strip().lower() for item in args.keyword if str(item).strip()]
    return _keywords_from_config(config, default_keywords)


def _collect_sources(config: dict[str, Any], default_sources: list[dict[str, Any]] | None = None) -> list[Any]:
    sources = [build_signal_source(cfg) for cfg in _source_configs(config, default_sources)]
    LOGGER.info("Configured source count=%d", len(sources))
    return sources


def _initial_runtime_stats() -> dict[str, int]:
    return {
        "signals_seen": 0,
        "signals_committed": 0,
        "world_updates": 0,
        "world_update_failures": 0,
        "verified_forecasts": 0,
        "idle_deliberations": 0,
        "watch_skipped": 0,
        "filtered_out_count": 0,
        "ontology_repairs_triggered": 0,
        "budget_blocked_tasks": 0,
        "queue_backpressure_skipped": 0,
    }


def _budget_tracking_enabled_for_runtime(ctx: RuntimeContext) -> bool:
    return ctx.budget_ledger is not None


def _budget_summary_for_runtime(ctx: RuntimeContext) -> dict[str, Any]:
    if ctx.budget_ledger is not None:
        return ctx.budget_ledger.summary()
    return {
        "tracking_enabled": bool(budget_tracking_enabled(ctx.config)),
        "ledger_path": str((ctx.runtime_path / "cost_ledger.jsonl").resolve()),
        "configured_usd_per_day": float(ctx.budget_policy.max_compute_budget_per_session),
        "spent_usd": 0.0,
        "remaining_usd": float(ctx.budget_policy.max_compute_budget_per_session),
        "entry_count": 0,
        "allowed_count": 0,
        "blocked_count": 0,
        "by_task_type": {},
        "stop_reasons": {},
    }


def _world_complexity_counts(world: WorldState) -> tuple[int, int, int]:
    return len(world.actors), len(world.resources), max(len(world.outcomes), 1)


def _approved_mode_cost(
    estimate_for_mode: Any,
    approved_mode: str,
    *,
    allowed: bool,
) -> float:
    if not allowed or approved_mode == "WATCH":
        return 0.0
    return float(estimate_for_mode(approved_mode).estimated_cost)


def _signal_budget_decision(ctx: RuntimeContext, signal_payload: Signal, trigger: Any) -> tuple[Any | None, float]:
    if not _budget_tracking_enabled_for_runtime(ctx):
        return None, 0.0
    actors, resources, domains = _world_complexity_counts(ctx.current_world)
    raw_requested_mode = str(getattr(trigger, "requested_mode", "") or "").upper()
    raw_mode = str(getattr(trigger, "mode", "") or "WATCH").upper()
    requested_mode = raw_requested_mode if raw_requested_mode not in {"", "WATCH"} or raw_mode == "WATCH" else raw_mode
    sim_steps_by_mode = {
        "WATCH": 0,
        "ANALYZE": max(1, min(int(ctx.pipeline.sim_config.max_steps), 12)),
        "DEEP_DIVE": max(1, int(ctx.pipeline.sim_config.max_steps)),
    }
    llm_calls_by_mode = {"WATCH": 0, "ANALYZE": 1, "DEEP_DIVE": 2}
    kg_updates_by_mode = {"WATCH": 0, "ANALYZE": 2, "DEEP_DIVE": 4}
    token_estimate = max(len(str(signal_payload.text).split()), 1) * 24

    def _estimate_for_mode(mode: str):
        normalized_mode = str(mode or "WATCH").upper()
        idle_mode = normalized_mode == "WATCH"
        return ctx.cost_model.estimate(
            task_id=f"signal:{signal_payload.signal_id}:{normalized_mode.lower()}",
            llm_calls=llm_calls_by_mode.get(normalized_mode, 0),
            sim_steps=sim_steps_by_mode.get(normalized_mode, 0),
            actors=0 if idle_mode else actors,
            resources=0 if idle_mode else resources,
            domains=0 if idle_mode else domains,
            kg_updates=kg_updates_by_mode.get(normalized_mode, 0),
            embedding_tokens_used=0 if idle_mode else token_estimate,
        )

    deep_dive_depth = int(ctx.pipeline.conscious_state.runtime_metadata.get("deep_dive_depth", 0))
    decision = resolve_budget_decision(
        cost_model=ctx.cost_model,
        requested_mode=requested_mode,
        estimate_for_mode=_estimate_for_mode,
        budget_spent=ctx.budget_ledger.spent_usd,
        deep_dive_depth=deep_dive_depth,
    )
    return decision, _approved_mode_cost(_estimate_for_mode, decision.approved_mode, allowed=decision.allowed)


def _repair_budget_decision(ctx: RuntimeContext, *, gap_topics: list[str]) -> tuple[Any | None, float]:
    if not _budget_tracking_enabled_for_runtime(ctx):
        return None, 0.0
    actors, resources, domains = _world_complexity_counts(ctx.current_world)
    topic_count = max(len([topic for topic in gap_topics if str(topic).strip()]), 1)

    def _estimate_for_mode(mode: str):
        normalized_mode = str(mode or "WATCH").upper()
        idle_mode = normalized_mode == "WATCH"
        return ctx.cost_model.estimate(
            task_id=f"ontology_repair:{ctx.current_world.runtime_step}:{topic_count}:{normalized_mode.lower()}",
            llm_calls=0 if normalized_mode == "WATCH" else (2 if ctx.bootstrap_mode == "llm_synthesize" else 0),
            sim_steps=0 if normalized_mode == "WATCH" else max(1, min(topic_count * 2, 8)),
            actors=0 if idle_mode else actors,
            resources=0 if idle_mode else resources,
            domains=0 if idle_mode else domains,
            kg_updates=0 if idle_mode else topic_count * 2,
            embedding_tokens_used=0 if idle_mode else topic_count * 32,
        )

    decision = resolve_budget_decision(
        cost_model=ctx.cost_model,
        requested_mode="ANALYZE",
        estimate_for_mode=_estimate_for_mode,
        budget_spent=ctx.budget_ledger.spent_usd,
        deep_dive_depth=0,
    )
    return decision, _approved_mode_cost(_estimate_for_mode, decision.approved_mode, allowed=decision.allowed)


def _build_runtime_context(
    *,
    args: argparse.Namespace,
    config: dict[str, Any],
    paths: RuntimePaths,
    storage: RuntimeStorage,
    bootstrap: BootstrapResult,
    default_sources: list[dict[str, Any]] | None,
    default_keywords: list[str] | None,
) -> RuntimeContext:
    started_at = _utc_now()
    hours_requested = float(args.hours)
    storage.snapshot_manager = snapshot_manager_from_config(config, runtime_path=paths.runtime_path, config_base=paths.config_base)
    policy = build_budget_policy(config)
    tracking_enabled = budget_tracking_enabled(config)
    budget_ledger = BudgetLedger(paths.runtime_path / "cost_ledger.jsonl", policy=policy, auto_load=True) if tracking_enabled else None
    return RuntimeContext(
        config=deep_copy_jsonable(config),
        paths=paths,
        pipeline=bootstrap.pipeline,
        current_world=bootstrap.current_world,
        base_world_template=bootstrap.base_world_template,
        estimator=bootstrap.estimator,
        ingestion_engine=SignalIngestionEngine(),
        llm_client=bootstrap.llm_client,
        event_log=storage.event_log,
        snapshot_manager=storage.snapshot_manager,
        logged_event_ids=storage.logged_event_ids,
        cursor_store=storage.cursor_store,
        signal_memory=storage.signal_memory,
        pending_signals=storage.pending_signals,
        queued_signal_ids=storage.queued_signal_ids,
        checkpoint_manager=storage.checkpoint_manager,
        runtime_path=paths.runtime_path,
        keywords=_runtime_keywords(args, config, default_keywords),
        filter_cfg=_stream_filter_config(config),
        args=args,
        package_path=bootstrap.package_path,
        bootstrap_mode=bootstrap.bootstrap_mode,
        provider=bootstrap.provider,
        model_name=bootstrap.model_name,
        sources=_collect_sources(config, default_sources),
        poll_seconds=_poll_seconds(args, config),
        analysis_interval_seconds=max(float(args.analysis_interval_seconds), 0.1),
        started_at=started_at,
        deadline=None if hours_requested <= 0.0 else started_at + timedelta(hours=hours_requested),
        budget_policy=policy,
        cost_model=CostModel(policy),
        budget_ledger=budget_ledger,
        stats=_initial_runtime_stats(),
    )


def _install_stop_handlers(ctx: RuntimeContext) -> tuple[Any, Any]:
    def _request_stop(signum, frame):  # noqa: ANN001
        del signum, frame
        ctx.stop_requested = True

    previous_sigint = signal.getsignal(signal.SIGINT)
    previous_sigterm = signal.getsignal(signal.SIGTERM)
    signal.signal(signal.SIGINT, _request_stop)
    signal.signal(signal.SIGTERM, _request_stop)
    return previous_sigint, previous_sigterm


def _restore_stop_handlers(previous_sigint: Any, previous_sigterm: Any) -> None:
    signal.signal(signal.SIGINT, previous_sigint)
    signal.signal(signal.SIGTERM, previous_sigterm)



__all__ = [
    "BootstrapResult",
    "DEFAULT_RUNTIME_CONFIG",
    "LoopSummary",
    "RuntimeContext",
    "RuntimePaths",
    "RuntimeStorage",
    "SignalResult",
    "_append_unlogged_trace_events",
    "_atomic_write_json",
    "_build_runtime_context",
    "_build_sim_config",
    "_budget_summary_for_runtime",
    "_collect_sources",
    "_initial_runtime_stats",
    "_initialize_runtime_storage",
    "_install_stop_handlers",
    "_load_yaml",
    "_persist_context",
    "_persist_runtime_state",
    "_read_optional_text",
    "_repair_budget_decision",
    "_resolve_runtime_paths",
    "_restore_stop_handlers",
    "_signal_budget_decision",
    "_source_configs",
    "_stream_filter_config",
    "_to_datetime",
    "_utc_now",
    "_write_kg_snapshot",
    "build_signal_source",
]
