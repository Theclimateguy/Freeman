"""Run Freeman against configurable signal streams with checkpoint/resume."""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
import json
import logging
import os
from pathlib import Path
import re
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
from freeman.core.scorer import score_outcomes
from freeman.game.runner import SimConfig
from freeman.interface.factory import build_embedding_adapter, build_knowledge_graph, build_vectorstore
from freeman.llm import DeepSeekChatClient, OllamaChatClient, OpenAIChatClient
from freeman.llm.orchestrator import DeepSeekFreemanOrchestrator
from freeman.memory.knowledgegraph import KnowledgeGraph
from freeman.core.world import WorldState
from freeman.runtime.checkpoint import CheckpointManager
from freeman.runtime.event_log import EventLog
from freeman.runtime.kgsnapshot import KGSnapshotManager, snapshot_manager_from_config
from freeman.runtime.queryengine import RuntimeAnswerEngine, RuntimeQueryEngine, load_runtime_artifacts as _load_runtime_query_artifacts
from freeman.runtime.stream import StreamCursorStore
from freeman.utils import deep_copy_jsonable

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


def _signal_haystack(signal_payload: Signal) -> str:
    haystack = " ".join(
        [
            signal_payload.topic,
            signal_payload.text,
            " ".join(signal_payload.entities),
            json.dumps(signal_payload.metadata, ensure_ascii=False),
        ]
    ).lower()
    return haystack


def _keyword_match_details(signal_payload: Signal, keywords: list[str]) -> tuple[int, list[str]]:
    if not keywords:
        return 0, []
    haystack = _signal_haystack(signal_payload)
    matched = [keyword for keyword in keywords if keyword in haystack]
    return len(matched), matched


def _tokenize_text(text: str) -> set[str]:
    return {token for token in re.findall(r"[a-z0-9_]+", str(text).lower()) if len(token) >= 3}


def _world_ontology_terms(world: WorldState | None) -> set[str]:
    if world is None:
        return set()
    terms = _tokenize_text(world.domain_id)
    for actor in world.actors.values():
        terms.update(_tokenize_text(actor.id))
        terms.update(_tokenize_text(getattr(actor, "name", "")))
    for resource in world.resources.values():
        terms.update(_tokenize_text(resource.id))
        terms.update(_tokenize_text(getattr(resource, "name", "")))
    for outcome in world.outcomes.values():
        terms.update(_tokenize_text(outcome.id))
        terms.update(_tokenize_text(getattr(outcome, "name", "")))
    return terms


def _active_hypothesis_terms(state: ConsciousState) -> set[str]:
    terms: set[str] = set()
    for node in state.self_model_ref.get_nodes_by_type("active_hypothesis"):
        terms.update(_tokenize_text(str(node.payload.get("outcome_id", ""))))
        terms.update(_tokenize_text(str(node.payload.get("domain", node.domain or ""))))
    for node_id in state.goal_state:
        terms.update(_tokenize_text(node_id))
    return terms


def _relevance_score(
    signal_payload: Signal,
    *,
    keywords: list[str],
    ontology_terms: set[str],
    hypothesis_terms: set[str],
) -> tuple[float, dict[str, Any]]:
    signal_tokens = _tokenize_text(_signal_haystack(signal_payload))
    keyword_hits, matched_keywords = _keyword_match_details(signal_payload, keywords)
    keyword_score = (keyword_hits / max(len(keywords), 1)) if keywords else 0.0
    ontology_overlap = len(signal_tokens & ontology_terms)
    ontology_score = ontology_overlap / max(min(len(ontology_terms), 12), 1) if ontology_terms else 0.0
    hypothesis_overlap = len(signal_tokens & hypothesis_terms)
    hypothesis_score = hypothesis_overlap / max(min(len(hypothesis_terms), 8), 1) if hypothesis_terms else 0.0
    score = max(0.0, min(1.0, 0.45 * keyword_score + 0.35 * ontology_score + 0.20 * hypothesis_score))
    return score, {
        "keyword_hits": keyword_hits,
        "matched_keywords": matched_keywords,
        "ontology_overlap": ontology_overlap,
        "hypothesis_overlap": hypothesis_overlap,
        "score": score,
    }


def _signal_matches_keywords(signal_payload: Signal, keywords: list[str], min_keyword_matches: int = 1) -> bool:
    if not keywords:
        return True
    keyword_hits, _matched = _keyword_match_details(signal_payload, keywords)
    return keyword_hits >= max(int(min_keyword_matches), 1)


def _self_calibration_started(pipeline: AnalysisPipeline) -> bool:
    return bool(pipeline.knowledge_graph.query(node_type="self_observation"))


def _update_stream_relevance_stats(state: ConsciousState, *, score: float, accepted: bool) -> dict[str, Any]:
    previous = dict(state.runtime_metadata.get("stream_relevance", {}) or {})
    scored = int(previous.get("scored_count", 0)) + 1
    accepted_count = int(previous.get("accepted_count", 0)) + (1 if accepted else 0)
    rejected_count = int(previous.get("rejected_count", 0)) + (0 if accepted else 1)
    total_score = float(previous.get("total_score", 0.0)) + float(score)
    return {
        "scored_count": scored,
        "accepted_count": accepted_count,
        "rejected_count": rejected_count,
        "total_score": total_score,
        "mean_score": total_score / max(scored, 1),
        "last_score": float(score),
        "last_decision": "accept" if accepted else "reject",
    }


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
    extra_diff: dict[str, Any] | None = None,
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
    if extra_diff:
        diff.update(json.loads(json.dumps(extra_diff, ensure_ascii=False)))
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


def _verify_due_forecasts(
    *,
    pipeline: AnalysisPipeline,
    state: ConsciousState,
    event_log: EventLog,
    logged_event_ids: set[str],
    current_world: WorldState,
    current_probs: dict[str, float],
    current_signal_id: str | None = None,
) -> int:
    """Verify all due forecasts against the current posterior using monotonic runtime_step."""

    if pipeline.forecast_registry is None:
        return 0
    due = pipeline.forecast_registry.due(current_world.runtime_step)
    if not due:
        return 0

    verified_ids: list[str] = []
    verification_errors: list[float] = []
    for forecast in due:
        verified = pipeline.verify_forecast(
            forecast.forecast_id,
            actual_prob=float(current_probs.get(forecast.outcome_id, 0.0)),
            verified_at=_utc_now(),
            current_signal_id=current_signal_id,
        )
        verified_ids.append(verified.forecast_id)
        if verified.error is not None:
            verification_errors.append(float(verified.error))
    if not verified_ids:
        return 0

    verify_trace = _runtime_trace_for_verification(
        state=state,
        verified_count=len(verified_ids),
        verified_ids=verified_ids,
        mean_abs_error=(sum(verification_errors) / len(verification_errors) if verification_errors else None),
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
            "last_runtime_step": int(current_world.runtime_step),
        },
    )
    pipeline.conscious_state = engine.state
    return len(verified_ids)


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
    snapshot_manager: KGSnapshotManager = field(
        default_factory=lambda: KGSnapshotManager(snapshot_dir=Path("."), enabled=False)
    )
    stats: dict[str, int] = field(default_factory=dict)
    stop_requested: bool = False


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
    parser.add_argument("--query", choices=["forecasts", "explain", "anomalies", "causal", "semantic", "answer"], default=None)
    parser.add_argument("--forecast-id", default=None)
    parser.add_argument("--text", default=None)
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--status", default=None)
    parser.add_argument("--log-level", default="INFO")
    return parser


def _load_query_pipeline(config: dict[str, Any], paths: RuntimePaths) -> AnalysisPipeline:
    del config
    return _load_runtime_query_artifacts(paths.config_path).pipeline


def _query_anomalies(pipeline: AnalysisPipeline) -> dict[str, Any]:
    anomaly_nodes = [
        node.snapshot()
        for node in pipeline.knowledge_graph.query(node_type="anomaly_candidate")
    ]
    ontology_gap_traits = [
        node.snapshot()
        for node in pipeline.knowledge_graph.query(
            node_type="identity_trait",
            metadata_filters={"payload.trait_key": "ontology_gap"},
        )
    ]
    return {
        "anomaly_candidates": anomaly_nodes,
        "ontology_gap_traits": ontology_gap_traits,
    }


def _query_causal_edges(pipeline: AnalysisPipeline, *, limit: int) -> list[dict[str, Any]]:
    causal_edges = [
        edge.snapshot()
        for edge in pipeline.knowledge_graph.edges()
        if edge.relation_type in {"causes", "propagates_to", "threshold_exceeded"}
    ]
    causal_edges.sort(
        key=lambda item: (
            int((item.get("metadata") or {}).get("runtime_step", -1)),
            str(item.get("updated_at", "")),
            str(item.get("id", "")),
        ),
        reverse=True,
    )
    return causal_edges[: max(int(limit), 1)]


def _handle_query_mode(args: argparse.Namespace, config: dict[str, Any], paths: RuntimePaths) -> int:
    if args.query in {"semantic", "answer"}:
        if not str(args.text or "").strip():
            raise RuntimeError(f"--query {args.query} requires --text.")
        artifacts = _load_runtime_query_artifacts(paths.config_path)
        if args.query == "semantic":
            print(
                json.dumps(
                    RuntimeQueryEngine(artifacts).semantic_query(str(args.text), limit=args.limit).to_dict(),
                    indent=2,
                    sort_keys=True,
                )
            )
            return 0
        print(
            json.dumps(
                RuntimeAnswerEngine(artifacts).answer(str(args.text), limit=args.limit),
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    pipeline = _load_query_pipeline(config, paths)
    if args.query == "forecasts":
        payload = [summary.to_dict() for summary in pipeline.list_forecasts(status=args.status)]
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0
    if args.query == "explain":
        if not args.forecast_id:
            raise RuntimeError("--query explain requires --forecast-id.")
        explanation = pipeline.explain_forecast(str(args.forecast_id))
        print(explanation.to_text())
        return 0
    if args.query == "anomalies":
        print(json.dumps(_query_anomalies(pipeline), indent=2, sort_keys=True))
        return 0
    if args.query == "causal":
        print(json.dumps(_query_causal_edges(pipeline, limit=args.limit), indent=2, sort_keys=True))
        return 0
    raise ValueError(f"Unsupported query mode: {args.query}")


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
    }


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

    bootstrap_mode = str(
        args.bootstrap_mode
        or bootstrap_cfg.get("mode")
        or ("schema_path" if paths.schema_path is not None else "llm_synthesize")
    ).strip().lower()
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
    synthesized_package = None if force_rebuild else (json.loads(package_path.read_text(encoding="utf-8")) if package_path.exists() else None)

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
        }
    elif synthesized_package is None and bootstrap_mode == "llm_synthesize":
        domain_brief = str(domain_brief_override or domain_brief_inline or _read_optional_text(domain_brief_path)).strip()
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
                max_retries=int(bootstrap_cfg.get("max_retries", 10)),
                trial_steps=int(bootstrap_cfg.get("trial_steps", 3)),
                config=_build_sim_config(config),
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
            }
            _atomic_write_json(package_path, synthesized_package)
            LOGGER.info("Synthesized bootstrap package attempts=%d saved=%s", attempts, package_path)
        except Exception as exc:  # noqa: BLE001
            fallback_candidate = bootstrap_cfg.get("fallback_schema_path")
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
            }
            _atomic_write_json(package_path, synthesized_package)
    elif synthesized_package is None:
        raise ValueError(f"Unsupported bootstrap mode: {bootstrap_mode}")

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


def _domain_brief_history_path(runtime_path: Path) -> Path:
    return runtime_path / "domain_brief_history.jsonl"


def _append_domain_brief_history(runtime_path: Path, *, gap_topics: list[str], brief: str) -> None:
    history_path = _domain_brief_history_path(runtime_path)
    version = 1
    if history_path.exists():
        lines = [line for line in history_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        version = len(lines) + 1
    payload = {
        "version": version,
        "timestamp": _utc_now().isoformat(),
        "gap_topics": list(gap_topics),
        "brief": str(brief),
    }
    history_path.parent.mkdir(parents=True, exist_ok=True)
    with history_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def _pending_repair_requests(ctx: RuntimeContext) -> list[TraceEvent]:
    handled_ids = set(ctx.pipeline.conscious_state.runtime_metadata.get("handled_ontology_repair_requests", []))
    return [
        event
        for event in ctx.pipeline.conscious_state.trace_state
        if event.operator == "ontology_repair_request" and event.event_id not in handled_ids
    ]


def _mark_gap_traits_failed(ctx: RuntimeContext, *, trait_ids: list[str]) -> None:
    for trait_id in trait_ids:
        node = ctx.pipeline.knowledge_graph.get_node(trait_id, lazy_embed=False)
        if node is None:
            continue
        payload = deep_copy_jsonable(node.metadata.get("payload", {}))
        payload["repair_failed"] = True
        node.metadata["payload"] = payload
        node.updated_at = _utc_now().isoformat()
        ctx.pipeline.knowledge_graph.update_node(node)


def _trigger_ontology_repair(ctx: RuntimeContext, *, gap_topics: list[str], repair_trait_ids: list[str]) -> bool:
    ontology_cfg = ((ctx.config.get("agent") or {}).get("ontology_repair") or {})
    max_repairs = max(int(ontology_cfg.get("max_repairs_per_session", 3)), 0)
    if max_repairs > 0 and int(ctx.stats.get("ontology_repairs_triggered", 0)) >= max_repairs:
        LOGGER.warning("Ontology repair skipped: max repairs per session reached.")
        return False

    package_payload = json.loads(ctx.package_path.read_text(encoding="utf-8")) if ctx.package_path.exists() else {}
    current_brief = str(
        package_payload.get("domain_brief")
        or ((ctx.config.get("agent") or {}).get("bootstrap") or {}).get("domain_brief")
        or _read_optional_text(((ctx.config.get("agent") or {}).get("bootstrap") or {}).get("domain_brief_path"))
    ).strip()
    if not current_brief:
        LOGGER.warning("Ontology repair skipped: no domain brief available.")
        _mark_gap_traits_failed(ctx, trait_ids=repair_trait_ids)
        return False

    unique_topics = sorted({str(topic).strip() for topic in gap_topics if str(topic).strip()})
    if not unique_topics:
        LOGGER.warning("Ontology repair skipped: no gap topics collected.")
        _mark_gap_traits_failed(ctx, trait_ids=repair_trait_ids)
        return False

    repair_addendum = "\n\n## Newly observed topics requiring integration:\n" + "\n".join(f"- {topic}" for topic in unique_topics)
    repaired_brief = current_brief + repair_addendum
    _append_domain_brief_history(ctx.runtime_path, gap_topics=unique_topics, brief=repaired_brief)
    if ctx.package_path.exists():
        ctx.package_path.unlink()

    previous_state = ConsciousState.from_dict(ctx.pipeline.conscious_state.to_dict(), ctx.pipeline.knowledge_graph)
    previous_runtime_step = int(ctx.current_world.runtime_step)
    try:
        bootstrap = _bootstrap(
            args=ctx.args,
            config=ctx.config,
            paths=ctx.paths,
            storage=RuntimeStorage(
                checkpoint_manager=ctx.checkpoint_manager,
                cursor_store=ctx.cursor_store,
                event_log=ctx.event_log,
                logged_event_ids=ctx.logged_event_ids,
                signal_memory=ctx.signal_memory,
                pending_signals=ctx.pending_signals,
                queued_signal_ids=ctx.queued_signal_ids,
            ),
            domain_brief_override=repaired_brief,
            force_rebuild=True,
            load_resume_state=False,
            load_resume_world=False,
            knowledge_graph=ctx.pipeline.knowledge_graph if bool(ontology_cfg.get("preserve_kg", True)) else None,
            forecast_registry=ctx.pipeline.forecast_registry,
        )
    except Exception as exc:  # noqa: BLE001
        LOGGER.warning("Ontology repair bootstrap failed: %s", exc)
        _mark_gap_traits_failed(ctx, trait_ids=repair_trait_ids)
        return False

    bootstrap.pipeline.conscious_state = ConsciousState.from_dict(previous_state.to_dict(), bootstrap.pipeline.knowledge_graph)
    bootstrap.current_world.runtime_step = previous_runtime_step
    bootstrap.base_world_template.runtime_step = previous_runtime_step
    ctx.pipeline = bootstrap.pipeline
    ctx.current_world = bootstrap.current_world
    ctx.base_world_template = bootstrap.base_world_template
    ctx.estimator = bootstrap.estimator
    ctx.llm_client = bootstrap.llm_client
    ctx.bootstrap_mode = bootstrap.bootstrap_mode
    ctx.provider = bootstrap.provider
    ctx.model_name = bootstrap.model_name
    ctx.package_path = bootstrap.package_path
    ctx.stats["ontology_repairs_triggered"] += 1
    ctx.pipeline.conscious_state.runtime_metadata["ontology_repairs_triggered"] = int(ctx.stats["ontology_repairs_triggered"])
    LOGGER.info("Ontology repair complete. gap_topics=%s", unique_topics)
    _persist_context(ctx)
    return True


def _check_and_handle_repair_request(ctx: RuntimeContext) -> bool:
    repair_events = _pending_repair_requests(ctx)
    if not repair_events:
        return False
    gap_topics: list[str] = []
    repair_trait_ids: list[str] = []
    for event in repair_events:
        gap_topics.extend(str(topic) for topic in event.diff.get("gap_topics", []) if str(topic).strip())
        repair_trait_ids.extend(str(node_id) for node_id in event.diff.get("repair_trait_ids", []) if str(node_id).strip())
    handled = _trigger_ontology_repair(
        ctx,
        gap_topics=gap_topics,
        repair_trait_ids=sorted(set(repair_trait_ids)),
    )
    handled_ids = list(dict.fromkeys([*ctx.pipeline.conscious_state.runtime_metadata.get("handled_ontology_repair_requests", []), *[event.event_id for event in repair_events]]))
    ctx.pipeline.conscious_state.runtime_metadata["handled_ontology_repair_requests"] = handled_ids
    if handled:
        _persist_context(ctx)
    return handled


def _run_poll(sources: list[Any], ctx: RuntimeContext) -> int:
    keywords = ctx.keywords
    min_relevance_score = float(ctx.filter_cfg.get("min_relevance_score", 0.0))
    min_keyword_matches = int(ctx.filter_cfg.get("min_keyword_matches", 0))
    fetched: list[Signal] = []
    for source in sources:
        try:
            fetched.extend(source.fetch())
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning("Source fetch failed (%s): %s", getattr(source, "url", "unknown"), exc)
    ontology_terms = _world_ontology_terms(ctx.current_world)
    eligible: list[Signal] = []
    poll_filtered_out = 0
    for item in fetched:
        if keywords and not _signal_matches_keywords(item, keywords, min_keyword_matches=max(min_keyword_matches, 1)):
            poll_filtered_out += 1
            continue
        relevance_score, details = _relevance_score(
            item,
            keywords=keywords,
            ontology_terms=ontology_terms,
            hypothesis_terms=set(),
        )
        if relevance_score < min_relevance_score:
            poll_filtered_out += 1
            continue
        item.metadata = {
            **dict(item.metadata),
            "external_relevance": {
                **details,
                "min_relevance_score": min_relevance_score,
                "min_keyword_matches": min_keyword_matches,
            },
        }
        eligible.append(item)
    ctx.stats["filtered_out_count"] += poll_filtered_out
    eligible.sort(key=lambda item: item.timestamp)

    enqueued = 0
    for signal_payload in eligible:
        if enqueued >= int(ctx.args.max_signals_per_poll):
            break
        signal_id = str(signal_payload.signal_id)
        if ctx.cursor_store.is_committed(signal_id) or signal_id in ctx.queued_signal_ids:
            continue
        ctx.pending_signals.append(signal_payload)
        ctx.queued_signal_ids.add(signal_id)
        ctx.stats["signals_seen"] += 1
        enqueued += 1
    if enqueued > 0:
        LOGGER.info("Enqueued %d new signals. queue_len=%d filtered_out_count=%d", enqueued, len(ctx.pending_signals), poll_filtered_out)
        _persist_context(ctx)
    else:
        LOGGER.info("No new eligible signals this poll. filtered_out_count=%d", poll_filtered_out)
    return enqueued


def _process_one_signal(signal_payload: Signal, *, ctx: RuntimeContext) -> SignalResult:
    result = SignalResult()
    signal_id = str(signal_payload.signal_id)
    ctx.queued_signal_ids.discard(signal_id)
    if ctx.cursor_store.is_committed(signal_id):
        return result
    ctx.current_world.runtime_step += 1
    agent_min_relevance_score = float(ctx.filter_cfg.get("agent_min_relevance_score", ctx.filter_cfg.get("min_relevance_score", 0.0)))

    if _self_calibration_started(ctx.pipeline):
        agent_score, agent_details = _relevance_score(
            signal_payload,
            keywords=ctx.keywords,
            ontology_terms=_world_ontology_terms(ctx.current_world),
            hypothesis_terms=_active_hypothesis_terms(ctx.pipeline.conscious_state),
        )
        stream_relevance_stats = _update_stream_relevance_stats(
            ctx.pipeline.conscious_state,
            score=agent_score,
            accepted=agent_score >= agent_min_relevance_score,
        )
        engine = ConsciousnessEngine(ctx.pipeline.conscious_state, ctx.pipeline.consciousness_config)
        ctx.pipeline.conscious_state = engine.refresh_after_runtime_feedback(
            world_ref=f"world:{ctx.current_world.domain_id}:{ctx.current_world.t}",
            runtime_metadata={
                "stream_relevance": stream_relevance_stats,
                "last_runtime_step": int(ctx.current_world.runtime_step),
            },
        )
        if agent_score < agent_min_relevance_score:
            ontology_overlap = int(agent_details.get("ontology_overlap", 0))
            hypothesis_overlap = int(agent_details.get("hypothesis_overlap", 0))
            if ontology_overlap == 0 and hypothesis_overlap == 0:
                anomaly_node_id = ctx.pipeline.record_anomaly_candidate(
                    signal_id=signal_id,
                    signal_text=signal_payload.text,
                    signal_topic=signal_payload.topic,
                    runtime_step=int(ctx.current_world.runtime_step),
                )
                result.processed = 1
                result.verified_forecasts += _verify_due_forecasts(
                    pipeline=ctx.pipeline,
                    state=ctx.pipeline.conscious_state,
                    event_log=ctx.event_log,
                    logged_event_ids=ctx.logged_event_ids,
                    current_world=ctx.current_world,
                    current_probs={outcome_id: float(probability) for outcome_id, probability in score_outcomes(ctx.current_world).items()},
                    current_signal_id=signal_id,
                )
                runtime_event = _runtime_trace_for_signal(
                    state=ctx.pipeline.conscious_state,
                    signal_payload=signal_payload,
                    trigger_mode="ANOMALY_CANDIDATE",
                    llm_used=False,
                    updated_world=False,
                    update_error=None,
                    extra_diff={
                        "filtered_out": False,
                        "anomaly_candidate": True,
                        "anomaly_candidate_node_id": anomaly_node_id,
                        "filter_phase": "agent",
                        "relevance_score": agent_score,
                        "agent_min_relevance_score": agent_min_relevance_score,
                        "relevance_details": agent_details,
                        "external_relevance": dict(signal_payload.metadata.get("external_relevance", {})),
                    },
                )
                ctx.pipeline.conscious_state.trace_state.append(runtime_event)
                ctx.event_log.append(runtime_event)
                ctx.logged_event_ids.add(runtime_event.event_id)
                ctx.cursor_store.commit(signal_id)
                _persist_context(ctx)
                _write_kg_snapshot(
                    ctx,
                    reason="signal_anomaly_candidate",
                    signal_payload=signal_payload,
                    trigger_mode="ANOMALY_CANDIDATE",
                    extra_metadata={"anomaly_candidate_node_id": anomaly_node_id},
                )
                return result

            result.processed = 1
            result.filtered_out = 1
            runtime_event = _runtime_trace_for_signal(
                state=ctx.pipeline.conscious_state,
                signal_payload=signal_payload,
                trigger_mode="FILTERED_OUT",
                llm_used=False,
                updated_world=False,
                update_error=None,
                extra_diff={
                    "filtered_out": True,
                    "filter_phase": "agent",
                    "relevance_score": agent_score,
                    "agent_min_relevance_score": agent_min_relevance_score,
                    "relevance_details": agent_details,
                    "external_relevance": dict(signal_payload.metadata.get("external_relevance", {})),
                },
            )
            ctx.pipeline.conscious_state.trace_state.append(runtime_event)
            ctx.event_log.append(runtime_event)
            ctx.logged_event_ids.add(runtime_event.event_id)
            ctx.cursor_store.commit(signal_id)
            _persist_context(ctx)
            _write_kg_snapshot(
                ctx,
                reason="signal_filtered_out",
                signal_payload=signal_payload,
                trigger_mode="FILTERED_OUT",
                extra_metadata={"filter_phase": "agent"},
            )
            return result

    triggers = ctx.ingestion_engine.ingest(
        ManualSignalSource([signal_payload]),
        classifier=ctx.llm_client,
        signal_memory=ctx.signal_memory,
        skip_duplicates_within_hours=1.0,
    )
    if not triggers:
        result.processed = 1
        runtime_event = _runtime_trace_for_signal(
            state=ctx.pipeline.conscious_state,
            signal_payload=signal_payload,
            trigger_mode="WATCH",
            llm_used=False,
            updated_world=False,
            update_error=None,
            extra_diff={"external_relevance": dict(signal_payload.metadata.get("external_relevance", {}))},
        )
        ctx.pipeline.conscious_state.trace_state.append(runtime_event)
        ctx.event_log.append(runtime_event)
        ctx.logged_event_ids.add(runtime_event.event_id)
        result.verified_forecasts += _verify_due_forecasts(
            pipeline=ctx.pipeline,
            state=ctx.pipeline.conscious_state,
            event_log=ctx.event_log,
            logged_event_ids=ctx.logged_event_ids,
            current_world=ctx.current_world,
            current_probs={outcome_id: float(probability) for outcome_id, probability in score_outcomes(ctx.current_world).items()},
            current_signal_id=signal_id,
        )
        ctx.cursor_store.commit(signal_id)
        _persist_context(ctx)
        _write_kg_snapshot(
            ctx,
            reason="signal_watch",
            signal_payload=signal_payload,
            trigger_mode="WATCH",
            extra_metadata={"world_updated": False},
        )
        return result

    trigger = triggers[0]
    should_update = trigger.mode in {"ANALYZE", "DEEP_DIVE"}
    llm_update_attempted = should_update
    update_error: str | None = None
    verification_probs: dict[str, float] | None = None
    if should_update:
        try:
            current_world_before_update = ctx.current_world.clone()
            parameter_vector = ctx.estimator.estimate(ctx.current_world, signal_payload.text)
            try:
                pipeline_result = ctx.pipeline.update(
                    ctx.current_world,
                    parameter_vector,
                    signal_text=signal_payload.text,
                    signal_id=signal_id,
                )
                ctx.current_world = pipeline_result.world.clone()
                result.updated += 1
            except Exception as primary_exc:  # noqa: BLE001
                LOGGER.warning("Primary update failed for signal_id=%s: %s; retrying from base world.", signal_id, primary_exc)
                fallback_world = ctx.base_world_template.clone()
                # Preserve the live runtime/world clocks so fallback updates do not
                # rewind the longitudinal trace even if they restart from a safe schema.
                fallback_world.runtime_step = current_world_before_update.runtime_step
                fallback_world.t = current_world_before_update.t
                fallback_world.seed = current_world_before_update.seed
                pipeline_result = ctx.pipeline.update(
                    fallback_world,
                    parameter_vector,
                    signal_text=signal_payload.text,
                    signal_id=signal_id,
                )
                if int(pipeline_result.world.t) < int(current_world_before_update.t):
                    raise RuntimeError(
                        "fallback_update_regressed_world_step: "
                        f"{pipeline_result.world.t} < {current_world_before_update.t}"
                    )
                ctx.current_world = pipeline_result.world.clone()
                result.updated += 1
                update_error = f"primary_update_failed: {primary_exc}; fallback=base_world"

            verification_probs = {
                key: float(value)
                for key, value in pipeline_result.simulation.get("final_outcome_probs", {}).items()
            }
            result.verified_forecasts += _verify_due_forecasts(
                pipeline=ctx.pipeline,
                state=ctx.pipeline.conscious_state,
                event_log=ctx.event_log,
                logged_event_ids=ctx.logged_event_ids,
                current_world=ctx.current_world,
                current_probs=verification_probs,
                current_signal_id=signal_id,
            )
            engine = ConsciousnessEngine(ctx.pipeline.conscious_state, ctx.pipeline.consciousness_config)
            engine.maybe_deliberate(_utc_now())
            ctx.pipeline.conscious_state = engine.state
        except Exception as exc:  # noqa: BLE001
            result.update_failures += 1
            update_error = str(exc)
            should_update = False
            LOGGER.warning("World update failed for signal_id=%s: %s", signal_id, exc)
    elif not ctx.args.include_watch:
        result.skipped_watch += 1

    if verification_probs is None:
        result.verified_forecasts += _verify_due_forecasts(
            pipeline=ctx.pipeline,
            state=ctx.pipeline.conscious_state,
            event_log=ctx.event_log,
            logged_event_ids=ctx.logged_event_ids,
            current_world=ctx.current_world,
            current_probs={outcome_id: float(probability) for outcome_id, probability in score_outcomes(ctx.current_world).items()},
            current_signal_id=signal_id,
        )

    _append_unlogged_trace_events(ctx.pipeline.conscious_state, ctx.event_log, ctx.logged_event_ids)
    runtime_event = _runtime_trace_for_signal(
        state=ctx.pipeline.conscious_state,
        signal_payload=signal_payload,
        trigger_mode=trigger.mode,
        llm_used=llm_update_attempted,
        updated_world=should_update,
        update_error=update_error,
        extra_diff={
            "filtered_out": False,
            "external_relevance": dict(signal_payload.metadata.get("external_relevance", {})),
        },
    )
    ctx.pipeline.conscious_state.trace_state.append(runtime_event)
    ctx.event_log.append(runtime_event)
    ctx.logged_event_ids.add(runtime_event.event_id)
    ctx.cursor_store.commit(signal_id)
    _persist_context(ctx)
    _write_kg_snapshot(
        ctx,
        reason="signal_processed",
        signal_payload=signal_payload,
        trigger_mode=trigger.mode,
        extra_metadata={
            "world_updated": bool(should_update),
            "update_error": update_error,
        },
    )
    result.processed = 1
    LOGGER.info(
        "Processed signal_id=%s mode=%s world_t=%s runtime_step=%s queue_len=%d",
        signal_id,
        trigger.mode,
        ctx.current_world.t,
        ctx.current_world.runtime_step,
        len(ctx.pending_signals),
    )
    return result


def _run_loop(
    ctx: RuntimeContext,
) -> LoopSummary:
    next_poll_at = ctx.started_at
    while not ctx.stop_requested:
        now = _utc_now()
        if ctx.deadline is not None and now >= ctx.deadline:
            LOGGER.info("Reached duration limit (hours=%s).", ctx.args.hours)
            break
        if now >= next_poll_at:
            _run_poll(ctx.sources, ctx)
            next_poll_at = now + timedelta(seconds=float(ctx.poll_seconds))
        if ctx.pending_signals:
            signal_result = _process_one_signal(ctx.pending_signals.pop(0), ctx=ctx)
            ctx.stats["signals_committed"] += signal_result.processed
            ctx.stats["world_updates"] += signal_result.updated
            ctx.stats["world_update_failures"] += signal_result.update_failures
            ctx.stats["verified_forecasts"] += signal_result.verified_forecasts
            ctx.stats["watch_skipped"] += signal_result.skipped_watch
            ctx.stats["filtered_out_count"] += signal_result.filtered_out
            if _check_and_handle_repair_request(ctx):
                continue
            continue
        engine = ConsciousnessEngine(ctx.pipeline.conscious_state, ctx.pipeline.consciousness_config)
        if engine.maybe_deliberate(now) is not None:
            ctx.stats["idle_deliberations"] += 1
            ctx.pipeline.conscious_state = engine.state
            if _check_and_handle_repair_request(ctx):
                continue
            _persist_context(ctx)
            continue
        sleep_seconds = ctx.analysis_interval_seconds
        if ctx.deadline is not None:
            sleep_seconds = min(sleep_seconds, max((ctx.deadline - _utc_now()).total_seconds(), 0.0))
        sleep_seconds = min(sleep_seconds, max((next_poll_at - _utc_now()).total_seconds(), 0.0))
        if sleep_seconds > 0.0:
            time.sleep(sleep_seconds)
    return LoopSummary(
        started_at=ctx.started_at,
        ended_at=_utc_now(),
        hours_requested=float(ctx.args.hours),
        bootstrap_mode=ctx.bootstrap_mode,
        bootstrap_package_path=str(ctx.package_path) if ctx.package_path.exists() else None,
        model=ctx.model_name,
        llm_provider=ctx.provider,
        runtime_path=str(ctx.runtime_path),
        event_log_path=str(ctx.event_log.path),
        signals_seen=int(ctx.stats["signals_seen"]),
        signals_committed=int(ctx.stats["signals_committed"]),
        world_updates=int(ctx.stats["world_updates"]),
        world_update_failures=int(ctx.stats["world_update_failures"]),
        verified_forecasts=int(ctx.stats["verified_forecasts"]),
        runtime_step=int(ctx.current_world.runtime_step),
        idle_deliberations=int(ctx.stats["idle_deliberations"]),
        queue_len=len(ctx.pending_signals),
        watch_skipped=int(ctx.stats["watch_skipped"]),
        filtered_out_count=int(ctx.stats["filtered_out_count"]),
        trace_events=len(ctx.pipeline.conscious_state.trace_state),
    )


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
    paths = _resolve_runtime_paths(args, config, config_path)
    if args.query is not None:
        return _handle_query_mode(args, config, paths)
    storage = _initialize_runtime_storage(args, paths.runtime_path, paths.event_log_path)
    bootstrap = _bootstrap(args=args, config=config, paths=paths, storage=storage)
    ctx = _build_runtime_context(
        args=args,
        config=config,
        paths=paths,
        storage=storage,
        bootstrap=bootstrap,
        default_sources=default_sources,
        default_keywords=default_keywords,
    )
    _write_kg_snapshot(
        ctx,
        reason="bootstrap",
        extra_metadata={"bootstrap_mode": ctx.bootstrap_mode, "resumed": bool(args.resume)},
    )
    previous_sigint, previous_sigterm = _install_stop_handlers(ctx)
    try:
        summary = _run_loop(ctx)
    finally:
        _restore_stop_handlers(previous_sigint, previous_sigterm)
        _persist_context(ctx)
    print(json.dumps(summary.to_dict(), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
