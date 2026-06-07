"""Run Freeman against configurable signal streams with checkpoint/resume."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from freeman.agent.analysispipeline import AnalysisPipeline
from freeman.agent.parameterestimator import ParameterEstimator
from freeman.agent.signalingestion import SignalIngestionEngine, SignalMemory
from freeman.logging_config import configure_logging
from freeman.runtime import bootstrap as _bootstrap_module
from freeman.runtime import lifecycle as _lifecycle_module
from freeman.runtime import signal_loop as _signal_loop_module
from freeman.runtime.bootstrap import FreemanOrchestrator, _build_chat_client
from freeman.runtime.lifecycle import (
    DEFAULT_RUNTIME_CONFIG,
    BootstrapResult,
    LoopSummary,
    RuntimeContext,
    RuntimePaths,
    RuntimeStorage,
    SignalResult,
    _budget_summary_for_runtime,
    _build_runtime_context,
    _initial_runtime_stats,
    _initialize_runtime_storage,
    _install_stop_handlers,
    _load_yaml,
    _resolve_runtime_paths,
    _restore_stop_handlers,
    _source_configs,
    _utc_now,
    _write_kg_snapshot,
    build_signal_source,
)
from freeman.runtime.query_handlers import _handle_query_mode
from freeman.runtime.signal_loop import (
    _normalize_relation_candidates,
    _run_loop,
    _verify_due_forecasts,
)
from freeman.runtime.startup_checks import validate_config


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
    parser.add_argument("--plain-logs", action="store_true")
    return parser


_bootstrap = _bootstrap_module._bootstrap


def _sync_patchable_symbols() -> None:
    _bootstrap_module._build_chat_client = _build_chat_client
    _bootstrap_module.FreemanOrchestrator = FreemanOrchestrator
    _bootstrap_module.AnalysisPipeline = AnalysisPipeline
    _bootstrap_module.ParameterEstimator = ParameterEstimator
    _lifecycle_module._source_configs = _source_configs
    _lifecycle_module.build_signal_source = build_signal_source
    _signal_loop_module._bootstrap = _bootstrap


def _trigger_ontology_repair(*args: Any, **kwargs: Any) -> bool:
    _sync_patchable_symbols()
    return _signal_loop_module._trigger_ontology_repair(*args, **kwargs)


def _process_one_signal(*args: Any, **kwargs: Any) -> SignalResult:
    _sync_patchable_symbols()
    return _signal_loop_module._process_one_signal(*args, **kwargs)


def main(
    argv: list[str] | None = None,
    *,
    default_config_path: str = "config.yaml",
    default_sources: list[dict[str, Any]] | None = None,
    default_keywords: list[str] | None = None,
) -> int:
    config_default = os.getenv("FREEMAN_CONFIG", default_config_path)
    args = _build_parser(default_config_path=config_default).parse_args(argv)
    run_id = configure_logging(
        level=args.log_level,
        json_logs=not bool(args.plain_logs),
        force=True,
    )
    _sync_patchable_symbols()
    config_path = Path(args.config_path).resolve()
    config = _load_yaml(config_path)
    paths = _resolve_runtime_paths(args, config, config_path)
    if args.query is not None:
        return _handle_query_mode(args, config, paths)
    startup_errors = validate_config(config, config_base=config_path.parent)
    if startup_errors:
        raise RuntimeError("Runtime startup validation failed: " + "; ".join(startup_errors))
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
    ctx.run_id = str(run_id or "")
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
        _lifecycle_module._persist_context(ctx)
    print(json.dumps(summary.to_dict(), indent=2, sort_keys=True))
    return 0


__all__ = [
    "AnalysisPipeline",
    "BootstrapResult",
    "DEFAULT_RUNTIME_CONFIG",
    "FreemanOrchestrator",
    "LoopSummary",
    "ParameterEstimator",
    "RuntimeContext",
    "RuntimePaths",
    "RuntimeStorage",
    "SignalIngestionEngine",
    "SignalMemory",
    "SignalResult",
    "_bootstrap",
    "_budget_summary_for_runtime",
    "_build_chat_client",
    "_initial_runtime_stats",
    "_normalize_relation_candidates",
    "_process_one_signal",
    "_source_configs",
    "_trigger_ontology_repair",
    "_utc_now",
    "_verify_due_forecasts",
    "build_signal_source",
    "main",
]


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
