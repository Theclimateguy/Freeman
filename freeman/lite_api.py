"""Public Python API for the Freeman lite runtime."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from freeman.agent.analysispipeline import AnalysisPipeline, AnalysisPipelineConfig
from freeman.agent.parameterestimator import ParameterEstimator
from freeman.agent.signalingestion import SignalIngestionEngine
from freeman.core.types import ParameterVector
from freeman.game.runner import SimConfig
from freeman.interface.factory import build_chat_client
from freeman.interface.kgexport import KnowledgeGraphExporter
from freeman.lite_config import LiteConfig, load_config
from freeman.lite_state import (
    append_error_log,
    bump_runtime_counters,
    load_runtime_bundle,
    projected_counters,
    save_world_state,
)
from freeman.llm.orchestrator import FreemanOrchestrator
from freeman.runtime.queryengine import RuntimeQueryEngine


def _load_schema_or_brief(value: str | Path | dict[str, Any]) -> tuple[dict[str, Any] | None, str | None]:
    if isinstance(value, dict):
        return value, None
    candidate = Path(value).expanduser()
    if candidate.exists():
        suffix = candidate.suffix.lower()
        if suffix in {".json", ".yaml", ".yml"}:
            text = candidate.read_text(encoding="utf-8")
            if suffix in {".yaml", ".yml"}:
                return yaml.safe_load(text), None
            import json

            return json.loads(text), None
        return None, candidate.read_text(encoding="utf-8")
    return None, str(value)


def _load_signal_text(value: str | Path) -> str:
    candidate = Path(value).expanduser()
    if candidate.exists():
        return candidate.read_text(encoding="utf-8")
    return str(value)


def _enforce_limits(config: LiteConfig, world, *, llm_calls: int = 0, simulation_steps: int = 0) -> None:
    counters = projected_counters(world, llm_calls=llm_calls, simulation_steps=simulation_steps)
    if counters["llm_calls_total"] > config.limits.max_llm_calls:
        raise RuntimeError(
            f"LLM call budget exceeded: {counters['llm_calls_total']} > {config.limits.max_llm_calls}"
        )
    if counters["simulation_steps_total"] > config.limits.max_simulation_steps:
        raise RuntimeError(
            "Simulation step budget exceeded: "
            f"{counters['simulation_steps_total']} > {config.limits.max_simulation_steps}"
        )


def _build_pipeline(config: LiteConfig, bundle) -> AnalysisPipeline:
    return AnalysisPipeline(
        sim_config=SimConfig(max_steps=config.sim_max_steps, dt=config.dt),
        knowledge_graph=bundle.knowledge_graph,
        forecast_registry=bundle.forecast_registry,
        config=AnalysisPipelineConfig(
            probability_conflict_threshold=config.signals.conflict_threshold,
            forecast_horizon_steps=config.sim_max_steps,
        ),
    )


def compile(
    domain_input: str | Path | dict[str, Any],
    *,
    config_path: str | Path | None = None,
    verify_level2: bool = False,
) -> dict[str, Any]:
    config = load_config(config_path)
    bundle = load_runtime_bundle(config)
    pipeline = _build_pipeline(config, bundle)
    schema, brief = _load_schema_or_brief(domain_input)
    llm_calls = 0
    synthesis: dict[str, Any] | None = None

    try:
        if schema is None:
            _enforce_limits(config, bundle.world_state, llm_calls=2, simulation_steps=config.sim_max_steps)
            client, error = build_chat_client(config)
            if client is None:
                raise RuntimeError(error or "LLM client is not configured.")
            orchestrator = FreemanOrchestrator(client)
            package, world_id, attempts, repair_history = orchestrator.compile_and_repair(
                brief or "",
                max_repairs=1,
                verify_level2=verify_level2,
            )
            llm_calls = attempts
            schema = package["schema"]
            synthesis = {
                "world_id": world_id,
                "assumptions": package.get("assumptions", []),
                "repair_history": repair_history,
                "attempts": attempts,
            }
        else:
            _enforce_limits(config, bundle.world_state, simulation_steps=config.sim_max_steps)

        result = pipeline.run(
            schema,
            verify_level2=verify_level2,
            source_text=brief,
            assumptions=synthesis.get("assumptions", []) if synthesis else (),
        )
        counters = bump_runtime_counters(
            result.world,
            llm_calls=llm_calls,
            simulation_steps=int(result.metadata.get("steps_run", 0)),
        )
        save_world_state(result.world, config.paths.world_state)
        bundle.knowledge_graph.save()
        bundle.forecast_registry.save()
        payload = result.snapshot()
        payload.update(
            {
                "status": "compiled",
                "counters": counters,
                "config": config.snapshot(),
            }
        )
        if synthesis is not None:
            payload["synthesis"] = synthesis
        return payload
    except Exception as exc:  # noqa: BLE001
        append_error_log(
            config.paths.error_log,
            {"stage": "compile", "error": str(exc), "input": str(domain_input)},
        )
        raise


def update(
    signal_input: str | Path,
    *,
    config_path: str | Path | None = None,
    verify_level2: bool = False,
) -> dict[str, Any]:
    config = load_config(config_path)
    bundle = load_runtime_bundle(config)
    if bundle.world_state is None:
        raise RuntimeError("world_state.json was not found. Compile a base model before calling update().")

    signal_text = _load_signal_text(signal_input)
    engine = SignalIngestionEngine(
        keywords=config.signals.keywords,
        min_keyword_hits=config.signals.min_keyword_hits,
    )
    signal_hashes = bundle.world_state.metadata.get("lite_signal_hashes", [])
    decision = engine.classify(signal_text, processed_hashes=signal_hashes)

    if decision.mode == "WATCH":
        engine.remember(bundle.world_state, decision.signal_hash, max_history=config.signals.max_signal_history)
        save_world_state(bundle.world_state, config.paths.world_state)
        bundle.knowledge_graph.save()
        return {
            "status": "watch",
            "mode": decision.mode,
            "decision": decision.snapshot(),
            "warning": decision.reason,
        }

    _enforce_limits(config, bundle.world_state, llm_calls=1, simulation_steps=config.sim_max_steps)
    client, error = build_chat_client(config)
    if client is None:
        raise RuntimeError(error or "LLM client is not configured.")
    estimator = ParameterEstimator(client)
    parameter_vector: ParameterVector = estimator.estimate(bundle.world_state, signal_text)
    pipeline = _build_pipeline(config, bundle)

    try:
        result = pipeline.update(
            bundle.world_state,
            parameter_vector,
            signal_text=signal_text,
            signal_id=decision.signal_id,
            verify_level2=verify_level2,
        )
        engine.remember(result.world, decision.signal_hash, max_history=config.signals.max_signal_history)
        counters = bump_runtime_counters(
            result.world,
            llm_calls=1,
            simulation_steps=int(result.metadata.get("steps_run", 0)),
        )
        save_world_state(result.world, config.paths.world_state)
        bundle.knowledge_graph.save()
        bundle.forecast_registry.save()
        payload = result.snapshot()
        payload.update(
            {
                "status": "updated",
                "mode": decision.mode,
                "decision": decision.snapshot(),
                "counters": counters,
            }
        )
        return payload
    except Exception as exc:  # noqa: BLE001
        append_error_log(
            config.paths.error_log,
            {
                "stage": "update",
                "signal_id": decision.signal_id,
                "error": str(exc),
            },
        )
        raise


def query(
    query_text: str,
    *,
    config_path: str | Path | None = None,
    top_k: int | None = None,
) -> dict[str, Any]:
    config = load_config(config_path)
    bundle = load_runtime_bundle(config)
    engine = RuntimeQueryEngine(bundle.knowledge_graph)
    result = engine.query(query_text, top_k=top_k or config.signals.query_top_k)
    return result.to_dict()


def export_kg(
    output_path: str | Path,
    *,
    config_path: str | Path | None = None,
    fmt: str = "json",
) -> Path:
    config = load_config(config_path)
    bundle = load_runtime_bundle(config)
    target = Path(output_path).expanduser().resolve()
    exporter = KnowledgeGraphExporter()
    format_name = str(fmt).strip().lower()
    if format_name == "json":
        target.parent.mkdir(parents=True, exist_ok=True)
        bundle.knowledge_graph.save(target)
        return target
    if format_name == "jsonld":
        return exporter.export_json_ld(bundle.knowledge_graph, target)
    if format_name == "dot":
        result = exporter.export_dot(bundle.knowledge_graph, target)
        return result if isinstance(result, Path) else target
    if format_name == "html":
        result = exporter.export_html(bundle.knowledge_graph, target)
        return result if isinstance(result, Path) else target
    raise ValueError(f"Unsupported export format: {fmt}")


__all__ = ["compile", "export_kg", "query", "update"]
