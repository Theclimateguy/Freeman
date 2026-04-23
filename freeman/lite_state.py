"""Persistence helpers for the Freeman lite runtime."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

from freeman.agent.forecastregistry import ForecastRegistry
from freeman.core.world import WorldState
from freeman.lite_config import LiteConfig
from freeman.memory.knowledgegraph import KnowledgeGraph
from freeman.utils import json_ready


def _now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


@dataclass(slots=True)
class RuntimeBundle:
    config: LiteConfig
    knowledge_graph: KnowledgeGraph
    forecast_registry: ForecastRegistry
    world_state: WorldState | None


def ensure_storage(config: LiteConfig) -> None:
    for path in (
        config.paths.kg_state,
        config.paths.forecasts,
        config.paths.world_state,
        config.paths.error_log,
    ):
        path.parent.mkdir(parents=True, exist_ok=True)


def load_world_state(path: str | Path) -> WorldState | None:
    source = Path(path).resolve()
    if not source.exists():
        return None
    payload = json.loads(source.read_text(encoding="utf-8"))
    return WorldState.from_snapshot(payload)


def save_world_state(world: WorldState, path: str | Path) -> Path:
    target = Path(path).resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    temp_path = target.with_suffix(f"{target.suffix}.tmp")
    temp_path.write_text(json.dumps(world.snapshot(), indent=2, sort_keys=True), encoding="utf-8")
    temp_path.replace(target)
    return target


def load_runtime_bundle(config: LiteConfig) -> RuntimeBundle:
    ensure_storage(config)
    knowledge_graph = KnowledgeGraph(
        json_path=config.paths.kg_state,
        auto_load=config.paths.kg_state.exists(),
        auto_save=False,
    )
    forecast_registry = ForecastRegistry(
        json_path=config.paths.forecasts,
        auto_load=config.paths.forecasts.exists(),
        auto_save=False,
    )
    world_state = load_world_state(config.paths.world_state)
    return RuntimeBundle(
        config=config,
        knowledge_graph=knowledge_graph,
        forecast_registry=forecast_registry,
        world_state=world_state,
    )


def append_error_log(path: str | Path, entry: dict[str, Any]) -> Path:
    target = Path(path).resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = {"timestamp": _now_iso(), **json_ready(entry)}
    with target.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")
    return target


def runtime_counters(world: WorldState | None) -> dict[str, int]:
    if world is None:
        return {"llm_calls_total": 0, "simulation_steps_total": 0}
    runtime_payload = world.metadata.get("lite_runtime", {})
    if not isinstance(runtime_payload, dict):
        runtime_payload = {}
    return {
        "llm_calls_total": int(runtime_payload.get("llm_calls_total", 0)),
        "simulation_steps_total": int(runtime_payload.get("simulation_steps_total", 0)),
    }


def bump_runtime_counters(
    world: WorldState,
    *,
    llm_calls: int = 0,
    simulation_steps: int = 0,
) -> dict[str, int]:
    counters = runtime_counters(world)
    counters["llm_calls_total"] += max(int(llm_calls), 0)
    counters["simulation_steps_total"] += max(int(simulation_steps), 0)
    world.metadata["lite_runtime"] = dict(counters)
    return counters


def projected_counters(
    world: WorldState | None,
    *,
    llm_calls: int = 0,
    simulation_steps: int = 0,
) -> dict[str, int]:
    counters = runtime_counters(world)
    counters["llm_calls_total"] += max(int(llm_calls), 0)
    counters["simulation_steps_total"] += max(int(simulation_steps), 0)
    return counters


__all__ = [
    "RuntimeBundle",
    "append_error_log",
    "bump_runtime_counters",
    "ensure_storage",
    "load_runtime_bundle",
    "load_world_state",
    "projected_counters",
    "runtime_counters",
    "save_world_state",
]
