"""Configuration helpers for the Freeman lite runtime."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


def _resolve_path(base: Path, candidate: str | None, default: str) -> Path:
    target = Path(candidate or default).expanduser()
    return target if target.is_absolute() else (base / target).resolve()


def _merge_dicts(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _merge_dicts(merged[key], value)
        else:
            merged[key] = value
    return merged


DEFAULT_CONFIG: dict[str, Any] = {
    "paths": {
        "kg_state": "./data/kg_state.json",
        "forecasts": "./data/forecasts.json",
        "world_state": "./data/world_state.json",
        "error_log": "./data/errors.jsonl",
    },
    "llm": {
        "provider": "ollama",
        "model": "qwen2.5-coder:14b",
        "base_url": "http://127.0.0.1:11434",
        "timeout_seconds": 90.0,
    },
    "limits": {
        "max_llm_calls": 50,
        "max_simulation_steps": 5000,
    },
    "signals": {
        "keywords": [],
        "min_keyword_hits": 1,
        "conflict_threshold": 0.25,
        "query_top_k": 10,
        "max_signal_history": 256,
    },
}


@dataclass(slots=True)
class LitePaths:
    kg_state: Path
    forecasts: Path
    world_state: Path
    error_log: Path


@dataclass(slots=True)
class LiteLLMConfig:
    provider: str = "ollama"
    model: str = "qwen2.5-coder:14b"
    base_url: str = "http://127.0.0.1:11434"
    timeout_seconds: float = 90.0


@dataclass(slots=True)
class LiteLimits:
    max_llm_calls: int = 50
    max_simulation_steps: int = 5000


@dataclass(slots=True)
class LiteSignalConfig:
    keywords: tuple[str, ...] = field(default_factory=tuple)
    min_keyword_hits: int = 1
    conflict_threshold: float = 0.25
    query_top_k: int = 10
    max_signal_history: int = 256


@dataclass(slots=True)
class LiteConfig:
    paths: LitePaths
    llm: LiteLLMConfig = field(default_factory=LiteLLMConfig)
    limits: LiteLimits = field(default_factory=LiteLimits)
    signals: LiteSignalConfig = field(default_factory=LiteSignalConfig)
    sim_max_steps: int = 50
    dt: float = 1.0

    def snapshot(self) -> dict[str, Any]:
        return {
            "paths": {
                "kg_state": str(self.paths.kg_state),
                "forecasts": str(self.paths.forecasts),
                "world_state": str(self.paths.world_state),
                "error_log": str(self.paths.error_log),
            },
            "llm": {
                "provider": self.llm.provider,
                "model": self.llm.model,
                "base_url": self.llm.base_url,
                "timeout_seconds": float(self.llm.timeout_seconds),
            },
            "limits": {
                "max_llm_calls": int(self.limits.max_llm_calls),
                "max_simulation_steps": int(self.limits.max_simulation_steps),
            },
            "signals": {
                "keywords": list(self.signals.keywords),
                "min_keyword_hits": int(self.signals.min_keyword_hits),
                "conflict_threshold": float(self.signals.conflict_threshold),
                "query_top_k": int(self.signals.query_top_k),
                "max_signal_history": int(self.signals.max_signal_history),
            },
        }


def default_config_path() -> Path:
    return (Path(__file__).resolve().parents[1] / "config.yaml").resolve()


def load_config(config_path: str | Path | None = None) -> LiteConfig:
    resolved = Path(config_path).expanduser().resolve() if config_path is not None else default_config_path()
    payload = dict(DEFAULT_CONFIG)
    if resolved.exists():
        loaded = yaml.safe_load(resolved.read_text(encoding="utf-8")) or {}
        payload = _merge_dicts(payload, loaded)

    paths_cfg = payload.get("paths", {})
    llm_cfg = payload.get("llm", {})
    limits_cfg = payload.get("limits", {})
    signals_cfg = payload.get("signals", {})
    base_dir = resolved.parent
    return LiteConfig(
        paths=LitePaths(
            kg_state=_resolve_path(base_dir, paths_cfg.get("kg_state"), "./data/kg_state.json"),
            forecasts=_resolve_path(base_dir, paths_cfg.get("forecasts"), "./data/forecasts.json"),
            world_state=_resolve_path(base_dir, paths_cfg.get("world_state"), "./data/world_state.json"),
            error_log=_resolve_path(base_dir, paths_cfg.get("error_log"), "./data/errors.jsonl"),
        ),
        llm=LiteLLMConfig(
            provider=str(llm_cfg.get("provider", "ollama")).strip().lower(),
            model=str(llm_cfg.get("model", "qwen2.5-coder:14b")).strip(),
            base_url=str(llm_cfg.get("base_url", "http://127.0.0.1:11434")).strip(),
            timeout_seconds=float(llm_cfg.get("timeout_seconds", 90.0)),
        ),
        limits=LiteLimits(
            max_llm_calls=max(int(limits_cfg.get("max_llm_calls", 50)), 0),
            max_simulation_steps=max(int(limits_cfg.get("max_simulation_steps", 5000)), 0),
        ),
        signals=LiteSignalConfig(
            keywords=tuple(str(item).strip().lower() for item in signals_cfg.get("keywords", []) if str(item).strip()),
            min_keyword_hits=max(int(signals_cfg.get("min_keyword_hits", 1)), 0),
            conflict_threshold=float(signals_cfg.get("conflict_threshold", 0.25)),
            query_top_k=max(int(signals_cfg.get("query_top_k", 10)), 1),
            max_signal_history=max(int(signals_cfg.get("max_signal_history", 256)), 1),
        ),
    )


__all__ = [
    "DEFAULT_CONFIG",
    "LiteConfig",
    "LiteLLMConfig",
    "LiteLimits",
    "LitePaths",
    "LiteSignalConfig",
    "default_config_path",
    "load_config",
]
