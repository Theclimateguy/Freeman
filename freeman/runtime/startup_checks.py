"""Fail-fast runtime startup validation."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any


def _resolve_path(base: Path, candidate: str | None, default: str) -> Path:
    target = Path(candidate or default).expanduser()
    return target if target.is_absolute() else (base / target).resolve()


def _nearest_existing_parent(path: Path) -> Path | None:
    current = path if path.exists() and path.is_dir() else path.parent
    while not current.exists() and current != current.parent:
        current = current.parent
    return current if current.exists() else None


def _writable_target(path: Path) -> bool:
    parent = _nearest_existing_parent(path)
    return parent is not None and os.access(parent, os.W_OK | os.X_OK)


def validate_config(config: dict[str, Any], *, config_base: str | Path | None = None) -> list[str]:
    """Return critical startup errors before the daemon enters the main loop."""

    errors: list[str] = []
    base = Path(config_base or ".").expanduser().resolve()
    agent_cfg = dict(config.get("agent", {}) or {})
    runtime_cfg = dict(config.get("runtime", {}) or {})
    memory_cfg = dict(config.get("memory", {}) or {})

    budget = float(agent_cfg.get("budget_usd_per_day", 0.5) if agent_cfg.get("budget_usd_per_day", None) is not None else 0.5)
    governance_budget = float(dict(agent_cfg.get("cost_governance", {}) or {}).get("max_compute_budget_per_session", budget) or 0.0)
    if budget <= 0.0 or governance_budget <= 0.0:
        errors.append("agent.budget_usd_per_day and cost_governance.max_compute_budget_per_session must be > 0.")

    runtime_path = _resolve_path(base, runtime_cfg.get("runtime_path"), "./data/runtime")
    memory_backend = str(memory_cfg.get("backend", "json") or "json").strip().lower()
    if memory_backend == "networkx-json":
        memory_backend = "json"
    kg_path = _resolve_path(
        base,
        memory_cfg.get("sqlite_path") if memory_backend == "sqlite" else memory_cfg.get("json_path"),
        "./data/kg.db" if memory_backend == "sqlite" else "./data/kg_state.json",
    )
    if not _writable_target(runtime_path):
        errors.append(f"runtime.runtime_path is not writable: {runtime_path}")
    if memory_backend not in {"json", "sqlite"}:
        errors.append(f"Unsupported memory backend '{memory_backend}'.")
    if not _writable_target(kg_path):
        field = "sqlite_path" if memory_backend == "sqlite" else "json_path"
        errors.append(f"memory.{field} parent is not writable: {kg_path.parent}")

    llm_cfg = dict(config.get("llm", {}) or {})
    provider = str(llm_cfg.get("provider", "none") or "none").strip().lower()
    if provider in {"", "none", "ollama"}:
        return errors
    configured_key = str(llm_cfg.get("api_key", "") or "").strip()
    if provider == "deepseek":
        key = configured_key or os.getenv("DEEPSEEK_API_KEY", "").strip() or os.getenv("LLM_API_KEY", "").strip()
        if not key:
            errors.append("LLM provider 'deepseek' requires DEEPSEEK_API_KEY or LLM_API_KEY.")
        return errors
    if provider in {"openai", "openai-compatible", "openai_compatible"}:
        key = configured_key or os.getenv("OPENAI_API_KEY", "").strip() or os.getenv("LLM_API_KEY", "").strip()
        if not key:
            errors.append(f"LLM provider '{provider}' requires OPENAI_API_KEY or LLM_API_KEY.")
        return errors
    errors.append(f"Unsupported LLM provider '{provider}'.")
    return errors


__all__ = ["validate_config"]
