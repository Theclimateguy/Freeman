"""Read-only runtime health checks."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Literal

import yaml

from freeman.agent.costmodel import BudgetLedger, build_budget_policy, budget_tracking_enabled
from freeman.core.world import WorldState

HealthStatus = Literal["ok", "degraded", "error"]


@dataclass
class HealthState:
    """Compact daemon readiness state."""

    last_signal_at: datetime | None
    last_kg_write_at: datetime | None
    world_t: int
    budget_remaining_usd: float
    status: HealthStatus
    runtime_path: str = ""
    kg_path: str = ""
    reasons: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "reasons": list(self.reasons),
            "last_signal_at": self.last_signal_at.isoformat() if self.last_signal_at else None,
            "last_kg_write_at": self.last_kg_write_at.isoformat() if self.last_kg_write_at else None,
            "world_t": int(self.world_t),
            "budget_remaining_usd": float(self.budget_remaining_usd),
            "runtime_path": self.runtime_path,
            "kg_path": self.kg_path,
        }


def _merge_dicts(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _merge_dicts(merged[key], value)
        else:
            merged[key] = value
    return merged


def _resolve_path(base: Path, candidate: str | None, default: str) -> Path:
    target = Path(candidate or default).expanduser()
    return target if target.is_absolute() else (base / target).resolve()


def _load_config(config_path: str | Path) -> tuple[dict[str, Any], Path]:
    path = Path(config_path).expanduser().resolve()
    defaults = {
        "agent": {"budget_usd_per_day": 0.50, "cost_governance": {}},
        "memory": {"backend": "json", "json_path": "./data/kg_state.json", "sqlite_path": "./data/kg.db"},
        "runtime": {"runtime_path": "./data/runtime", "event_log_path": "./data/runtime/event_log.jsonl"},
    }
    if not path.exists():
        return defaults, path
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return _merge_dicts(defaults, dict(payload)), path


def _parse_datetime(value: Any) -> datetime | None:
    if value is None:
        return None
    try:
        text = str(value).replace("Z", "+00:00")
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _last_signal_time(event_log_path: Path) -> datetime | None:
    if not event_log_path.exists():
        return None
    last_seen: datetime | None = None
    for line in event_log_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        diff = event.get("diff", {}) if isinstance(event, dict) else {}
        if isinstance(diff, dict) and not diff.get("signal_id") and event.get("trigger_type") != "signal":
            continue
        candidate = _parse_datetime(event.get("timestamp") or event.get("ts") or diff.get("timestamp"))
        if candidate is not None:
            last_seen = candidate if last_seen is None else max(last_seen, candidate)
    return last_seen


def _world_t(world_state_path: Path) -> int:
    if not world_state_path.exists():
        return -1
    return int(WorldState.from_snapshot(json.loads(world_state_path.read_text(encoding="utf-8"))).t)


def _budget_remaining(config: dict[str, Any], runtime_path: Path) -> float:
    policy = build_budget_policy(config)
    if not budget_tracking_enabled(config):
        return float(policy.max_compute_budget_per_session)
    ledger = BudgetLedger(runtime_path / "cost_ledger.jsonl", policy=policy, auto_load=(runtime_path / "cost_ledger.jsonl").exists())
    return float(ledger.summary()["remaining_usd"])


def get_health(runtime_state: Any) -> HealthState:
    """Build a health state from a mapping or object with runtime paths."""

    def _read_attr(name: str, default: Any) -> Any:
        if isinstance(runtime_state, dict):
            return runtime_state.get(name, default)
        return getattr(runtime_state, name, default)

    config = dict(_read_attr("config", {}) or {})
    runtime_path = Path(_read_attr("runtime_path", "")).resolve()
    kg_path = Path(_read_attr("kg_path", "")).resolve()
    event_log_path = Path(_read_attr("event_log_path", runtime_path / "event_log.jsonl")).resolve()
    world_state_path = runtime_path / "world_state.json"
    reasons: list[str] = []
    status: HealthStatus = "ok"

    if not runtime_path.exists():
        reasons.append("runtime_path_missing")
        status = "error"
    if not kg_path.exists():
        reasons.append("kg_path_missing")
        status = "error"
    world_t = _world_t(world_state_path)
    if world_t < 0:
        reasons.append("world_state_missing")
        status = "error"

    last_signal_at = _last_signal_time(event_log_path)
    stale_seconds = float(config.get("runtime", {}).get("health_signal_stale_seconds", 1800.0))
    if last_signal_at is None:
        reasons.append("no_signal_events")
        if status == "ok":
            status = "degraded"
    elif (datetime.now(timezone.utc) - last_signal_at).total_seconds() > stale_seconds:
        reasons.append("last_signal_stale")
        if status == "ok":
            status = "degraded"

    last_kg_write_at = datetime.fromtimestamp(kg_path.stat().st_mtime, tz=timezone.utc) if kg_path.exists() else None
    budget_remaining = _budget_remaining(config, runtime_path)
    if budget_remaining < float(config.get("runtime", {}).get("health_budget_warning_usd", 0.10)):
        reasons.append("budget_low")
        if status == "ok":
            status = "degraded"

    return HealthState(
        last_signal_at=last_signal_at,
        last_kg_write_at=last_kg_write_at,
        world_t=world_t,
        budget_remaining_usd=budget_remaining,
        status=status,
        runtime_path=str(runtime_path),
        kg_path=str(kg_path),
        reasons=reasons,
    )


def health_from_config(config_path: str | Path = "config.yaml") -> HealthState:
    """Load config paths and return the persisted runtime health."""

    config, resolved = _load_config(config_path)
    runtime_cfg = dict(config.get("runtime", {}) or {})
    memory_cfg = dict(config.get("memory", {}) or {})
    runtime_path = _resolve_path(resolved.parent, runtime_cfg.get("runtime_path"), "./data/runtime")
    memory_backend = str(memory_cfg.get("backend", "json") or "json").strip().lower()
    if memory_backend == "networkx-json":
        memory_backend = "json"
    kg_path = _resolve_path(
        resolved.parent,
        memory_cfg.get("sqlite_path") if memory_backend == "sqlite" else memory_cfg.get("json_path"),
        "./data/kg.db" if memory_backend == "sqlite" else "./data/kg_state.json",
    )
    event_log_path = _resolve_path(
        resolved.parent,
        runtime_cfg.get("event_log_path"),
        str(runtime_path / "event_log.jsonl"),
    )
    return get_health(
        {
            "config": config,
            "runtime_path": runtime_path,
            "kg_path": kg_path,
            "event_log_path": event_log_path,
        }
    )


__all__ = ["HealthState", "get_health", "health_from_config"]
