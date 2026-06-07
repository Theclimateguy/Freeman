"""Prometheus-compatible read-only runtime metrics."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import sqlite3
from typing import Any

import yaml

from freeman.agent.costmodel import BudgetLedger, build_budget_policy, budget_tracking_enabled
from freeman.core.world import WorldState


@dataclass(frozen=True)
class MetricSample:
    """One Prometheus metric sample."""

    name: str
    value: float
    labels: dict[str, str] | None = None


_METRIC_HELP: dict[str, tuple[str, str]] = {
    "signals_ingested_total": ("counter", "Runtime signal events persisted in the event log."),
    "signals_filtered_total": ("counter", "Runtime signal events filtered before world update."),
    "ontology_repairs_total": ("counter", "Ontology repair requests observed in the event log."),
    "llm_calls_total": ("counter", "LLM calls inferred from runtime events and budget ledger entries."),
    "llm_errors_total": ("counter", "LLM or provider errors observed in runtime events."),
    "world_t": ("gauge", "Current persisted world domain step."),
    "kg_node_count": ("gauge", "Persisted knowledge-graph node count."),
    "budget_spent_usd": ("gauge", "Budget ledger spend in USD."),
    "budget_remaining_usd": ("gauge", "Remaining configured budget in USD."),
    "active_forecasts": ("gauge", "Pending forecasts awaiting verification."),
    "signal_processing_seconds": ("histogram", "Observed signal processing latency in seconds."),
    "analysis_pipeline_seconds": ("histogram", "Observed analysis pipeline latency in seconds."),
    "llm_call_seconds": ("histogram", "Observed LLM call latency in seconds."),
}

_HISTOGRAM_BUCKETS = (0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0)


def _merge_dicts(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _merge_dicts(merged[key], value)
        else:
            merged[key] = value
    return merged


def _load_config(config_path: str | Path) -> tuple[dict[str, Any], Path]:
    path = Path(config_path).expanduser().resolve()
    defaults = {
        "agent": {"budget_usd_per_day": 0.50, "cost_governance": {}},
        "llm": {"provider": "none"},
        "memory": {"backend": "json", "json_path": "./data/kg_state.json", "sqlite_path": "./data/kg.db"},
        "runtime": {"runtime_path": "./data/runtime", "event_log_path": "./data/runtime/event_log.jsonl"},
    }
    if not path.exists():
        return defaults, path
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return _merge_dicts(defaults, dict(payload)), path


def _resolve_path(base: Path, candidate: str | None, default: str) -> Path:
    target = Path(candidate or default).expanduser()
    return target if target.is_absolute() else (base / target).resolve()


def _kg_path(config: dict[str, Any], *, config_path: Path) -> Path:
    memory_cfg = dict(config.get("memory", {}) or {})
    backend = str(memory_cfg.get("backend", "json") or "json").strip().lower()
    if backend == "networkx-json":
        backend = "json"
    return _resolve_path(
        config_path.parent,
        memory_cfg.get("sqlite_path") if backend == "sqlite" else memory_cfg.get("json_path"),
        "./data/kg.db" if backend == "sqlite" else "./data/kg_state.json",
    )


def _read_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return default


def _read_events(event_log_path: Path) -> list[dict[str, Any]]:
    if not event_log_path.exists():
        return []
    events: list[dict[str, Any]] = []
    for line in event_log_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(event, dict):
            events.append(event)
    return events


def _world_t(runtime_path: Path) -> int:
    payload = _read_json(runtime_path / "world_state.json", None)
    if not isinstance(payload, dict):
        return -1
    try:
        return int(WorldState.from_snapshot(payload).t)
    except Exception:
        return -1


def _kg_node_count(config: dict[str, Any], kg_path: Path) -> int:
    if not kg_path.exists():
        return 0
    memory_cfg = dict(config.get("memory", {}) or {})
    backend = str(memory_cfg.get("backend", "json") or "json").strip().lower()
    if backend == "networkx-json":
        backend = "json"
    if backend == "sqlite":
        try:
            with sqlite3.connect(f"file:{kg_path}?mode=ro", uri=True) as conn:
                return int(conn.execute("SELECT COUNT(*) FROM nodes").fetchone()[0])
        except sqlite3.Error:
            return 0
    payload = _read_json(kg_path, {})
    return len(payload.get("nodes", [])) if isinstance(payload, dict) else 0


def _budget_summary(config: dict[str, Any], runtime_path: Path) -> dict[str, Any]:
    policy = build_budget_policy(config)
    ledger_path = runtime_path / "cost_ledger.jsonl"
    if budget_tracking_enabled(config) and ledger_path.exists():
        return BudgetLedger(ledger_path, policy=policy, auto_load=True).summary()
    return {
        "spent_usd": 0.0,
        "remaining_usd": float(policy.max_compute_budget_per_session),
        "by_task_type": {},
        "stop_reasons": {},
    }


def _active_forecasts(runtime_path: Path) -> int:
    payload = _read_json(runtime_path / "forecasts.json", [])
    if not isinstance(payload, list):
        return 0
    return sum(1 for item in payload if isinstance(item, dict) and str(item.get("status", "pending")) == "pending")


def _label_value(value: Any) -> str:
    return str(value).replace("\\", "\\\\").replace("\n", "\\n").replace('"', '\\"')


def _format_sample(sample: MetricSample) -> str:
    labels = sample.labels or {}
    if labels:
        label_text = ",".join(f'{key}="{_label_value(value)}"' for key, value in sorted(labels.items()))
        return f"{sample.name}{{{label_text}}} {float(sample.value):.12g}"
    return f"{sample.name} {float(sample.value):.12g}"


def _metric_family(name: str) -> str:
    for suffix in ("_bucket", "_sum", "_count"):
        if name.endswith(suffix):
            base = name[: -len(suffix)]
            if _METRIC_HELP.get(base, ("", ""))[0] == "histogram":
                return base
    return name


def _duration_values(events: list[dict[str, Any]], field: str) -> list[float]:
    values: list[float] = []
    for event in events:
        diff = event.get("diff", {}) if isinstance(event.get("diff"), dict) else {}
        value = diff.get(field)
        if value is None:
            continue
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            continue
        if numeric >= 0.0:
            values.append(numeric)
    return values


def _histogram_samples(name: str, values: list[float], labels: dict[str, str] | None = None) -> list[MetricSample]:
    if not values:
        return []
    samples: list[MetricSample] = []
    for bucket in _HISTOGRAM_BUCKETS:
        bucket_labels = {**dict(labels or {}), "le": f"{bucket:g}"}
        samples.append(MetricSample(f"{name}_bucket", sum(1 for value in values if value <= bucket), bucket_labels))
    samples.append(MetricSample(f"{name}_bucket", len(values), {**dict(labels or {}), "le": "+Inf"}))
    samples.append(MetricSample(f"{name}_sum", sum(values), labels))
    samples.append(MetricSample(f"{name}_count", len(values), labels))
    return samples


def _count_llm_events(events: list[dict[str, Any]], provider: str) -> list[MetricSample]:
    calls: dict[tuple[str, str], int] = {}
    errors: dict[tuple[str, str], int] = {}
    provider = provider or "none"
    for event in events:
        diff = event.get("diff", {}) if isinstance(event.get("diff"), dict) else {}
        task_type = "signal_processing" if event.get("operator") == "runtime_signal_ingest" else str(event.get("operator", "runtime"))
        if bool(diff.get("llm_used")):
            calls[(provider, task_type)] = calls.get((provider, task_type), 0) + 1
        error_text = diff.get("update_error") or diff.get("llm_error")
        if error_text:
            error_type = str(error_text).split(":", 1)[0].strip() or "unknown"
            errors[(provider, error_type)] = errors.get((provider, error_type), 0) + 1
    samples = [
        MetricSample("llm_calls_total", count, {"provider": provider_name, "task_type": task_type})
        for (provider_name, task_type), count in sorted(calls.items())
    ]
    samples.extend(
        MetricSample("llm_errors_total", count, {"provider": provider_name, "error_type": error_type})
        for (provider_name, error_type), count in sorted(errors.items())
    )
    if not calls:
        samples.append(MetricSample("llm_calls_total", 0, {"provider": provider, "task_type": "signal_processing"}))
    if not errors:
        samples.append(MetricSample("llm_errors_total", 0, {"provider": provider, "error_type": "none"}))
    return samples


def collect_metrics(config_path: str | Path = "config.yaml") -> list[MetricSample]:
    """Collect read-only runtime metrics from persisted Freeman artifacts."""

    config, resolved = _load_config(config_path)
    runtime_cfg = dict(config.get("runtime", {}) or {})
    runtime_path = _resolve_path(resolved.parent, runtime_cfg.get("runtime_path"), "./data/runtime")
    event_log_path = _resolve_path(
        resolved.parent,
        runtime_cfg.get("event_log_path"),
        str(runtime_path / "event_log.jsonl"),
    )
    kg_path = _kg_path(config, config_path=resolved)
    events = _read_events(event_log_path)
    signal_events = [event for event in events if event.get("operator") == "runtime_signal_ingest"]
    filtered = [
        event
        for event in signal_events
        if isinstance(event.get("diff"), dict) and bool(event["diff"].get("filtered_out"))
    ]
    ontology_repairs = [event for event in events if event.get("operator") == "ontology_repair_request"]
    budget = _budget_summary(config, runtime_path)
    provider = str(dict(config.get("llm", {}) or {}).get("provider", "none") or "none")
    samples = [
        MetricSample("signals_ingested_total", len(signal_events)),
        MetricSample("signals_filtered_total", len(filtered)),
        MetricSample("ontology_repairs_total", len(ontology_repairs)),
        MetricSample("world_t", _world_t(runtime_path)),
        MetricSample("kg_node_count", _kg_node_count(config, kg_path)),
        MetricSample("budget_spent_usd", float(budget.get("spent_usd", 0.0))),
        MetricSample("budget_remaining_usd", float(budget.get("remaining_usd", 0.0))),
        MetricSample("active_forecasts", _active_forecasts(runtime_path)),
    ]
    samples.extend(_count_llm_events(events, provider))
    samples.extend(_histogram_samples("signal_processing_seconds", _duration_values(events, "signal_processing_seconds")))
    samples.extend(_histogram_samples("analysis_pipeline_seconds", _duration_values(events, "analysis_pipeline_seconds")))
    samples.extend(
        _histogram_samples(
            "llm_call_seconds",
            _duration_values(events, "llm_call_seconds"),
            {"provider": provider},
        )
    )
    return samples


def render_prometheus(samples: list[MetricSample]) -> str:
    """Render samples in Prometheus text exposition format."""

    seen: set[str] = set()
    lines: list[str] = []
    for sample in samples:
        family = _metric_family(sample.name)
        if family not in seen:
            metric_type, help_text = _METRIC_HELP.get(family, ("gauge", family))
            lines.append(f"# HELP {family} {help_text}")
            lines.append(f"# TYPE {family} {metric_type}")
            seen.add(family)
        lines.append(_format_sample(sample))
    return "\n".join(lines) + "\n"


def metrics_from_config(config_path: str | Path = "config.yaml") -> str:
    """Return Prometheus text metrics for a config path."""

    return render_prometheus(collect_metrics(config_path))


__all__ = ["MetricSample", "collect_metrics", "metrics_from_config", "render_prometheus"]
