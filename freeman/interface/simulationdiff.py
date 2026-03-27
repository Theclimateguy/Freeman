"""Simulation diff and override-history export."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any, Dict, List


@dataclass
class DiffEntry:
    """One changed field between two payloads."""

    path: str
    before: Any
    after: Any


@dataclass
class SimulationDiffReport:
    """Structured diff for world/simulation comparisons."""

    domain_id: str
    changes: List[DiffEntry]
    override_history: List[Dict[str, Any]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def snapshot(self) -> Dict[str, Any]:
        return {
            "domain_id": self.domain_id,
            "changes": [entry.__dict__ for entry in self.changes],
            "override_history": self.override_history,
            "metadata": self.metadata,
        }


def _flatten(value: Any, prefix: str = "") -> Dict[str, Any]:
    flat: Dict[str, Any] = {}
    if isinstance(value, dict):
        for key, nested in value.items():
            next_prefix = f"{prefix}.{key}" if prefix else str(key)
            flat.update(_flatten(nested, next_prefix))
        return flat
    if isinstance(value, list):
        for index, nested in enumerate(value):
            next_prefix = f"{prefix}.{index}" if prefix else str(index)
            flat.update(_flatten(nested, next_prefix))
        return flat
    flat[prefix] = value
    return flat


def build_simulation_diff(
    *,
    domain_id: str,
    before: Dict[str, Any],
    after: Dict[str, Any],
    override_history: List[Dict[str, Any]] | None = None,
) -> SimulationDiffReport:
    before_flat = _flatten(before)
    after_flat = _flatten(after)
    changes = [
        DiffEntry(path=path, before=before_flat.get(path), after=after_flat.get(path))
        for path in sorted(set(before_flat) | set(after_flat))
        if before_flat.get(path) != after_flat.get(path)
    ]
    return SimulationDiffReport(
        domain_id=domain_id,
        changes=changes,
        override_history=override_history or [],
        metadata={"change_count": len(changes)},
    )


def export_simulation_diff(report: SimulationDiffReport, path: str | Path) -> Path:
    target = Path(path).resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report.snapshot(), indent=2, sort_keys=True), encoding="utf-8")
    return target


__all__ = ["DiffEntry", "SimulationDiffReport", "build_simulation_diff", "export_simulation_diff"]
