"""Human override API for domain parameters and causal-edge signs."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List

from freeman_librarian.core.world import WorldState
from freeman_librarian.game.runner import GameRunner, SimConfig
from freeman_librarian.interface.simulationdiff import SimulationDiffReport, build_simulation_diff


def _now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _resolve_path(container: Dict[str, Any], path: str) -> tuple[Any, str]:
    parts = path.split(".")
    cursor: Any = container
    for part in parts[:-1]:
        cursor = cursor[int(part)] if part.isdigit() else cursor[part]
    return cursor, parts[-1]


def _get_path_value(container: Dict[str, Any], path: str) -> Any:
    cursor: Any = container
    for part in path.split("."):
        cursor = cursor[int(part)] if part.isdigit() else cursor[part]
    return cursor


def _set_path_value(container: Dict[str, Any], path: str, value: Any) -> None:
    cursor, leaf = _resolve_path(container, path)
    if leaf.isdigit():
        cursor[int(leaf)] = value
    else:
        cursor[leaf] = value


@dataclass
class OverrideAuditEntry:
    """Audit-log entry for a human override."""

    domain_id: str
    version: int
    action: str
    path: str
    before: Any
    after: Any
    actor: str = "human"
    timestamp: str = field(default_factory=_now_iso)

    def snapshot(self) -> Dict[str, Any]:
        return {
            "domain_id": self.domain_id,
            "version": self.version,
            "action": self.action,
            "path": self.path,
            "before": self.before,
            "after": self.after,
            "actor": self.actor,
            "timestamp": self.timestamp,
        }


@dataclass
class DomainOverrideRecord:
    """Base machine hypothesis plus editable human-adjusted versions."""

    domain_id: str
    machine_world: WorldState
    current_world: WorldState
    machine_simulation: Dict[str, Any] | None = None
    current_simulation: Dict[str, Any] | None = None
    audit_log: List[OverrideAuditEntry] = field(default_factory=list)
    version: int = 0


class ModelOverrideAPI:
    """Editable domain API matching the v0.2 override requirements."""

    def __init__(self, sim_config: SimConfig | None = None) -> None:
        self.sim_config = sim_config or SimConfig(max_steps=5)
        self.runner = GameRunner(self.sim_config)
        self.records: Dict[str, DomainOverrideRecord] = {}

    def register_domain(
        self,
        domain_id: str,
        world: WorldState,
        *,
        machine_simulation: Dict[str, Any] | None = None,
    ) -> None:
        self.records[domain_id] = DomainOverrideRecord(
            domain_id=domain_id,
            machine_world=world.clone(),
            current_world=world.clone(),
            machine_simulation=machine_simulation,
            current_simulation=machine_simulation,
        )

    def patch_params(
        self,
        domain_id: str,
        overrides: Dict[str, Any],
        *,
        actor: str = "human",
    ) -> Dict[str, Any]:
        record = self.records[domain_id]
        snapshot = record.current_world.snapshot()
        audit_entries: List[OverrideAuditEntry] = []
        for path, value in overrides.items():
            before = _get_path_value(snapshot, path)
            _set_path_value(snapshot, path, value)
            audit_entries.append(
                OverrideAuditEntry(
                    domain_id=domain_id,
                    version=record.version + 1,
                    action="PATCH_PARAM",
                    path=path,
                    before=before,
                    after=value,
                    actor=actor,
                )
            )
        record.current_world = WorldState.from_snapshot(snapshot)
        record.version += 1
        record.audit_log.extend(audit_entries)
        return {
            "domain_id": domain_id,
            "version": record.version,
            "audit_log": [entry.snapshot() for entry in audit_entries],
        }

    def patch_edge(
        self,
        domain_id: str,
        edge_id: int | str,
        expected_sign: str,
        *,
        actor: str = "human",
    ) -> Dict[str, Any]:
        record = self.records[domain_id]
        snapshot = record.current_world.snapshot()

        if isinstance(edge_id, int):
            index = edge_id
        else:
            source, target = edge_id.split("->", 1)
            index = next(
                idx
                for idx, edge in enumerate(snapshot["causal_dag"])
                if edge["source"] == source and edge["target"] == target
            )

        path = f"causal_dag.{index}.expected_sign"
        before = _get_path_value(snapshot, path)
        _set_path_value(snapshot, path, expected_sign)
        entry = OverrideAuditEntry(
            domain_id=domain_id,
            version=record.version + 1,
            action="PATCH_EDGE",
            path=path,
            before=before,
            after=expected_sign,
            actor=actor,
        )
        record.current_world = WorldState.from_snapshot(snapshot)
        record.version += 1
        record.audit_log.append(entry)
        return {"domain_id": domain_id, "version": record.version, "audit_log": [entry.snapshot()]}

    def rerun_domain(
        self,
        domain_id: str,
        *,
        policies: Iterable[Any] = (),
    ) -> Dict[str, Any]:
        record = self.records[domain_id]
        simulation = self.runner.run(record.current_world.clone(), list(policies))
        record.current_simulation = simulation.snapshot()
        return record.current_simulation

    def get_diff(self, domain_id: str) -> Dict[str, Any]:
        record = self.records[domain_id]
        before = record.machine_simulation or record.machine_world.snapshot()
        after = record.current_simulation or record.current_world.snapshot()
        diff = build_simulation_diff(
            domain_id=domain_id,
            before=before,
            after=after,
            override_history=[entry.snapshot() for entry in record.audit_log],
        )
        return diff.snapshot()

    def get_diff_report(self, domain_id: str) -> SimulationDiffReport:
        record = self.records[domain_id]
        before = record.machine_simulation or record.machine_world.snapshot()
        after = record.current_simulation or record.current_world.snapshot()
        return build_simulation_diff(
            domain_id=domain_id,
            before=before,
            after=after,
            override_history=[entry.snapshot() for entry in record.audit_log],
        )


__all__ = ["DomainOverrideRecord", "ModelOverrideAPI", "OverrideAuditEntry"]
