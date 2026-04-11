"""Checkpoint persistence for deterministic Freeman consciousness state."""

from __future__ import annotations

import json
from pathlib import Path

from freeman.agent.consciousness import ConsciousState
from freeman.memory.knowledgegraph import KnowledgeGraph


class CheckpointManager:
    """Save and load atomic consciousness checkpoints."""

    def save(self, state: ConsciousState, path: Path) -> None:
        target = Path(path).resolve()
        target.parent.mkdir(parents=True, exist_ok=True)
        temp_path = target.with_suffix(f"{target.suffix}.tmp")
        payload = {
            "schema_version": int(state.runtime_metadata.get("schema_version", 1)),
            "state": state.to_dict(),
        }
        temp_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        temp_path.replace(target)

    def load(self, path: Path) -> ConsciousState:
        source = Path(path).resolve()
        payload = json.loads(source.read_text(encoding="utf-8"))
        kg = KnowledgeGraph(auto_load=False, auto_save=False)
        return ConsciousState.from_dict(payload["state"], kg)

    def list_checkpoints(self, runtime_path: Path) -> list[Path]:
        base = Path(runtime_path).resolve()
        if not base.exists():
            return []
        return sorted(path for path in base.glob("*.json") if path.is_file())

    def validate(self, path: Path) -> bool:
        source = Path(path).resolve()
        if not source.exists():
            return False
        payload = json.loads(source.read_text(encoding="utf-8"))
        return int(payload.get("schema_version", 0)) >= 1 and isinstance(payload.get("state"), dict)


__all__ = ["CheckpointManager"]
