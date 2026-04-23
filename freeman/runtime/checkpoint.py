"""Minimal world-state checkpoint persistence."""

from __future__ import annotations

from pathlib import Path

from freeman.core.world import WorldState
from freeman.lite_state import load_world_state, save_world_state


class CheckpointManager:
    """Save and load a single world-state checkpoint."""

    def save(self, state: WorldState, path: Path) -> None:
        save_world_state(state, path)

    def load(self, path: Path) -> WorldState | None:
        return load_world_state(path)

    def list_checkpoints(self, runtime_path: Path) -> list[Path]:
        base = Path(runtime_path).resolve()
        if not base.exists():
            return []
        return sorted(path for path in base.glob("*.json") if path.is_file())

    def validate(self, path: Path) -> bool:
        return load_world_state(path) is not None


__all__ = ["CheckpointManager"]
