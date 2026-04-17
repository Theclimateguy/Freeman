"""Persistent knowledge-graph snapshots for evolution playback."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any

from freeman_librarian.memory.knowledgegraph import KnowledgeGraph


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _stable_suffix(value: str, *, length: int = 10) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:length]


def _compact_token(value: str | None, *, fallback: str = "snapshot", max_len: int = 48) -> str:
    if not value:
        return fallback
    cleaned = "".join(char.lower() if char.isalnum() else "_" for char in str(value))
    while "__" in cleaned:
        cleaned = cleaned.replace("__", "_")
    cleaned = cleaned.strip("_")
    if not cleaned:
        return fallback
    return cleaned[:max_len]


@dataclass
class KGSnapshotManager:
    """Write ordered KG snapshots plus a compact manifest for timeline viewers."""

    snapshot_dir: Path
    enabled: bool = False
    max_snapshots: int = 0

    def write_snapshot(
        self,
        knowledge_graph: KnowledgeGraph,
        *,
        runtime_step: int,
        reason: str,
        domain_id: str,
        signal_id: str | None = None,
        trigger_mode: str | None = None,
        world_t: int | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> Path | None:
        if not self.enabled:
            return None

        payload = knowledge_graph.to_payload()
        timestamp = _utc_now_iso()
        signal_token = _compact_token(signal_id, fallback=reason)
        unique_suffix = _stable_suffix(f"{runtime_step}|{reason}|{signal_id or ''}|{timestamp}")
        filename = f"{int(runtime_step):06d}__{_compact_token(reason)}__{signal_token}__{unique_suffix}.json"
        target = self.snapshot_dir / filename
        self.snapshot_dir.mkdir(parents=True, exist_ok=True)

        snapshot_meta = {
            "snapshot_id": target.stem,
            "timestamp": timestamp,
            "runtime_step": int(runtime_step),
            "world_t": int(world_t) if world_t is not None else None,
            "reason": str(reason),
            "domain_id": str(domain_id),
            "signal_id": str(signal_id) if signal_id else None,
            "trigger_mode": str(trigger_mode) if trigger_mode else None,
            "node_count": len(payload.get("nodes", [])),
            "edge_count": len(payload.get("edges", [])),
            "knowledge_graph_path": str(knowledge_graph.json_path),
            "extra": json.loads(json.dumps(metadata or {}, ensure_ascii=False)),
        }
        payload["snapshot_meta"] = snapshot_meta
        target.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")

        manifest_path = self.snapshot_dir / "manifest.jsonl"
        with manifest_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps({"path": str(target), **snapshot_meta}, ensure_ascii=False, sort_keys=True))
            handle.write("\n")

        self._prune_if_needed(manifest_path)
        return target

    def _prune_if_needed(self, manifest_path: Path) -> None:
        if self.max_snapshots <= 0:
            return
        snapshots = sorted(path for path in self.snapshot_dir.glob("*.json") if path.name != "manifest.json")
        if len(snapshots) <= self.max_snapshots:
            return
        remove_count = len(snapshots) - self.max_snapshots
        to_remove = snapshots[:remove_count]
        remove_set = {str(path.resolve()) for path in to_remove}
        for path in to_remove:
            path.unlink(missing_ok=True)
        if not manifest_path.exists():
            return
        kept_lines = []
        for line in manifest_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            payload = json.loads(line)
            if str(Path(payload["path"]).resolve()) in remove_set:
                continue
            kept_lines.append(json.dumps(payload, ensure_ascii=False, sort_keys=True))
        manifest_path.write_text("\n".join(kept_lines) + ("\n" if kept_lines else ""), encoding="utf-8")


def snapshot_manager_from_config(config: dict[str, Any], *, runtime_path: Path, config_base: Path) -> KGSnapshotManager:
    runtime_cfg = dict(config.get("runtime", {}))
    snapshot_cfg = dict(runtime_cfg.get("kg_snapshots", {}))
    enabled = bool(snapshot_cfg.get("enabled", False))
    snapshot_path_value = snapshot_cfg.get("path")
    snapshot_dir = (
        Path(snapshot_path_value).expanduser()
        if snapshot_path_value
        else runtime_path / "kg_snapshots"
    )
    if not snapshot_dir.is_absolute():
        snapshot_dir = (config_base / snapshot_dir).resolve()
    return KGSnapshotManager(
        snapshot_dir=snapshot_dir,
        enabled=enabled,
        max_snapshots=max(int(snapshot_cfg.get("max_snapshots", 0)), 0),
    )


__all__ = ["KGSnapshotManager", "snapshot_manager_from_config"]
