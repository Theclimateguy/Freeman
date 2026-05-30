"""Pluggable node-lock backends for Freeman runtimes."""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
import os
from pathlib import Path
import time
from typing import Any, Callable, Protocol, runtime_checkable


SaveCallback = Callable[[], None]
REDIS_UNLOCK_SCRIPT = (
    "if redis.call('get', KEYS[1]) == ARGV[1] then "
    "return redis.call('del', KEYS[1]) else return 0 end"
)


@runtime_checkable
class LockBackend(Protocol):
    """Minimal distributed/cooperative node-lock interface."""

    backend_name: str
    stores_graph_lock_state: bool

    def try_lock(
        self,
        node_id: str,
        agent_id: str,
        *,
        graph: Any,
        namespace: str,
        lock_ttl_seconds: float | None = None,
        save_callback: SaveCallback | None = None,
    ) -> bool:
        """Attempt to acquire a lock for ``node_id``."""

    def unlock(
        self,
        node_id: str,
        agent_id: str | None = None,
        *,
        graph: Any,
        namespace: str,
        force: bool = False,
        save_callback: SaveCallback | None = None,
    ) -> bool:
        """Release a lock for ``node_id``."""


def _now() -> float:
    return time.time()


def _validate_agent_id(agent_id: str) -> str:
    normalized = str(agent_id).strip()
    if not normalized:
        raise ValueError("agent_id must be non-empty")
    return normalized


def _safe_key(value: str) -> str:
    digest = hashlib.sha256(str(value).encode("utf-8")).hexdigest()
    return digest[:32]


@dataclass
class InMemoryLockBackend:
    """Graph-backed cooperative lock with legacy Freeman semantics."""

    backend_name: str = "memory"
    stores_graph_lock_state: bool = True

    def try_lock(
        self,
        node_id: str,
        agent_id: str,
        *,
        graph: Any,
        namespace: str,
        lock_ttl_seconds: float | None = None,
        save_callback: SaveCallback | None = None,
    ) -> bool:
        del namespace
        if node_id not in graph:
            return False
        owner = _validate_agent_id(agent_id)
        node_payload = graph.nodes[node_id]
        now = _now()
        lock_owner = node_payload.get("locked_by")
        lock_timestamp = node_payload.get("locked_at")
        if lock_owner is None:
            node_payload["locked_by"] = owner
            node_payload["locked_at"] = now
            if save_callback is not None:
                save_callback()
            return True
        if lock_owner == owner:
            node_payload["locked_at"] = now
            if save_callback is not None:
                save_callback()
            return True
        if lock_ttl_seconds is not None and lock_timestamp is not None:
            if (now - float(lock_timestamp)) >= max(float(lock_ttl_seconds), 0.0):
                node_payload["locked_by"] = owner
                node_payload["locked_at"] = now
                if save_callback is not None:
                    save_callback()
                return True
        return False

    def unlock(
        self,
        node_id: str,
        agent_id: str | None = None,
        *,
        graph: Any,
        namespace: str,
        force: bool = False,
        save_callback: SaveCallback | None = None,
    ) -> bool:
        del namespace
        if node_id not in graph:
            return False
        node_payload = graph.nodes[node_id]
        lock_owner = node_payload.get("locked_by")
        if lock_owner is None:
            return False
        if not force and agent_id is not None and str(lock_owner) != str(agent_id):
            return False
        node_payload["locked_by"] = None
        node_payload["locked_at"] = None
        if save_callback is not None:
            save_callback()
        return True


@dataclass
class FileSystemLockBackend:
    """Single-host multi-process lock backend using atomic lock files."""

    lock_dir: Path
    backend_name: str = "filesystem"
    stores_graph_lock_state: bool = False

    def __post_init__(self) -> None:
        self.lock_dir = Path(self.lock_dir).expanduser().resolve()

    def try_lock(
        self,
        node_id: str,
        agent_id: str,
        *,
        graph: Any,
        namespace: str,
        lock_ttl_seconds: float | None = None,
        save_callback: SaveCallback | None = None,
    ) -> bool:
        del save_callback
        if node_id not in graph:
            return False
        owner = _validate_agent_id(agent_id)
        self.lock_dir.mkdir(parents=True, exist_ok=True)
        path = self._lock_path(namespace, node_id)
        payload = self._payload(node_id=node_id, agent_id=owner, namespace=namespace)
        try:
            fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            if not self._is_stale(path, lock_ttl_seconds=lock_ttl_seconds):
                return self._refresh_if_owner(path, owner)
            self._remove_stale(path)
            try:
                fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            except FileExistsError:
                return False
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, sort_keys=True)
        return True

    def unlock(
        self,
        node_id: str,
        agent_id: str | None = None,
        *,
        graph: Any,
        namespace: str,
        force: bool = False,
        save_callback: SaveCallback | None = None,
    ) -> bool:
        del graph, save_callback
        path = self._lock_path(namespace, node_id)
        if not path.exists():
            return False
        payload = self._read_payload(path)
        lock_owner = str(payload.get("agent_id", ""))
        if not force and agent_id is not None and lock_owner != str(agent_id):
            return False
        try:
            path.unlink()
        except FileNotFoundError:
            return False
        return True

    def _lock_path(self, namespace: str, node_id: str) -> Path:
        return self.lock_dir / f"{_safe_key(namespace)}__{_safe_key(node_id)}.lock"

    def _payload(self, *, node_id: str, agent_id: str, namespace: str) -> dict[str, Any]:
        return {
            "node_id": str(node_id),
            "agent_id": str(agent_id),
            "namespace": str(namespace),
            "locked_at": _now(),
        }

    def _read_payload(self, path: Path) -> dict[str, Any]:
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return {}

    def _is_stale(self, path: Path, *, lock_ttl_seconds: float | None) -> bool:
        if lock_ttl_seconds is None:
            return False
        payload = self._read_payload(path)
        locked_at = float(payload.get("locked_at", path.stat().st_mtime))
        return (_now() - locked_at) >= max(float(lock_ttl_seconds), 0.0)

    def _remove_stale(self, path: Path) -> None:
        try:
            path.unlink()
        except FileNotFoundError:
            return

    def _refresh_if_owner(self, path: Path, agent_id: str) -> bool:
        payload = self._read_payload(path)
        if str(payload.get("agent_id", "")) != str(agent_id):
            return False
        payload["locked_at"] = _now()
        path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
        return True


@dataclass
class RedisLockBackend:
    """Distributed Redis lock backend using ``SET NX PX`` leases."""

    redis_url: str | None = None
    client: Any | None = None
    key_prefix: str = "freeman:node-lock"
    backend_name: str = "redis"
    stores_graph_lock_state: bool = False
    _unlock_script: Any | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        if self.client is not None:
            return
        try:
            import redis  # type: ignore[import-not-found]
        except ImportError as exc:  # pragma: no cover - optional dependency boundary
            raise RuntimeError("redis package is required for RedisLockBackend") from exc
        if not self.redis_url:
            raise RuntimeError("redis_url is required for RedisLockBackend")
        self.client = redis.Redis.from_url(self.redis_url)

    def try_lock(
        self,
        node_id: str,
        agent_id: str,
        *,
        graph: Any,
        namespace: str,
        lock_ttl_seconds: float | None = None,
        save_callback: SaveCallback | None = None,
    ) -> bool:
        del save_callback
        if node_id not in graph:
            return False
        owner = _validate_agent_id(agent_id)
        key = self._key(namespace, node_id)
        ttl_ms = None if lock_ttl_seconds is None else max(int(float(lock_ttl_seconds) * 1000), 1)
        acquired = bool(self.client.set(key, owner, nx=True, px=ttl_ms))
        if acquired:
            return True
        current_owner = self._decode(self.client.get(key))
        if current_owner != owner:
            return False
        if ttl_ms is not None:
            self.client.pexpire(key, ttl_ms)
        return True

    def unlock(
        self,
        node_id: str,
        agent_id: str | None = None,
        *,
        graph: Any,
        namespace: str,
        force: bool = False,
        save_callback: SaveCallback | None = None,
    ) -> bool:
        del graph, save_callback
        key = self._key(namespace, node_id)
        if force or agent_id is None:
            return bool(self.client.delete(key))
        return bool(self._registered_unlock_script()(keys=[key], args=[str(agent_id)]))

    def _key(self, namespace: str, node_id: str) -> str:
        return f"{self.key_prefix}:{_safe_key(namespace)}:{_safe_key(node_id)}"

    def _decode(self, value: Any) -> str | None:
        if value is None:
            return None
        if isinstance(value, bytes):
            return value.decode("utf-8")
        return str(value)

    def _registered_unlock_script(self) -> Any:
        if self._unlock_script is None:
            self._unlock_script = self.client.register_script(REDIS_UNLOCK_SCRIPT)
        return self._unlock_script


__all__ = [
    "FileSystemLockBackend",
    "InMemoryLockBackend",
    "LockBackend",
    "REDIS_UNLOCK_SCRIPT",
    "RedisLockBackend",
]
