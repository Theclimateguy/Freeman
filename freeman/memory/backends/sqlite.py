"""SQLite KnowledgeGraph backend."""

from __future__ import annotations

from contextlib import contextmanager
import json
from pathlib import Path
import sqlite3
from typing import Any, Iterator


class SqliteKGBackend:
    """Persist KnowledgeGraph nodes and edges in SQLite tables."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path).expanduser().resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._ensure_schema()

    def load(self) -> dict[str, Any]:
        self._ensure_schema()
        with self._connect() as conn:
            nodes = [
                json.loads(row["data"])
                for row in conn.execute("SELECT data FROM nodes ORDER BY id").fetchall()
            ]
            edges = [
                json.loads(row["data"])
                for row in conn.execute("SELECT data FROM edges ORDER BY rowid").fetchall()
            ]
        return {
            "backend": "sqlite",
            "sqlite_path": str(self.path),
            "nodes": nodes,
            "edges": edges,
        }

    def save(self, data: dict[str, Any]) -> None:
        self._ensure_schema()
        with self._connect() as conn:
            conn.execute("BEGIN")
            conn.execute("DELETE FROM edges")
            conn.execute("DELETE FROM nodes")
            for node in data.get("nodes", []):
                node_payload = dict(node)
                conn.execute(
                    "INSERT OR REPLACE INTO nodes(id, type, status, data) VALUES (?, ?, ?, ?)",
                    (
                        str(node_payload["id"]),
                        str(node_payload.get("node_type", "")),
                        str(node_payload.get("status", "")),
                        json.dumps(node_payload, ensure_ascii=False, sort_keys=True),
                    ),
                )
            for index, edge in enumerate(data.get("edges", [])):
                edge_payload = dict(edge)
                edge_id = str(
                    edge_payload.get("id")
                    or f"{edge_payload.get('source')}:{edge_payload.get('relation_type')}:{edge_payload.get('target')}:{index}"
                )
                edge_payload["id"] = edge_id
                conn.execute(
                    "INSERT OR REPLACE INTO edges(id, source, target, relation, data) VALUES (?, ?, ?, ?, ?)",
                    (
                        edge_id,
                        str(edge_payload.get("source", "")),
                        str(edge_payload.get("target", "")),
                        str(edge_payload.get("relation_type", "")),
                        json.dumps(edge_payload, ensure_ascii=False, sort_keys=True),
                    ),
                )
            conn.commit()

    @contextmanager
    def transaction(self) -> Iterator[None]:
        self._ensure_schema()
        with self._connect() as conn:
            conn.execute("SAVEPOINT kg_backend")
            try:
                yield
            except Exception:
                conn.execute("ROLLBACK TO kg_backend")
                conn.execute("RELEASE kg_backend")
                raise
            else:
                conn.execute("RELEASE kg_backend")

    def node_count(self) -> int:
        self._ensure_schema()
        with self._connect() as conn:
            return int(conn.execute("SELECT COUNT(*) FROM nodes").fetchone()[0])

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path)
        conn.row_factory = sqlite3.Row
        return conn

    def _ensure_schema(self) -> None:
        with sqlite3.connect(self.path) as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS nodes (
                    id TEXT PRIMARY KEY,
                    type TEXT NOT NULL DEFAULT '',
                    status TEXT NOT NULL DEFAULT '',
                    data TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS edges (
                    id TEXT PRIMARY KEY,
                    source TEXT NOT NULL DEFAULT '',
                    target TEXT NOT NULL DEFAULT '',
                    relation TEXT NOT NULL DEFAULT '',
                    data TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_nodes_type ON nodes(type);
                CREATE INDEX IF NOT EXISTS idx_nodes_status ON nodes(status);
                CREATE INDEX IF NOT EXISTS idx_edges_source ON edges(source);
                CREATE INDEX IF NOT EXISTS idx_edges_target ON edges(target);
                """
            )


__all__ = ["SqliteKGBackend"]
