"""Session log primitives for attention and KG updates."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from freeman_librarian.utils import deep_copy_jsonable, json_ready


def _now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


@dataclass
class KGDelta:
    """A knowledge-graph mutation emitted during a session."""

    operation: str
    target_id: str | None = None
    payload: Dict[str, Any] = field(default_factory=dict)
    support: int = 1
    contradiction: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(default_factory=_now_iso)

    def snapshot(self) -> Dict[str, Any]:
        return json_ready(
            {
                "operation": self.operation,
                "target_id": self.target_id,
                "payload": self.payload,
                "support": self.support,
                "contradiction": self.contradiction,
                "metadata": self.metadata,
                "timestamp": self.timestamp,
            }
        )

    @classmethod
    def from_snapshot(cls, data: Dict[str, Any]) -> "KGDelta":
        return cls(
            operation=data["operation"],
            target_id=data.get("target_id"),
            payload=deep_copy_jsonable(data.get("payload", {})),
            support=int(data.get("support", 1)),
            contradiction=int(data.get("contradiction", 0)),
            metadata=deep_copy_jsonable(data.get("metadata", {})),
            timestamp=data.get("timestamp", _now_iso()),
        )


@dataclass
class AttentionStep:
    """One attention-allocation step."""

    step_index: int
    task_id: str
    status: str
    interest_score: float
    exploration_bonus: float
    utility_score: float
    note: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(default_factory=_now_iso)

    def snapshot(self) -> Dict[str, Any]:
        return json_ready(
            {
                "step_index": self.step_index,
                "task_id": self.task_id,
                "status": self.status,
                "interest_score": self.interest_score,
                "exploration_bonus": self.exploration_bonus,
                "utility_score": self.utility_score,
                "note": self.note,
                "metadata": self.metadata,
                "timestamp": self.timestamp,
            }
        )

    @classmethod
    def from_snapshot(cls, data: Dict[str, Any]) -> "AttentionStep":
        return cls(
            step_index=int(data["step_index"]),
            task_id=data["task_id"],
            status=data["status"],
            interest_score=float(data.get("interest_score", 0.0)),
            exploration_bonus=float(data.get("exploration_bonus", 0.0)),
            utility_score=float(data.get("utility_score", 0.0)),
            note=data.get("note", ""),
            metadata=deep_copy_jsonable(data.get("metadata", {})),
            timestamp=data.get("timestamp", _now_iso()),
        )


@dataclass
class TaskRecord:
    """Tracked task inside a session."""

    task_id: str
    domain_id: str
    query: str
    status: str = "PENDING"
    task_type: str = "analysis"
    outcome: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    attention_steps: List[AttentionStep] = field(default_factory=list)
    created_at: str = field(default_factory=_now_iso)
    updated_at: str = field(default_factory=_now_iso)

    def add_attention_step(self, step: AttentionStep) -> None:
        self.attention_steps.append(step)
        self.updated_at = step.timestamp

    def snapshot(self) -> Dict[str, Any]:
        return json_ready(
            {
                "task_id": self.task_id,
                "domain_id": self.domain_id,
                "query": self.query,
                "status": self.status,
                "task_type": self.task_type,
                "outcome": self.outcome,
                "metadata": self.metadata,
                "attention_steps": [step.snapshot() for step in self.attention_steps],
                "created_at": self.created_at,
                "updated_at": self.updated_at,
            }
        )

    @classmethod
    def from_snapshot(cls, data: Dict[str, Any]) -> "TaskRecord":
        return cls(
            task_id=data["task_id"],
            domain_id=data["domain_id"],
            query=data["query"],
            status=data.get("status", "PENDING"),
            task_type=data.get("task_type", "analysis"),
            outcome=data.get("outcome", ""),
            metadata=deep_copy_jsonable(data.get("metadata", {})),
            attention_steps=[AttentionStep.from_snapshot(step) for step in data.get("attention_steps", [])],
            created_at=data.get("created_at", _now_iso()),
            updated_at=data.get("updated_at", _now_iso()),
        )


@dataclass
class SessionLog:
    """Serializable record of tasks, attention, and KG deltas."""

    session_id: str
    task_records: List[TaskRecord] = field(default_factory=list)
    kg_deltas: List[KGDelta] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    started_at: str = field(default_factory=_now_iso)
    ended_at: Optional[str] = None

    def get_task(self, task_id: str) -> TaskRecord | None:
        for task in self.task_records:
            if task.task_id == task_id:
                return task
        return None

    def add_task(self, task: TaskRecord) -> None:
        self.task_records.append(task)

    def add_attention_step(self, task_id: str, step: AttentionStep) -> None:
        task = self.get_task(task_id)
        if task is None:
            raise KeyError(task_id)
        task.add_attention_step(step)

    def add_kg_delta(self, delta: KGDelta) -> None:
        self.kg_deltas.append(delta)

    def snapshot(self) -> Dict[str, Any]:
        return json_ready(
            {
                "session_id": self.session_id,
                "task_records": [task.snapshot() for task in self.task_records],
                "kg_deltas": [delta.snapshot() for delta in self.kg_deltas],
                "metadata": self.metadata,
                "started_at": self.started_at,
                "ended_at": self.ended_at,
            }
        )

    @classmethod
    def from_snapshot(cls, data: Dict[str, Any]) -> "SessionLog":
        return cls(
            session_id=data["session_id"],
            task_records=[TaskRecord.from_snapshot(task) for task in data.get("task_records", [])],
            kg_deltas=[KGDelta.from_snapshot(delta) for delta in data.get("kg_deltas", [])],
            metadata=deep_copy_jsonable(data.get("metadata", {})),
            started_at=data.get("started_at", _now_iso()),
            ended_at=data.get("ended_at"),
        )

    def save(self, path: str | Path) -> Path:
        target = Path(path).resolve()
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(self.snapshot(), indent=2, sort_keys=True), encoding="utf-8")
        return target

    @classmethod
    def load(cls, path: str | Path) -> "SessionLog":
        payload = json.loads(Path(path).resolve().read_text(encoding="utf-8"))
        return cls.from_snapshot(payload)


__all__ = ["AttentionStep", "KGDelta", "SessionLog", "TaskRecord"]
