"""Forecast registry with verification horizons and optional persistence."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Dict, List

from freeman_librarian.agent.attentionscheduler import ForecastDebt, ObligationQueue


def _now_utc() -> datetime:
    return datetime.now(timezone.utc).replace(microsecond=0)


def _serialize_dt(value: datetime | None) -> str | None:
    if value is None:
        return None
    return value.astimezone(timezone.utc).replace(microsecond=0).isoformat()


def _parse_dt(value: str | None) -> datetime | None:
    if value is None:
        return None
    return datetime.fromisoformat(value)


@dataclass
class Forecast:
    """One probabilistic forecast awaiting later verification."""

    forecast_id: str
    domain_id: str
    outcome_id: str
    predicted_prob: float
    session_id: str
    horizon_steps: int
    created_at: datetime
    created_step: int = 0
    created_runtime_step: int | None = None
    verified_at: datetime | None = None
    actual_prob: float | None = None
    error: float | None = None
    status: str = "pending"
    causal_path: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.predicted_prob = float(self.predicted_prob)
        self.horizon_steps = int(self.horizon_steps)
        self.created_step = int(self.created_step)
        if self.created_runtime_step is not None:
            self.created_runtime_step = int(self.created_runtime_step)
        if self.actual_prob is not None:
            self.actual_prob = float(self.actual_prob)
        if self.error is not None:
            self.error = float(self.error)
        if self.status not in {"pending", "verified", "expired"}:
            raise ValueError(f"Invalid forecast status: {self.status}")

    @property
    def deadline_step(self) -> int:
        base_step = self.created_runtime_step if self.created_runtime_step is not None else self.created_step
        return int(base_step) + self.horizon_steps

    def snapshot(self) -> dict:
        return {
            "forecast_id": self.forecast_id,
            "domain_id": self.domain_id,
            "outcome_id": self.outcome_id,
            "predicted_prob": self.predicted_prob,
            "session_id": self.session_id,
            "horizon_steps": self.horizon_steps,
            "created_at": _serialize_dt(self.created_at),
            "created_step": self.created_step,
            "created_runtime_step": self.created_runtime_step,
            "verified_at": _serialize_dt(self.verified_at),
            "actual_prob": self.actual_prob,
            "error": self.error,
            "status": self.status,
            "causal_path": list(self.causal_path),
            "metadata": self.metadata,
        }

    @classmethod
    def from_snapshot(cls, data: dict) -> "Forecast":
        created_at = _parse_dt(data.get("created_at"))
        if created_at is None:
            created_at = _now_utc()
        return cls(
            forecast_id=data["forecast_id"],
            domain_id=data["domain_id"],
            outcome_id=data["outcome_id"],
            predicted_prob=float(data["predicted_prob"]),
            session_id=data["session_id"],
            horizon_steps=int(data["horizon_steps"]),
            created_at=created_at,
            created_step=int(data.get("created_step", 0)),
            created_runtime_step=(
                int(data["created_runtime_step"])
                if data.get("created_runtime_step") is not None
                else None
            ),
            verified_at=_parse_dt(data.get("verified_at")),
            actual_prob=data.get("actual_prob"),
            error=data.get("error"),
            status=data.get("status", "pending"),
            causal_path=[str(item) for item in data.get("causal_path", [])],
            metadata=dict(data.get("metadata", {})),
        )


class ForecastRegistry:
    """In-memory forecast registry with optional JSON persistence."""

    def __init__(
        self,
        *,
        json_path: str | Path | None = None,
        auto_load: bool = True,
        auto_save: bool = True,
        obligation_queue: ObligationQueue | None = None,
    ) -> None:
        self.json_path = Path(json_path).resolve() if json_path is not None else None
        self.auto_save = auto_save
        self.obligation_queue = obligation_queue
        self._forecasts: Dict[str, Forecast] = {}
        if auto_load and self.json_path is not None and self.json_path.exists():
            self.load()

    def record(self, forecast: Forecast) -> None:
        is_new = forecast.forecast_id not in self._forecasts
        self._forecasts[forecast.forecast_id] = Forecast.from_snapshot(forecast.snapshot())
        if is_new and self.obligation_queue is not None and forecast.status == "pending":
            self.obligation_queue.add_forecast_debt(
                ForecastDebt(
                    task_id=forecast.forecast_id,
                    domain_id=forecast.domain_id,
                    horizon_remaining=forecast.horizon_steps,
                )
            )
        self._maybe_save()

    def pending(self) -> List[Forecast]:
        return [forecast for forecast in self._forecasts.values() if forecast.status == "pending"]

    def all(self) -> List[Forecast]:
        return [
            Forecast.from_snapshot(forecast.snapshot())
            for forecast in sorted(
                self._forecasts.values(),
                key=lambda item: (
                    int(item.created_runtime_step if item.created_runtime_step is not None else item.created_step),
                    str(item.forecast_id),
                ),
            )
        ]

    def get(self, forecast_id: str) -> Forecast | None:
        forecast = self._forecasts.get(str(forecast_id))
        if forecast is None:
            return None
        return Forecast.from_snapshot(forecast.snapshot())

    def due(self, current_step: int) -> List[Forecast]:
        deadline = int(current_step)
        return [forecast for forecast in self.pending() if forecast.deadline_step <= deadline]

    def verify(self, forecast_id: str, actual_prob: float, verified_at: datetime) -> Forecast:
        if forecast_id not in self._forecasts:
            raise KeyError(forecast_id)
        forecast = self._forecasts[forecast_id]
        forecast.actual_prob = float(actual_prob)
        forecast.verified_at = verified_at.astimezone(timezone.utc).replace(microsecond=0)
        forecast.error = float(round(abs(forecast.predicted_prob - forecast.actual_prob), 12))
        forecast.status = "verified"
        self._maybe_save()
        return Forecast.from_snapshot(forecast.snapshot())

    def snapshot(self) -> List[dict]:
        return [forecast.snapshot() for forecast in self._forecasts.values()]

    def save(self, path: str | Path | None = None) -> Path:
        target = Path(path).resolve() if path is not None else self.json_path
        if target is None:
            raise ValueError("json_path is not configured for this ForecastRegistry")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(self.snapshot(), indent=2, sort_keys=True), encoding="utf-8")
        return target

    def load(self, path: str | Path | None = None) -> None:
        source = Path(path).resolve() if path is not None else self.json_path
        if source is None:
            raise ValueError("json_path is not configured for this ForecastRegistry")
        payload = json.loads(source.read_text(encoding="utf-8"))
        self._forecasts = {
            item["forecast_id"]: Forecast.from_snapshot(item)
            for item in payload
        }

    def _maybe_save(self) -> None:
        if self.auto_save and self.json_path is not None:
            self.save()


__all__ = ["Forecast", "ForecastRegistry"]
