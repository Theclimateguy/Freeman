"""Signal ingestion, anomaly detection, and trigger logic."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from math import exp, log
from typing import Any, Iterable, List, Sequence

import numpy as np


def _now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _parse_timestamp(value: str | datetime) -> datetime:
    if isinstance(value, datetime):
        return value.astimezone(timezone.utc).replace(microsecond=0)
    return datetime.fromisoformat(value).astimezone(timezone.utc).replace(microsecond=0)


@dataclass
class Signal:
    """Normalized incoming signal."""

    signal_id: str
    source_type: str
    text: str
    topic: str
    entities: List[str] = field(default_factory=list)
    sentiment: float = 0.0
    timestamp: str = field(default_factory=_now_iso)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class SignalRecord:
    """One persisted signal observation across sessions."""

    signal_id: str
    topic: str
    last_seen: datetime
    times_seen: int = 1
    last_trigger_mode: str = "WATCH"

    def __post_init__(self) -> None:
        self.last_seen = _parse_timestamp(self.last_seen)
        self.times_seen = int(self.times_seen)

    def snapshot(self) -> dict[str, Any]:
        return {
            "signal_id": self.signal_id,
            "topic": self.topic,
            "last_seen": self.last_seen.isoformat(),
            "times_seen": self.times_seen,
            "last_trigger_mode": self.last_trigger_mode,
        }

    @classmethod
    def from_snapshot(cls, data: dict[str, Any]) -> "SignalRecord":
        return cls(
            signal_id=data["signal_id"],
            topic=data["topic"],
            last_seen=data["last_seen"],
            times_seen=int(data.get("times_seen", 1)),
            last_trigger_mode=data.get("last_trigger_mode", "WATCH"),
        )


class SignalMemory:
    """Cross-session deduplication and exponential decay of signal weights."""

    def __init__(self, decay_halflife_hours: float = 24.0) -> None:
        self._records: dict[str, SignalRecord] = {}
        self.decay_halflife_hours = float(decay_halflife_hours)

    def see(self, signal: Signal, trigger_mode: str) -> SignalRecord:
        seen_at = _parse_timestamp(signal.timestamp)
        record = self._records.get(signal.signal_id)
        if record is None:
            record = SignalRecord(
                signal_id=signal.signal_id,
                topic=signal.topic,
                last_seen=seen_at,
                times_seen=1,
                last_trigger_mode=trigger_mode,
            )
            self._records[signal.signal_id] = record
            return SignalRecord.from_snapshot(record.snapshot())

        record.topic = signal.topic
        record.last_seen = seen_at
        record.times_seen += 1
        record.last_trigger_mode = trigger_mode
        return SignalRecord.from_snapshot(record.snapshot())

    def is_duplicate(self, signal: Signal, *, within_hours: float = 1.0) -> bool:
        """Return True when the same signal was seen within the duplicate window."""

        record = self._records.get(signal.signal_id)
        if record is None:
            return False
        seen_at = _parse_timestamp(signal.timestamp)
        elapsed_hours = max((seen_at - record.last_seen).total_seconds() / 3600.0, 0.0)
        return elapsed_hours <= float(within_hours)

    def effective_weight(self, signal_id: str, now: datetime | None = None) -> float:
        """Return exponentially decayed signal weight."""

        record = self._records.get(signal_id)
        if record is None:
            return 0.0
        now_dt = _parse_timestamp(now or datetime.now(timezone.utc))
        elapsed_hours = max((now_dt - record.last_seen).total_seconds() / 3600.0, 0.0)
        return float(exp(-log(2.0) * elapsed_hours / max(self.decay_halflife_hours, 1.0e-8)))

    def snapshot(self) -> list[dict[str, Any]]:
        return [record.snapshot() for record in self._records.values()]

    def load_snapshot(self, records: list[dict[str, Any]]) -> None:
        self._records = {
            record["signal_id"]: SignalRecord.from_snapshot(record)
            for record in records
        }


@dataclass
class ShockClassification:
    """Semantic interpretation of a signal."""

    shock_type: str
    severity: float
    semantic_gap: float
    rationale: str = ""


@dataclass
class SignalTrigger:
    """Trigger decision for a signal."""

    signal_id: str
    mahalanobis_score: float
    classification: ShockClassification
    mode: str


class ManualSignalSource:
    """Manual signals provided directly by the caller."""

    def __init__(self, signals: Iterable[Signal | dict[str, Any]]) -> None:
        self.signals = [signal if isinstance(signal, Signal) else Signal(**signal) for signal in signals]

    def fetch(self) -> List[Signal]:
        return list(self.signals)


class RSSSignalSource(ManualSignalSource):
    """RSS-backed source over already fetched items."""

    def __init__(self, items: Iterable[dict[str, Any]]) -> None:
        super().__init__(
            Signal(
                signal_id=item["signal_id"],
                source_type="rss",
                text=item["text"],
                topic=item.get("topic", "rss"),
                entities=list(item.get("entities", [])),
                sentiment=float(item.get("sentiment", 0.0)),
                timestamp=item.get("timestamp", _now_iso()),
                metadata={k: v for k, v in item.items() if k not in {"signal_id", "text", "topic", "entities", "sentiment", "timestamp"}},
            )
            for item in items
        )


class TavilySignalSource(ManualSignalSource):
    """Tavily-backed source over already fetched items."""

    def __init__(self, items: Iterable[dict[str, Any]]) -> None:
        super().__init__(
            Signal(
                signal_id=item["signal_id"],
                source_type="tavily",
                text=item["text"],
                topic=item.get("topic", "search"),
                entities=list(item.get("entities", [])),
                sentiment=float(item.get("sentiment", 0.0)),
                timestamp=item.get("timestamp", _now_iso()),
                metadata={k: v for k, v in item.items() if k not in {"signal_id", "text", "topic", "entities", "sentiment", "timestamp"}},
            )
            for item in items
        )


class SignalIngestionEngine:
    """Normalize signals, compute anomaly scores, and decide trigger modes."""

    def feature_matrix(self, signals: Sequence[Signal]) -> np.ndarray:
        rows = []
        for signal in signals:
            rows.append(
                [
                    float(signal.sentiment),
                    float(len(signal.entities)),
                    float(len(signal.text.split())),
                    float(len(signal.topic)),
                ]
            )
        return np.array(rows, dtype=np.float64) if rows else np.zeros((0, 4), dtype=np.float64)

    def mahalanobis_scores(self, signals: Sequence[Signal], ridge: float = 1.0e-6) -> List[float]:
        """Return Mahalanobis distances for all signals."""

        matrix = self.feature_matrix(signals)
        if len(matrix) == 0:
            return []
        center = matrix.mean(axis=0)
        if len(matrix) == 1:
            return [0.0]
        covariance = np.cov(matrix, rowvar=False)
        precision = np.linalg.pinv(covariance + ridge * np.eye(covariance.shape[0], dtype=np.float64))
        scores: List[float] = []
        for row in matrix:
            delta = row - center
            distance = float(np.sqrt(delta.T @ precision @ delta))
            scores.append(distance)
        return scores

    def classify_shock(self, signal: Signal, classifier: Any | None = None) -> ShockClassification:
        """Classify a signal into a shock type using an injected classifier or heuristics."""

        if callable(classifier):
            result = classifier(signal)
            if isinstance(result, ShockClassification):
                return result
            return ShockClassification(**result)

        if classifier is not None and hasattr(classifier, "chat_json"):
            prompt = (
                "Classify this signal into shock_type, severity in [0,1], semantic_gap in [0,1], "
                f"and rationale: {signal.text}"
            )
            result = classifier.chat_json([{"role": "user", "content": prompt}], temperature=0.0, max_tokens=300)
            return ShockClassification(**result)

        text = signal.text.lower()
        if any(keyword in text for keyword in ["crisis", "collapse", "war", "default", "flood", "drought"]):
            return ShockClassification("shock", 0.9, 0.9, "Keyword-based severe shock heuristic.")
        if any(keyword in text for keyword in ["policy", "tariff", "regulation", "sanction"]):
            return ShockClassification("policy_shift", 0.7, 0.6, "Keyword-based policy-shift heuristic.")
        return ShockClassification("routine", 0.2, 0.1, "No strong shock keywords detected.")

    def trigger_mode(
        self,
        mahalanobis_score: float,
        classification: ShockClassification,
        *,
        anomaly_lambda: float = 3.0,
        semantic_threshold: float = 0.5,
    ) -> str:
        """Map statistical and semantic triggers into WATCH / ANALYZE / DEEP_DIVE."""

        stat_trigger = mahalanobis_score >= anomaly_lambda
        semantic_trigger = classification.semantic_gap >= semantic_threshold
        if stat_trigger and semantic_trigger:
            return "DEEP_DIVE"
        if stat_trigger or semantic_trigger or classification.severity >= 0.5:
            return "ANALYZE"
        return "WATCH"

    def ingest(
        self,
        source: ManualSignalSource,
        *,
        classifier: Any | None = None,
        signal_memory: SignalMemory | None = None,
        skip_duplicates_within_hours: float = 1.0,
        anomaly_lambda: float = 3.0,
        semantic_threshold: float = 0.5,
    ) -> List[SignalTrigger]:
        """Fetch, score, classify, and trigger on a batch of signals."""

        signals = source.fetch()
        scores = self.mahalanobis_scores(signals)
        triggers: List[SignalTrigger] = []
        for signal, score in zip(signals, scores, strict=False):
            if signal_memory is not None and signal_memory.is_duplicate(
                signal,
                within_hours=skip_duplicates_within_hours,
            ):
                continue
            classification = self.classify_shock(signal, classifier=classifier)
            mode = self.trigger_mode(
                score,
                classification,
                anomaly_lambda=anomaly_lambda,
                semantic_threshold=semantic_threshold,
            )
            if signal_memory is not None:
                signal_memory.see(signal, mode)
            triggers.append(
                SignalTrigger(
                    signal_id=signal.signal_id,
                    mahalanobis_score=score,
                    classification=classification,
                    mode=mode,
                )
            )
        return triggers


__all__ = [
    "ManualSignalSource",
    "RSSSignalSource",
    "ShockClassification",
    "Signal",
    "SignalMemory",
    "SignalRecord",
    "SignalIngestionEngine",
    "SignalTrigger",
    "TavilySignalSource",
]
