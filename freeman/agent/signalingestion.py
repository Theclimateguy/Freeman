"""Signal ingestion, anomaly detection, and trigger logic."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Iterable, List, Sequence

import numpy as np


def _now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


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
        anomaly_lambda: float = 3.0,
        semantic_threshold: float = 0.5,
    ) -> List[SignalTrigger]:
        """Fetch, score, classify, and trigger on a batch of signals."""

        signals = source.fetch()
        scores = self.mahalanobis_scores(signals)
        triggers: List[SignalTrigger] = []
        for signal, score in zip(signals, scores, strict=False):
            classification = self.classify_shock(signal, classifier=classifier)
            triggers.append(
                SignalTrigger(
                    signal_id=signal.signal_id,
                    mahalanobis_score=score,
                    classification=classification,
                    mode=self.trigger_mode(
                        score,
                        classification,
                        anomaly_lambda=anomaly_lambda,
                        semantic_threshold=semantic_threshold,
                    ),
                )
            )
        return triggers


__all__ = [
    "ManualSignalSource",
    "RSSSignalSource",
    "ShockClassification",
    "Signal",
    "SignalIngestionEngine",
    "SignalTrigger",
    "TavilySignalSource",
]
