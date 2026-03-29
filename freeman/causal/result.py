"""Structured results for data-driven causal edge estimation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterator, Mapping

from freeman.utils import json_ready

EdgeKey = tuple[str, str]


@dataclass
class EstimationResult(Mapping[EdgeKey, float]):
    """Point estimates, confidence intervals, and provenance for causal edges."""

    weights: Dict[EdgeKey, float]
    confidence_intervals: Dict[EdgeKey, tuple[float, float]] = field(default_factory=dict)
    edge_metadata: Dict[EdgeKey, Dict[str, Any]] = field(default_factory=dict)
    backend: str = "t_learner"
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __getitem__(self, key: EdgeKey) -> float:
        return self.weights[key]

    def __iter__(self) -> Iterator[EdgeKey]:
        return iter(self.weights)

    def __len__(self) -> int:
        return len(self.weights)

    def to_weight_dict(self) -> Dict[EdgeKey, float]:
        """Return a shallow copy of the estimated edge weights."""

        return dict(self.weights)

    def snapshot(self) -> Dict[str, Any]:
        """Return a JSON-serializable result payload."""

        return {
            "weights": {f"{source}->{target}": value for (source, target), value in self.weights.items()},
            "confidence_intervals": {
                f"{source}->{target}": [low, high]
                for (source, target), (low, high) in self.confidence_intervals.items()
            },
            "edge_metadata": {
                f"{source}->{target}": json_ready(payload)
                for (source, target), payload in self.edge_metadata.items()
            },
            "backend": self.backend,
            "metadata": json_ready(self.metadata),
        }

