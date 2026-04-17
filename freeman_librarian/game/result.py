"""Simulation result container."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List

from freeman_librarian.core.types import Violation
from freeman_librarian.utils import stable_json_dumps


@dataclass
class SimResult:
    """Serializable simulation output."""

    domain_id: str
    trajectory: List[Dict[str, Any]]
    outcome_probs: List[Dict[str, float]]
    final_outcome_probs: Dict[str, float]
    confidence: float
    violations: List[Violation]
    converged: bool
    steps_run: int
    metadata: Dict[str, Any] = field(default_factory=dict)

    def snapshot(self) -> Dict[str, Any]:
        """Return a JSON-serializable simulation result."""

        return {
            "domain_id": self.domain_id,
            "trajectory": self.trajectory,
            "outcome_probs": self.outcome_probs,
            "final_outcome_probs": self.final_outcome_probs,
            "confidence": self.confidence,
            "violations": [violation.snapshot() for violation in self.violations],
            "converged": self.converged,
            "steps_run": self.steps_run,
            "metadata": self.metadata,
        }

    def to_json(self) -> str:
        """Serialize the result to deterministic JSON."""

        return stable_json_dumps(self.snapshot())
