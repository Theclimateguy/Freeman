"""Verification report container."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List

from freeman.core.types import Violation
from freeman.utils import stable_json_dumps


@dataclass
class VerificationReport:
    """Aggregated result of one or more verification levels."""

    world_id: str
    domain_id: str
    levels_run: List[int]
    violations: List[Violation]
    passed: bool
    metadata: Dict[str, Any] = field(default_factory=dict)

    def snapshot(self) -> Dict[str, Any]:
        """Return a JSON-serializable report snapshot."""

        return {
            "world_id": self.world_id,
            "domain_id": self.domain_id,
            "levels_run": list(self.levels_run),
            "violations": [violation.snapshot() for violation in self.violations],
            "passed": self.passed,
            "metadata": self.metadata,
        }

    def to_json(self) -> str:
        """Serialize the report to deterministic JSON."""

        return stable_json_dumps(self.snapshot())
