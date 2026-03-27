"""Package-wide exceptions."""

from __future__ import annotations

from typing import Iterable

from freeman.core.types import Violation


class HardStopException(RuntimeError):
    """Raised when a hard verifier violation requires stopping the simulation."""

    def __init__(self, violations: Iterable[Violation]):
        self.violations = list(violations)
        message = "; ".join(v.description for v in self.violations) or "Hard stop"
        super().__init__(message)


class ValidationError(ValueError):
    """Raised when a domain schema is invalid."""
