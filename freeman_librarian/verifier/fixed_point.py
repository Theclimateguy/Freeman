"""Backward-compatible wrapper for ``freeman.verifier.fixedpoint``."""

from freeman_librarian.verifier.fixedpoint import (
    FixedPointResult,
    apply_corrections,
    compute_corrections,
    find_fixed_point,
    iterate_fixed_point,
)

__all__ = [
    "FixedPointResult",
    "apply_corrections",
    "compute_corrections",
    "find_fixed_point",
    "iterate_fixed_point",
]
