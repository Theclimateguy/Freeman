"""Verifier package with lazy re-exports to avoid import cycles."""

from __future__ import annotations

from typing import Any

__all__ = [
    "FixedPointResult",
    "level3_check",
    "VerificationReport",
    "Verifier",
    "VerifierConfig",
    "iterate_fixed_point",
]


def __getattr__(name: str) -> Any:
    if name in {"FixedPointResult", "iterate_fixed_point"}:
        from freeman_librarian.verifier.fixedpoint import FixedPointResult, iterate_fixed_point

        mapping = {
            "FixedPointResult": FixedPointResult,
            "iterate_fixed_point": iterate_fixed_point,
        }
        return mapping[name]

    if name == "VerificationReport":
        from freeman_librarian.verifier.report import VerificationReport

        return VerificationReport

    if name == "level3_check":
        from freeman_librarian.verifier.level3 import level3_check

        return level3_check

    if name in {"Verifier", "VerifierConfig"}:
        from freeman_librarian.verifier.verifier import Verifier, VerifierConfig

        mapping = {"Verifier": Verifier, "VerifierConfig": VerifierConfig}
        return mapping[name]

    raise AttributeError(name)
