"""Verifier package with lazy re-exports to avoid import cycles."""

from __future__ import annotations

from typing import Any

__all__ = [
    "FixedPointResult",
    "VerificationReport",
    "Verifier",
    "VerifierConfig",
    "iterate_fixed_point",
]


def __getattr__(name: str) -> Any:
    if name in {"FixedPointResult", "iterate_fixed_point"}:
        from freeman.verifier.fixedpoint import FixedPointResult, iterate_fixed_point

        mapping = {
            "FixedPointResult": FixedPointResult,
            "iterate_fixed_point": iterate_fixed_point,
        }
        return mapping[name]

    if name == "VerificationReport":
        from freeman.verifier.report import VerificationReport

        return VerificationReport

    if name in {"Verifier", "VerifierConfig"}:
        from freeman.verifier.verifier import Verifier, VerifierConfig

        mapping = {"Verifier": Verifier, "VerifierConfig": VerifierConfig}
        return mapping[name]

    raise AttributeError(name)
