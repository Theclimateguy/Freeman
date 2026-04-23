"""Minimal signal ingestion for the Freeman lite runtime."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import re
from typing import Iterable

from freeman.core.world import WorldState


def _normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", str(text).strip().lower())


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", _normalize_text(text))


@dataclass
class SignalDecision:
    """Deterministic WATCH vs ANALYZE routing decision."""

    signal_id: str
    signal_hash: str
    mode: str
    relevance_score: float
    keyword_hits: list[str]
    duplicate: bool
    reason: str

    def snapshot(self) -> dict[str, object]:
        return {
            "signal_id": self.signal_id,
            "signal_hash": self.signal_hash,
            "mode": self.mode,
            "relevance_score": float(self.relevance_score),
            "keyword_hits": list(self.keyword_hits),
            "duplicate": bool(self.duplicate),
            "reason": self.reason,
        }


class SignalIngestionEngine:
    """Process signals in arrival order with a keyword gate and dedupe check."""

    def __init__(
        self,
        *,
        keywords: Iterable[str] = (),
        min_keyword_hits: int = 1,
    ) -> None:
        self.keywords = tuple(str(item).strip().lower() for item in keywords if str(item).strip())
        self.min_keyword_hits = max(int(min_keyword_hits), 0)

    def classify(self, signal_text: str, *, processed_hashes: Iterable[str] = ()) -> SignalDecision:
        normalized = _normalize_text(signal_text)
        signal_hash = hashlib.sha1(normalized.encode("utf-8")).hexdigest()
        keyword_hits = self._keyword_hits(normalized)
        relevance = self._relevance_score(keyword_hits)
        duplicate = signal_hash in {str(item) for item in processed_hashes}

        if duplicate:
            return SignalDecision(
                signal_id=f"sig:{signal_hash[:12]}",
                signal_hash=signal_hash,
                mode="WATCH",
                relevance_score=float(relevance),
                keyword_hits=keyword_hits,
                duplicate=True,
                reason="duplicate_signal",
            )
        if self.keywords and len(keyword_hits) < self.min_keyword_hits:
            return SignalDecision(
                signal_id=f"sig:{signal_hash[:12]}",
                signal_hash=signal_hash,
                mode="WATCH",
                relevance_score=float(relevance),
                keyword_hits=keyword_hits,
                duplicate=False,
                reason="below_keyword_threshold",
            )
        return SignalDecision(
            signal_id=f"sig:{signal_hash[:12]}",
            signal_hash=signal_hash,
            mode="ANALYZE",
            relevance_score=float(relevance),
            keyword_hits=keyword_hits,
            duplicate=False,
            reason="keyword_match" if keyword_hits else "no_keywords_configured",
        )

    def remember(self, world: WorldState, signal_hash: str, *, max_history: int = 256) -> None:
        history = [str(item) for item in world.metadata.get("lite_signal_hashes", []) if str(item)]
        history.append(str(signal_hash))
        world.metadata["lite_signal_hashes"] = history[-max(int(max_history), 1) :]

    def _keyword_hits(self, normalized_text: str) -> list[str]:
        if not self.keywords:
            return []
        tokens = set(_tokenize(normalized_text))
        hits = []
        for keyword in self.keywords:
            keyword_tokens = tuple(_tokenize(keyword))
            if not keyword_tokens:
                continue
            if len(keyword_tokens) == 1 and keyword_tokens[0] in tokens:
                hits.append(keyword)
                continue
            if keyword in normalized_text:
                hits.append(keyword)
        return sorted(set(hits))

    def _relevance_score(self, keyword_hits: list[str]) -> float:
        if not self.keywords:
            return 1.0
        return float(len(keyword_hits) / max(len(self.keywords), 1))


__all__ = ["SignalDecision", "SignalIngestionEngine"]
