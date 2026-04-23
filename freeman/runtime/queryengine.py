"""Lexical runtime retrieval for the Freeman lite knowledge graph."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import re
from typing import Any, List, Sequence

from freeman.memory.knowledgegraph import KGNode, KnowledgeGraph


@dataclass
class RuntimeQueryResult:
    """Structured lexical query result."""

    query: str
    matched: bool
    matches: list[dict[str, Any]]
    ranking_trace: list[dict[str, Any]]
    no_match_reason: str | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "query": self.query,
            "matched": self.matched,
            "count": len(self.matches),
            "matches": self.matches,
            "ranking_trace": list(self.ranking_trace),
            "no_match_reason": self.no_match_reason,
        }


def _semantic_tokens(text: str) -> List[str]:
    normalized = str(text).lower().replace("_", " ")
    return re.findall(r"[a-z0-9]+", normalized)


def _semantic_ngrams(tokens: Sequence[str], *, size: int) -> set[str]:
    values = [str(token) for token in tokens if str(token)]
    if len(values) < size:
        return set()
    return {" ".join(values[index : index + size]) for index in range(len(values) - size + 1)}


def _lexical_score(query_text: str, document: str) -> float:
    haystack = str(document).strip().lower()
    query_text = str(query_text).strip().lower()
    if not haystack or not query_text:
        return 0.0
    score = 0.0
    if query_text in haystack:
        score += 1.0
    query_tokens = _semantic_tokens(query_text)
    haystack_token_list = _semantic_tokens(haystack)
    haystack_tokens = set(haystack_token_list)
    if query_tokens:
        overlap = sum(1 for token in query_tokens if token in haystack_tokens)
        score += 0.6 * (overlap / len(query_tokens))
    query_bigrams = _semantic_ngrams(query_tokens, size=2)
    haystack_bigrams = _semantic_ngrams(haystack_token_list, size=2)
    if query_bigrams:
        overlap = len(query_bigrams & haystack_bigrams)
        score += 0.3 * (overlap / len(query_bigrams))
    query_trigrams = _semantic_ngrams(query_tokens, size=3)
    haystack_trigrams = _semantic_ngrams(haystack_token_list, size=3)
    if query_trigrams:
        overlap = len(query_trigrams & haystack_trigrams)
        score += 0.2 * (overlap / len(query_trigrams))
    return float(score)


def _parse_iso(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        return None


def _recency_bonus(node: KGNode) -> float:
    updated_at = _parse_iso(node.updated_at) or _parse_iso(node.created_at)
    if updated_at is None:
        return 0.0
    age_seconds = max((datetime.now(timezone.utc) - updated_at.astimezone(timezone.utc)).total_seconds(), 0.0)
    age_days = age_seconds / 86400.0
    return max(0.0, 0.25 - min(age_days / 120.0, 0.25))


def _node_document(node: KGNode) -> str:
    return " ".join(
        [
            str(node.id),
            str(node.label),
            str(node.content),
            " ".join(str(item) for item in node.evidence),
            " ".join(str(item) for item in node.sources),
        ]
    )


class RuntimeQueryEngine:
    """Deterministic lexical query engine for the persisted knowledge graph."""

    def __init__(self, knowledge_graph: KnowledgeGraph) -> None:
        self.knowledge_graph = knowledge_graph

    def query(self, query_text: str, *, top_k: int = 10) -> RuntimeQueryResult:
        query_text = str(query_text).strip()
        candidates: list[tuple[float, KGNode, dict[str, Any]]] = []
        for node in self.knowledge_graph.nodes(lazy_embed=False):
            if node.status == "archived":
                continue
            lexical = _lexical_score(query_text, _node_document(node))
            confidence_bonus = 0.15 * float(node.confidence)
            freshness_bonus = _recency_bonus(node)
            total_score = lexical + confidence_bonus + freshness_bonus
            if query_text and lexical <= 0.0:
                continue
            candidates.append(
                (
                    total_score,
                    node,
                    {
                        "node_id": node.id,
                        "lexical_score": float(lexical),
                        "confidence_bonus": float(confidence_bonus),
                        "recency_bonus": float(freshness_bonus),
                        "total_score": float(total_score),
                    },
                )
            )

        candidates.sort(key=lambda item: (-item[0], -item[1].confidence, item[1].updated_at, item[1].id))
        matches = []
        ranking_trace = []
        for _score, node, trace in candidates[: max(int(top_k), 1)]:
            matches.append(node.snapshot())
            ranking_trace.append(trace)
        return RuntimeQueryResult(
            query=query_text,
            matched=bool(matches),
            matches=matches,
            ranking_trace=ranking_trace,
            no_match_reason=None if matches else "no lexical matches above threshold",
        )


__all__ = ["RuntimeQueryEngine", "RuntimeQueryResult"]
