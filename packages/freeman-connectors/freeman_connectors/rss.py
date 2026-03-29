"""RSS and Atom signal sources for Freeman."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import feedparser
import requests

from freeman.agent.signalingestion import Signal

from freeman_connectors.base import coerce_entities, coerce_text, hostname_topic, now_iso, stable_signal_id


def _entry_authors(entry: dict[str, Any]) -> list[str]:
    authors = entry.get("authors", [])
    if isinstance(authors, list):
        return [str(author.get("name", "")).strip() for author in authors if str(author.get("name", "")).strip()]
    return []


@dataclass
class RSSFeedSignalSource:
    """Fetch an RSS or Atom feed and emit Freeman signals."""

    url: str
    default_topic: str | None = None
    max_entries: int = 20
    timeout_seconds: float = 30.0
    headers: dict[str, str] = field(default_factory=dict)
    source_type: str = "rss"
    static_metadata: dict[str, Any] = field(default_factory=dict)

    def fetch(self) -> list[Signal]:
        """Fetch and normalize feed entries."""

        response = requests.get(self.url, headers=self.headers or None, timeout=self.timeout_seconds)
        response.raise_for_status()
        feed = feedparser.parse(response.content)
        feed_title = coerce_text(feed.feed.get("title"), fallback=hostname_topic(self.url))
        topic = self.default_topic or feed_title
        signals: list[Signal] = []
        for index, entry in enumerate(feed.entries[: self.max_entries]):
            title = coerce_text(entry.get("title"))
            summary = coerce_text(entry.get("summary"))
            text = "\n\n".join(part for part in [title, summary] if part).strip() or coerce_text(entry)
            published = (
                coerce_text(entry.get("published"))
                or coerce_text(entry.get("updated"))
                or now_iso()
            )
            link = coerce_text(entry.get("link"))
            signals.append(
                Signal(
                    signal_id=coerce_text(
                        entry.get("id"),
                        fallback=stable_signal_id(f"{self.source_type}:{self.url}:{index}", {"title": title, "link": link}),
                    ),
                    source_type=self.source_type,
                    text=text,
                    topic=topic,
                    entities=coerce_entities(_entry_authors(entry)),
                    sentiment=0.0,
                    timestamp=published,
                    metadata={
                        **self.static_metadata,
                        "connector_type": "rss",
                        "feed_title": feed_title,
                        "feed_url": self.url,
                        "link": link,
                        "tags": [tag.get("term") for tag in entry.get("tags", []) if tag.get("term")],
                        "entry_index": index,
                    },
                )
            )
        return signals


@dataclass
class ArxivSignalSource(RSSFeedSignalSource):
    """Thin arXiv adapter built on top of the Atom feed API."""

    query: str = ""
    start: int = 0
    max_results: int = 20
    sort_by: str = "submittedDate"
    sort_order: str = "descending"
    source_type: str = "arxiv"

    def __post_init__(self) -> None:
        if not self.query:
            raise ValueError("ArxivSignalSource requires a non-empty query.")
        if not self.url:
            self.url = "http://export.arxiv.org/api/query"

    def fetch(self) -> list[Signal]:
        params = {
            "search_query": self.query,
            "start": self.start,
            "max_results": self.max_results,
            "sortBy": self.sort_by,
            "sortOrder": self.sort_order,
        }
        response = requests.get(self.url, params=params, headers=self.headers or None, timeout=self.timeout_seconds)
        response.raise_for_status()
        feed = feedparser.parse(response.content)
        topic = self.default_topic or f"arxiv:{self.query}"
        signals: list[Signal] = []
        for index, entry in enumerate(feed.entries[: self.max_entries]):
            title = coerce_text(entry.get("title"))
            summary = coerce_text(entry.get("summary"))
            authors = _entry_authors(entry)
            categories = [tag.get("term") for tag in entry.get("tags", []) if tag.get("term")]
            text = "\n\n".join(part for part in [title, summary] if part).strip()
            signals.append(
                Signal(
                    signal_id=coerce_text(
                        entry.get("id"),
                        fallback=stable_signal_id(
                            f"{self.source_type}:{self.query}:{index}",
                            {"title": title, "authors": authors},
                        ),
                    ),
                    source_type=self.source_type,
                    text=text,
                    topic=topic,
                    entities=coerce_entities([*authors, *categories]),
                    sentiment=0.0,
                    timestamp=coerce_text(entry.get("published"), fallback=now_iso()),
                    metadata={
                        **self.static_metadata,
                        "connector_type": "arxiv",
                        "query": self.query,
                        "link": coerce_text(entry.get("link")),
                        "authors": authors,
                        "categories": categories,
                        "entry_index": index,
                    },
                )
            )
        return signals


__all__ = ["ArxivSignalSource", "RSSFeedSignalSource"]
