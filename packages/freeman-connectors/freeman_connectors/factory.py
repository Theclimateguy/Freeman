"""Factory for config-driven Freeman connector construction."""

from __future__ import annotations

from typing import Any

from freeman_connectors.http import HTTPJSONSignalSource
from freeman_connectors.rss import ArxivSignalSource, RSSFeedSignalSource
from freeman_connectors.web import WebPageSignalSource


def build_signal_source(config: dict[str, Any]) -> Any:
    """Construct a signal source from a config mapping."""

    source_type = str(config.get("type", "")).strip().lower()
    if source_type == "http_json":
        return HTTPJSONSignalSource(
            url=config["url"],
            method=str(config.get("method", "GET")),
            params=dict(config.get("params", {})),
            headers=dict(config.get("headers", {})),
            json_body=config.get("json_body"),
            timeout_seconds=float(config.get("timeout_seconds", 30.0)),
            item_path=config.get("item_path"),
            field_map=dict(config.get("field_map", {})),
            source_type=str(config.get("source_type", "http_json")),
            default_topic=config.get("default_topic"),
            static_metadata=dict(config.get("static_metadata", {})),
        )
    if source_type in {"rss", "atom"}:
        return RSSFeedSignalSource(
            url=config["url"],
            default_topic=config.get("default_topic"),
            max_entries=int(config.get("max_entries", 20)),
            timeout_seconds=float(config.get("timeout_seconds", 30.0)),
            headers=dict(config.get("headers", {})),
            source_type=str(config.get("source_type", "rss")),
            static_metadata=dict(config.get("static_metadata", {})),
        )
    if source_type == "arxiv":
        return ArxivSignalSource(
            url=str(config.get("url", "http://export.arxiv.org/api/query")),
            query=str(config["query"]),
            default_topic=config.get("default_topic"),
            max_entries=int(config.get("max_entries", 20)),
            timeout_seconds=float(config.get("timeout_seconds", 30.0)),
            headers=dict(config.get("headers", {})),
            static_metadata=dict(config.get("static_metadata", {})),
            start=int(config.get("start", 0)),
            max_results=int(config.get("max_results", 20)),
            sort_by=str(config.get("sort_by", "submittedDate")),
            sort_order=str(config.get("sort_order", "descending")),
        )
    if source_type in {"web", "webpage", "html"}:
        return WebPageSignalSource(
            url=config["url"],
            default_topic=config.get("default_topic"),
            timeout_seconds=float(config.get("timeout_seconds", 30.0)),
            headers=dict(config.get("headers", {})),
            max_text_chars=int(config.get("max_text_chars", 4000)),
            source_type=str(config.get("source_type", "web")),
            static_metadata=dict(config.get("static_metadata", {})),
        )
    raise ValueError(f"Unsupported connector source type: {source_type}")


__all__ = ["build_signal_source"]
