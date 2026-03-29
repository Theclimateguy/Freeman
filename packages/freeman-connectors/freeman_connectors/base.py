"""Shared helpers for universal Freeman signal connectors."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from typing import Any, Iterable, Sequence


def now_iso() -> str:
    """Return the current UTC timestamp in ISO format."""

    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def stable_signal_id(prefix: str, payload: Any) -> str:
    """Build a stable signal identifier from arbitrary JSON-like payload."""

    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str)
    digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]
    return f"{prefix}:{digest}"


def lookup_path(payload: Any, path: str | None) -> Any:
    """Resolve a dotted path inside a nested dict/list payload."""

    if path in {None, "", "."}:
        return payload
    current = payload
    for part in str(path).split("."):
        if isinstance(current, dict):
            if part not in current:
                return None
            current = current[part]
            continue
        if isinstance(current, list):
            try:
                index = int(part)
            except ValueError:
                return None
            if index < 0 or index >= len(current):
                return None
            current = current[index]
            continue
        return None
    return current


def ensure_sequence(value: Any) -> list[Any]:
    """Normalize scalars and collections into a list."""

    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    return [value]


def coerce_entities(value: Any) -> list[str]:
    """Normalize entity payloads into a list of strings."""

    entities = ensure_sequence(value)
    cleaned = []
    for entity in entities:
        token = str(entity).strip()
        if token:
            cleaned.append(token)
    return cleaned


def coerce_sentiment(value: Any) -> float:
    """Normalize sentiment values into a bounded float."""

    if value is None:
        return 0.0
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def coerce_text(value: Any, *, fallback: str = "") -> str:
    """Return compact text from arbitrary payloads."""

    if value is None:
        return fallback
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    return str(value).strip()


def extract_metadata(
    item: Any,
    *,
    consumed_paths: Iterable[str] = (),
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Return the remaining item payload as metadata."""

    metadata = dict(extra or {})
    if isinstance(item, dict):
        consumed_top_level = {path.split(".", 1)[0] for path in consumed_paths if path}
        for key, value in item.items():
            if key in consumed_top_level:
                continue
            metadata[key] = value
    else:
        metadata["raw_item"] = item
    return metadata


def ensure_item_list(payload: Any, *, item_path: str | None = None) -> list[Any]:
    """Extract one or more items from a fetched payload."""

    extracted = lookup_path(payload, item_path)
    if extracted is None:
        return []
    if isinstance(extracted, list):
        return extracted
    return [extracted]


def hostname_topic(url: str) -> str:
    """Infer a topic token from a URL hostname."""

    from urllib.parse import urlparse

    parsed = urlparse(url)
    return parsed.netloc or "web"


__all__ = [
    "coerce_entities",
    "coerce_sentiment",
    "coerce_text",
    "ensure_item_list",
    "ensure_sequence",
    "extract_metadata",
    "hostname_topic",
    "lookup_path",
    "now_iso",
    "stable_signal_id",
]
