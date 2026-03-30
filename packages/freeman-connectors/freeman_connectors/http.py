"""Universal HTTP/JSON signal source for Freeman."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from typing import Any

import requests

from freeman.agent.signalingestion import Signal

from freeman_connectors.base import (
    coerce_entities,
    coerce_sentiment,
    coerce_text,
    ensure_item_list,
    extract_metadata,
    hostname_topic,
    lookup_path,
    now_iso,
    stable_signal_id,
)


@dataclass
class HTTPJSONSignalSource:
    """Fetch an arbitrary HTTP/JSON endpoint and map items into Freeman signals."""

    url: str
    method: str = "GET"
    params: dict[str, Any] = field(default_factory=dict)
    headers: dict[str, str] = field(default_factory=dict)
    json_body: dict[str, Any] | None = None
    timeout_seconds: float = 30.0
    item_path: str | None = None
    field_map: dict[str, str] = field(default_factory=dict)
    source_type: str = "http_json"
    default_topic: str | None = None
    static_metadata: dict[str, Any] = field(default_factory=dict)

    def _mapped_value(self, item: Any, field_name: str) -> Any:
        """Return a mapped payload value or None when the field is not configured."""

        path = self.field_map.get(field_name)
        if path in {None, ""}:
            return None
        return lookup_path(item, path)

    def _request_identity(self) -> str:
        """Return a stable request fingerprint for signal IDs across similar endpoints."""

        request_payload = {
            "method": self.method.upper(),
            "url": self.url,
            "params": self.params,
            "headers": self.headers,
            "json_body": self.json_body,
            "item_path": self.item_path,
            "default_topic": self.default_topic,
        }
        raw = json.dumps(request_payload, ensure_ascii=False, sort_keys=True, default=str)
        return stable_signal_id(f"{self.source_type}:request", raw)

    def fetch_payload(self) -> Any:
        """Fetch and parse the endpoint JSON payload."""

        response = requests.request(
            method=self.method.upper(),
            url=self.url,
            params=self.params or None,
            headers=self.headers or None,
            json=self.json_body,
            timeout=self.timeout_seconds,
        )
        response.raise_for_status()
        return response.json()

    def fetch(self) -> list[Signal]:
        """Return normalized Freeman signals from the endpoint payload."""

        payload = self.fetch_payload()
        items = ensure_item_list(payload, item_path=self.item_path)
        if not items and self.item_path in {None, "", "."} and payload is not None:
            items = [payload]
        topic = self.default_topic or hostname_topic(self.url)
        request_identity = self._request_identity()
        signals: list[Signal] = []
        for index, item in enumerate(items):
            consumed_paths = [path for path in self.field_map.values() if path]
            signal_id = coerce_text(
                self._mapped_value(item, "signal_id"),
                fallback=stable_signal_id(f"{self.source_type}:{request_identity}:{index}", item),
            )
            text_value = self._mapped_value(item, "text")
            text = coerce_text(
                text_value,
                fallback=coerce_text(item),
            )
            timestamp = coerce_text(
                self._mapped_value(item, "timestamp"),
                fallback=now_iso(),
            )
            signals.append(
                Signal(
                    signal_id=signal_id,
                    source_type=self.source_type,
                    text=text,
                    topic=coerce_text(self._mapped_value(item, "topic"), fallback=topic),
                    entities=coerce_entities(self._mapped_value(item, "entities")),
                    sentiment=coerce_sentiment(self._mapped_value(item, "sentiment")),
                    timestamp=timestamp,
                    metadata=extract_metadata(
                        item,
                        consumed_paths=consumed_paths,
                        extra={
                            **self.static_metadata,
                            "connector_type": "http_json",
                            "request_url": self.url,
                            "request_method": self.method.upper(),
                            "item_index": index,
                        },
                    ),
                )
            )
        return signals


__all__ = ["HTTPJSONSignalSource"]
