"""Connector-specific tests for the released freeman-connectors package."""

from __future__ import annotations

from freeman_connectors import (
    HTTPJSONSignalSource,
    RSSFeedSignalSource,
    WebPageSignalSource,
    build_signal_source,
)


class _JSONResponse:
    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self) -> None:
        return None

    def json(self):
        return self._payload


class _TextResponse:
    def __init__(self, *, text: str = "", content: bytes | None = None, headers: dict[str, str] | None = None):
        self.text = text
        self.content = content if content is not None else text.encode("utf-8")
        self.headers = headers or {}

    def raise_for_status(self) -> None:
        return None


def test_build_signal_source_supports_all_runtime_connector_types() -> None:
    rss = build_signal_source({"type": "rss", "url": "https://example.com/feed.xml"})
    http = build_signal_source({"type": "http_json", "url": "https://example.com/api"})
    web = build_signal_source({"type": "web", "url": "https://example.com/page"})

    assert isinstance(rss, RSSFeedSignalSource)
    assert isinstance(http, HTTPJSONSignalSource)
    assert isinstance(web, WebPageSignalSource)


def test_http_json_signal_source_maps_items(monkeypatch) -> None:
    payload = {
        "items": [
            {
                "id": "evt-1",
                "headline": "River treaty breaks down",
                "published_at": "2026-04-13T10:00:00+00:00",
                "entities": ["country_a", "country_b"],
                "category": "water_conflict",
                "score": -0.4,
                "extra_field": "kept in metadata",
            }
        ]
    }

    monkeypatch.setattr(
        "freeman_connectors.http.requests.request",
        lambda **kwargs: _JSONResponse(payload),
    )
    source = HTTPJSONSignalSource(
        url="https://example.com/api/events",
        item_path="items",
        field_map={
            "signal_id": "id",
            "text": "headline",
            "timestamp": "published_at",
            "entities": "entities",
            "topic": "category",
            "sentiment": "score",
        },
    )

    signals = source.fetch()

    assert len(signals) == 1
    assert signals[0].signal_id == "evt-1"
    assert signals[0].topic == "water_conflict"
    assert signals[0].entities == ["country_a", "country_b"]
    assert signals[0].metadata["extra_field"] == "kept in metadata"


def test_rss_feed_signal_source_parses_feed_entries(monkeypatch) -> None:
    rss_text = """<?xml version="1.0" encoding="UTF-8"?>
<rss version="2.0">
  <channel>
    <title>Climate Feed</title>
    <item>
      <guid>climate-1</guid>
      <title>Climate policy tightens</title>
      <description>New climate law accelerates mitigation spending.</description>
      <pubDate>Sun, 13 Apr 2026 12:00:00 GMT</pubDate>
      <link>https://example.com/climate-1</link>
    </item>
  </channel>
</rss>
"""
    monkeypatch.setattr(
        "freeman_connectors.rss.requests.get",
        lambda *args, **kwargs: _TextResponse(content=rss_text.encode("utf-8")),
    )
    source = RSSFeedSignalSource(url="https://example.com/feed.xml", source_type="rss_test")

    signals = source.fetch()

    assert len(signals) == 1
    assert signals[0].signal_id == "climate-1"
    assert signals[0].topic == "Climate Feed"
    assert "Climate policy tightens" in signals[0].text
    assert signals[0].metadata["connector_type"] == "rss"


def test_web_page_signal_source_extracts_text(monkeypatch) -> None:
    html = """
    <html>
      <head><title>Climate Update</title></head>
      <body>
        <p>Heat stress rises across southern Europe.</p>
        <p>Adaptation budgets remain constrained.</p>
      </body>
    </html>
    """
    monkeypatch.setattr(
        "freeman_connectors.web.requests.get",
        lambda *args, **kwargs: _TextResponse(text=html, headers={"Content-Type": "text/html"}),
    )
    source = WebPageSignalSource(url="https://example.com/page")

    signals = source.fetch()

    assert len(signals) == 1
    assert signals[0].topic == "example.com"
    assert "Climate Update" in signals[0].text
    assert "Heat stress rises across southern Europe." in signals[0].text
