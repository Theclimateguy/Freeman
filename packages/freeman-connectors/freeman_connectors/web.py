"""Generic HTML/webpage connector for Freeman."""

from __future__ import annotations

from dataclasses import dataclass, field

import requests
from bs4 import BeautifulSoup

from freeman.agent.signalingestion import Signal

from freeman_connectors.base import coerce_text, hostname_topic, now_iso, stable_signal_id


@dataclass
class WebPageSignalSource:
    """Fetch a webpage and emit one normalized Freeman signal."""

    url: str
    default_topic: str | None = None
    timeout_seconds: float = 30.0
    headers: dict[str, str] = field(default_factory=dict)
    max_text_chars: int = 4000
    source_type: str = "web"
    static_metadata: dict[str, object] = field(default_factory=dict)

    def fetch(self) -> list[Signal]:
        """Fetch one page and turn it into a single signal."""

        response = requests.get(self.url, headers=self.headers or None, timeout=self.timeout_seconds)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")
        title = coerce_text(soup.title.string if soup.title else "", fallback=hostname_topic(self.url))
        paragraphs = [paragraph.get_text(" ", strip=True) for paragraph in soup.find_all("p")]
        body_text = "\n".join(text for text in paragraphs if text).strip()
        if self.max_text_chars > 0:
            body_text = body_text[: self.max_text_chars]
        text = "\n\n".join(part for part in [title, body_text] if part).strip()
        if not text:
            text = coerce_text(soup.get_text(" ", strip=True))[: self.max_text_chars]
        topic = self.default_topic or hostname_topic(self.url)
        return [
            Signal(
                signal_id=stable_signal_id(f"{self.source_type}:{self.url}", {"title": title, "text": text}),
                source_type=self.source_type,
                text=text,
                topic=topic,
                entities=[],
                sentiment=0.0,
                timestamp=now_iso(),
                metadata={
                    **self.static_metadata,
                    "connector_type": "web",
                    "page_title": title,
                    "request_url": self.url,
                    "content_type": response.headers.get("Content-Type", ""),
                },
            )
        ]


__all__ = ["WebPageSignalSource"]
