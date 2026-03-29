# freeman-connectors

`freeman-connectors` is a separate ingestion package for Freeman. It keeps source adapters out of the core agent while providing universal HTTP-first signal sources that emit native `freeman.agent.signalingestion.Signal` objects.

## Install

From the monorepo:

```bash
pip install ./packages/freeman-connectors
```

From GitHub:

```bash
pip install "git+https://github.com/Theclimateguy/Freeman.git#subdirectory=packages/freeman-connectors"
```

## Included sources

- `HTTPJSONSignalSource` for arbitrary HTTP/JSON endpoints
- `RSSFeedSignalSource` for RSS and Atom feeds
- `WebPageSignalSource` for generic HTML pages
- `ArxivSignalSource` for arXiv search feeds

## Example

```python
from freeman.agent import SignalIngestionEngine
from freeman_connectors import HTTPJSONSignalSource

source = HTTPJSONSignalSource(
    url="https://example.com/api/events",
    item_path="items",
    field_map={
        "signal_id": "id",
        "text": "summary",
        "topic": "category",
        "timestamp": "published_at",
    },
    source_type="http_json",
)

engine = SignalIngestionEngine()
signals = source.fetch()
triggers = engine.ingest(source)
```

## Config-driven construction

```python
from freeman_connectors import build_signal_source

source = build_signal_source(
    {
        "type": "http_json",
        "url": "https://example.com/api/events",
        "item_path": "items",
        "field_map": {"text": "headline"},
    }
)
```
