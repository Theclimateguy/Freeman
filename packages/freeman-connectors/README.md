# freeman-connectors

`freeman-connectors` is a separate ingestion package for Freeman. It keeps source adapters out of the core agent while providing universal HTTP-first signal sources that emit native `freeman.agent.signalingestion.Signal` objects.

This package is part of the actual operational runtime path. The daemon loop in `python -m freeman.runtime.stream_runtime` reads `agent.sources` from config and instantiates connectors through `build_signal_source(...)`.

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

## Runtime config example

```yaml
agent:
  sources:
    - type: rss
      url: https://www.carbonbrief.org/feed/
    - type: webpage
      url: https://www.noaa.gov/news
      item_selector: article
      link_selector: a
```

These sources can then be consumed directly by:

```bash
python -m freeman.runtime.stream_runtime --config-path config.yaml --hours 8 --resume --model auto
```

Connector-specific release tests live in `tests/test_connectors.py`, while runtime integration remains covered by the main daemon tests in `tests/test_runtime.py`.

License: Apache License 2.0. See [LICENSE](/Users/theclimateguy/Documents/science/Freeman/packages/freeman-connectors/LICENSE).
