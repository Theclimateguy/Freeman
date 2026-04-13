"""Universal connectors package for Freeman."""

from freeman_connectors.factory import build_signal_source
from freeman_connectors.http import HTTPJSONSignalSource
from freeman_connectors.rss import ArxivSignalSource, RSSFeedSignalSource
from freeman_connectors.web import WebPageSignalSource

__version__ = "2.0.1"

__all__ = [
    "__version__",
    "ArxivSignalSource",
    "HTTPJSONSignalSource",
    "RSSFeedSignalSource",
    "WebPageSignalSource",
    "build_signal_source",
]
