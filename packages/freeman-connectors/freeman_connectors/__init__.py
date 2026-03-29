"""Universal connectors package for Freeman."""

from freeman_connectors.factory import build_signal_source
from freeman_connectors.http import HTTPJSONSignalSource
from freeman_connectors.rss import ArxivSignalSource, RSSFeedSignalSource
from freeman_connectors.web import WebPageSignalSource

__all__ = [
    "ArxivSignalSource",
    "HTTPJSONSignalSource",
    "RSSFeedSignalSource",
    "WebPageSignalSource",
    "build_signal_source",
]
