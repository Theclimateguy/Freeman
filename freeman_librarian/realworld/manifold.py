"""Manifold-backed real-world experiment for Freeman."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta, timezone
from email.utils import parsedate_to_datetime
import hashlib
import json
import math
import os
import socket
import time
from pathlib import Path
import re
from typing import Any, Iterable, Sequence
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode, urlparse
from urllib.request import Request, urlopen
import xml.etree.ElementTree as ET

import numpy as np

from freeman_librarian.agent.analysispipeline import AnalysisPipeline
from freeman_librarian.agent.forecastregistry import Forecast, ForecastRegistry
from freeman_librarian.agent.parameterestimator import ParameterEstimator
from freeman_librarian.agent.signalingestion import RSSSignalSource, SignalIngestionEngine, SignalMemory
from freeman_librarian.game.runner import SimConfig
from freeman_librarian.llm.deepseek import DeepSeekChatClient
from freeman_librarian.memory.knowledgegraph import KnowledgeGraph
from freeman_librarian.memory.sessionlog import SessionLog

DAY_MS = 86_400_000
DEFAULT_TIMEOUT = 30
_USER_AGENT = "FreemanRealWorldExperiment/0.1"
DEFAULT_MARKET_SEARCH_TERMS = (
    "climate",
    "carbon",
    "emissions",
    "warming",
    "temperature",
    "wildfire",
    "drought",
    "flood",
    "heat",
    "cop",
)
_TOKEN_RE = re.compile(r"[a-z0-9]+")
_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "be",
    "by",
    "for",
    "if",
    "in",
    "is",
    "of",
    "on",
    "or",
    "the",
    "to",
    "will",
    "with",
}
_CLIMATE_TERMS = {
    "climate",
    "warming",
    "temperature",
    "heat",
    "hot",
    "flood",
    "drought",
    "storm",
    "emissions",
    "carbon",
    "wildfire",
    "weather",
    "glacier",
    "ocean",
    "sea",
    "disaster",
}
_CLIMATE_MARKET_TERMS = _CLIMATE_TERMS | {
    "cop",
    "paris",
    "agreement",
    "net",
    "zero",
    "greenhouse",
    "methane",
}
_NEGATIVE_POLARITY_PHRASES = (
    "withdraw from",
    "deplete the remaining carbon budget",
    "deplete remaining carbon budget",
    "record heat",
    "record temperature",
)
_NEGATIVE_POLARITY_TOKENS = {
    "abnormal",
    "abnormally",
    "deplete",
    "disaster",
    "disasters",
    "exceed",
    "flood",
    "hotter",
    "increase",
    "record",
    "retreat",
    "storm",
    "warmer",
    "withdraw",
    "wildfire",
}
_STRONG_NEGATIVE_POLARITY_TOKENS = {
    "above",
    "deplete",
    "disaster",
    "exceed",
    "record",
    "withdraw",
}
_POSITIVE_POLARITY_PHRASES = (
    "binding agreement",
    "below",
    "limit warming",
    "net zero",
    "phase out",
    "still be",
)
_POSITIVE_POLARITY_TOKENS = {
    "achieve",
    "agreement",
    "below",
    "binding",
    "cut",
    "limit",
    "lower",
    "neutral",
    "ratify",
    "reduce",
    "remain",
    "stay",
}
_STRONG_POSITIVE_POLARITY_TOKENS = {
    "achieve",
    "below",
    "binding",
    "ratify",
}


def _now_utc() -> datetime:
    return datetime.now(timezone.utc).replace(microsecond=0)


def _clip(value: float, lower: float, upper: float) -> float:
    return float(min(max(value, lower), upper))


def _logit(probability: float, eps: float = 1.0e-6) -> float:
    p = _clip(float(probability), eps, 1.0 - eps)
    return float(math.log(p / (1.0 - p)))


def _ms_to_iso(timestamp_ms: int) -> str:
    return datetime.fromtimestamp(float(timestamp_ms) / 1000.0, tz=timezone.utc).replace(microsecond=0).isoformat()


def _rss_timestamp_to_iso(value: str) -> str:
    if not value:
        return _now_utc().isoformat()
    try:
        return parsedate_to_datetime(value).astimezone(timezone.utc).replace(microsecond=0).isoformat()
    except (TypeError, ValueError, IndexError):
        return _now_utc().isoformat()


def _gdelt_timestamp_to_iso(value: str) -> str:
    if not value:
        return _now_utc().isoformat()
    try:
        return datetime.strptime(value, "%Y%m%dT%H%M%SZ").replace(tzinfo=timezone.utc).isoformat()
    except ValueError:
        return _now_utc().isoformat()


def _to_gdelt_datetime(value: datetime) -> str:
    return value.astimezone(timezone.utc).strftime("%Y%m%d%H%M%S")


def _ordered_tokens(text: str) -> list[str]:
    tokens: list[str] = []
    seen: set[str] = set()
    for token in _TOKEN_RE.findall(text.lower()):
        if len(token) <= 2 or token in _STOPWORDS or token in seen:
            continue
        seen.add(token)
        tokens.append(token)
    return tokens


def _normalize_tokens(text: str) -> set[str]:
    return set(_ordered_tokens(text))


def _question_direction(question: str) -> int:
    positive = {
        "abnormal",
        "abnormally",
        "above",
        "disaster",
        "disasters",
        "exceed",
        "higher",
        "hotter",
        "increase",
        "more",
        "record",
        "warmer",
    }
    negative = {"below", "cooler", "decrease", "drop", "fewer", "lower", "less"}
    tokens = _normalize_tokens(question)
    pos_hits = len(tokens & positive)
    neg_hits = len(tokens & negative)
    if pos_hits > neg_hits:
        return 1
    if neg_hits > pos_hits:
        return -1
    return 0


def _infer_domain_polarity(question: str, description: str = "") -> str:
    """Infer whether the literal YES outcome is favorable or adverse."""

    text = f"{question} {description}".lower()
    negative_hits = sum(phrase in text for phrase in _NEGATIVE_POLARITY_PHRASES)
    positive_hits = sum(phrase in text for phrase in _POSITIVE_POLARITY_PHRASES)
    tokens = _normalize_tokens(text)
    negative_hits += len(tokens & _NEGATIVE_POLARITY_TOKENS)
    positive_hits += len(tokens & _POSITIVE_POLARITY_TOKENS)
    strong_negative = bool((tokens & _STRONG_NEGATIVE_POLARITY_TOKENS) or negative_hits > 0)
    strong_positive = bool((tokens & _STRONG_POSITIVE_POLARITY_TOKENS) or positive_hits > 0)
    if strong_negative and not strong_positive:
        return "negative"
    if strong_positive and not strong_negative:
        return "positive"
    if negative_hits > positive_hits:
        return "negative"
    if strong_negative and negative_hits >= positive_hits:
        return "negative"
    return "positive"


def _http_get_json(
    url: str,
    params: dict[str, Any] | None = None,
    timeout: int = DEFAULT_TIMEOUT,
    headers: dict[str, str] | None = None,
) -> Any:
    target = url
    if params:
        encoded = urlencode(
            {key: value for key, value in params.items() if value is not None},
            doseq=True,
        )
        target = f"{url}?{encoded}"
    request_headers = {"User-Agent": _USER_AGENT, "Accept": "application/json"}
    request_headers.update(headers or {})
    request = Request(target, headers=request_headers)
    with urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def _http_get_text(url: str, timeout: int = DEFAULT_TIMEOUT) -> str:
    request = Request(url, headers={"User-Agent": _USER_AGENT, "Accept": "*/*"})
    with urlopen(request, timeout=timeout) as response:
        return response.read().decode("utf-8")


def _market_target_probability(market: dict[str, Any]) -> float | None:
    resolution = market.get("resolution")
    if resolution == "YES":
        return 1.0
    if resolution == "NO":
        return 0.0
    if market.get("resolutionProbability") is not None:
        return float(market["resolutionProbability"])
    return None


@dataclass(frozen=True)
class MarketBet:
    """Compact Manifold bet record."""

    bet_id: str
    contract_id: str
    created_time: int
    prob_before: float | None
    prob_after: float | None
    amount: float
    outcome: str | None = None


@dataclass(frozen=True)
class RSSHeadline:
    """Normalized BBC RSS headline."""

    signal_id: str
    title: str
    description: str
    link: str
    published_at: str
    feed: str


@dataclass(frozen=True)
class GDELTArticle:
    """Normalized GDELT article hit."""

    article_id: str
    title: str
    url: str
    published_at: str
    domain: str
    language: str
    source_country: str


@dataclass(frozen=True)
class MarketFeatures:
    """Structured state for one market at a cutoff."""

    cutoff_probability: float
    probability_7d: float
    probability_30d: float
    momentum_7d: float
    momentum_30d: float
    flow_7d: float
    flow_30d: float
    turnover_7d: float
    turnover_30d: float
    bets_total: int
    bets_7d: int
    bets_30d: int
    liquidity: float
    age_days: float
    horizon_days: float
    cutoff_time_ms: int


@dataclass(frozen=True)
class ManifoldBacktestResult:
    """Evaluation row for one resolved market."""

    market_id: str
    question: str
    target_probability: float
    cutoff_probability: float
    freeman_probability: float
    market_brier: float
    freeman_brier: float
    resolution: str | None
    resolution_time: int
    cutoff_time: int
    features: MarketFeatures
    freeman_probability_with_news: float | None = None
    freeman_with_news_brier: float | None = None
    historical_news_edge: float | None = None
    historical_news_article_count: int = 0
    historical_news_titles: list[str] = field(default_factory=list)
    historical_news_query: str | None = None
    llm_rationale: str | None = None
    llm_parameter_vector: dict[str, Any] | None = None


@dataclass(frozen=True)
class LiveMarketSnapshot:
    """Live open-market analysis snapshot."""

    market_id: str
    question: str
    market_probability: float
    freeman_probability: float
    news_edge: float
    relevant_headlines: list[str]
    trigger_modes: list[str]


@dataclass
class ExperimentReport:
    """Saved experiment artifact."""

    created_at: str
    term: str
    market_terms: list[str]
    horizon_days: int
    market_limit: int
    bootstrap_samples: int
    resolved_evaluated: int
    skipped_markets: int
    market_brier_mean: float | None
    freeman_brier_mean: float | None
    freeman_minus_market_brier: float | None
    freeman_minus_market_brier_ci_low: float | None
    freeman_minus_market_brier_ci_high: float | None
    historical_news_evaluated: int
    freeman_with_news_brier_mean: float | None
    freeman_with_news_minus_market_brier: float | None
    freeman_with_news_minus_market_brier_ci_low: float | None
    freeman_with_news_minus_market_brier_ci_high: float | None
    live_market_count: int
    notes: list[str] = field(default_factory=list)


class ManifoldClient:
    """Thin client for the public Manifold API."""

    def __init__(self, base_url: str = "https://api.manifold.markets/v0", timeout: int = DEFAULT_TIMEOUT) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = int(timeout)

    def search_markets(
        self,
        *,
        term: str,
        limit: int = 20,
        market_filter: str | None = None,
        contract_type: str = "BINARY",
        sort: str = "liquidity",
        topic_slug: str | None = None,
    ) -> list[dict[str, Any]]:
        payload = _http_get_json(
            f"{self.base_url}/search-markets",
            params={
                "term": term,
                "limit": int(limit),
                "filter": market_filter,
                "contractType": contract_type,
                "sort": sort,
                "topicSlug": topic_slug,
            },
            timeout=self.timeout,
        )
        if not isinstance(payload, list):
            raise ValueError(f"Unexpected search response: {payload!r}")
        return payload

    def get_market(self, market_id: str) -> dict[str, Any]:
        payload = _http_get_json(f"{self.base_url}/market/{market_id}", timeout=self.timeout)
        if not isinstance(payload, dict):
            raise ValueError(f"Unexpected market response for {market_id}: {payload!r}")
        return payload

    def get_bets(self, contract_id: str, limit: int = 1000) -> list[MarketBet]:
        payload = _http_get_json(
            f"{self.base_url}/bets",
            params={"contractId": contract_id, "limit": int(limit)},
            timeout=self.timeout,
        )
        if not isinstance(payload, list):
            raise ValueError(f"Unexpected bets response for {contract_id}: {payload!r}")
        bets = [
            MarketBet(
                bet_id=str(item["id"]),
                contract_id=str(item["contractId"]),
                created_time=int(item["createdTime"]),
                prob_before=float(item["probBefore"]) if item.get("probBefore") is not None else None,
                prob_after=float(item["probAfter"]) if item.get("probAfter") is not None else None,
                amount=float(item.get("amount", 0.0)),
                outcome=item.get("outcome"),
            )
            for item in payload
            if not bool(item.get("isCancelled", False))
        ]
        return sorted(bets, key=lambda bet: (bet.created_time, bet.bet_id))


class BBCRSSClient:
    """Fetch a small BBC RSS snapshot for live climate signals."""

    DEFAULT_FEEDS = {
        "science_and_environment": "https://feeds.bbci.co.uk/news/science_and_environment/rss.xml",
        "world": "https://feeds.bbci.co.uk/news/world/rss.xml",
    }

    def __init__(self, feeds: dict[str, str] | None = None, timeout: int = DEFAULT_TIMEOUT) -> None:
        self.feeds = dict(feeds or self.DEFAULT_FEEDS)
        self.timeout = int(timeout)

    def fetch(self) -> list[RSSHeadline]:
        headlines: list[RSSHeadline] = []
        for feed_name, url in self.feeds.items():
            xml_text = _http_get_text(url, timeout=self.timeout)
            root = ET.fromstring(xml_text)
            for item in root.findall("./channel/item"):
                guid = item.findtext("guid") or item.findtext("link") or item.findtext("title") or ""
                title = (item.findtext("title") or "").strip()
                description = (item.findtext("description") or "").strip()
                link = (item.findtext("link") or "").strip()
                pub_date = (item.findtext("pubDate") or "").strip()
                if not title:
                    continue
                headlines.append(
                    RSSHeadline(
                        signal_id=f"bbc:{feed_name}:{guid}",
                        title=title,
                        description=description,
                        link=link,
                        published_at=_rss_timestamp_to_iso(pub_date),
                        feed=feed_name,
                    )
                )
        return headlines


class HistoricalNewsProvider:
    """Minimal interface for historical article backfills."""

    name = "historical-news"

    def search_articles(
        self,
        query: str,
        *,
        start_time: datetime,
        end_time: datetime,
        max_records: int = 25,
    ) -> list[GDELTArticle]:
        raise NotImplementedError


class GDELTDocClient:
    """Rate-limited GDELT DOC API client with local JSON caching."""

    name = "gdelt"

    def __init__(
        self,
        *,
        cache_dir: str | Path | None = None,
        timeout: int = 60,
        rate_limit_seconds: float = 8.0,
        max_retries: int = 5,
    ) -> None:
        self.cache_dir = (
            Path(cache_dir).resolve()
            if cache_dir is not None
            else Path(__file__).resolve().parents[2] / "data" / "gdelt_cache"
        )
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.timeout = int(timeout)
        self.rate_limit_seconds = float(rate_limit_seconds)
        self.max_retries = int(max_retries)
        self._last_request_monotonic = 0.0

    def search_articles(
        self,
        query: str,
        *,
        start_time: datetime,
        end_time: datetime,
        max_records: int = 25,
    ) -> list[GDELTArticle]:
        """Fetch and cache article hits for one query window."""

        cache_key = hashlib.sha1(
            f"{query}|{_to_gdelt_datetime(start_time)}|{_to_gdelt_datetime(end_time)}|{max_records}".encode("utf-8")
        ).hexdigest()
        cache_path = self.cache_dir / f"{cache_key}.json"
        if cache_path.exists():
            payload = json.loads(cache_path.read_text(encoding="utf-8"))
            return self._parse_articles(payload)

        params = {
            "query": query,
            "mode": "ArtList",
            "maxrecords": int(max_records),
            "format": "json",
            "startdatetime": _to_gdelt_datetime(start_time),
            "enddatetime": _to_gdelt_datetime(end_time),
        }
        attempts = max(self.max_retries, 1)
        last_error: Exception | None = None
        for attempt in range(1, attempts + 1):
            self._sleep_if_needed()
            try:
                payload = _http_get_json("https://api.gdeltproject.org/api/v2/doc/doc", params=params, timeout=self.timeout)
                self._last_request_monotonic = time.monotonic()
                cache_path.write_text(json.dumps(payload, ensure_ascii=False, sort_keys=True), encoding="utf-8")
                return self._parse_articles(payload)
            except HTTPError as exc:
                last_error = exc
                self._last_request_monotonic = time.monotonic()
                if exc.code == 429 and attempt < attempts:
                    time.sleep(self.rate_limit_seconds * attempt)
                    continue
                raise
            except (URLError, TimeoutError, socket.timeout) as exc:
                last_error = exc
                self._last_request_monotonic = time.monotonic()
                if attempt < attempts:
                    time.sleep(min(self.rate_limit_seconds * attempt, 20.0))
                    continue
                raise
        if last_error is not None:
            raise last_error
        return []

    def _sleep_if_needed(self) -> None:
        elapsed = time.monotonic() - self._last_request_monotonic
        if self._last_request_monotonic > 0.0 and elapsed < self.rate_limit_seconds:
            time.sleep(self.rate_limit_seconds - elapsed)

    def _parse_articles(self, payload: Any) -> list[GDELTArticle]:
        if not isinstance(payload, dict):
            return []
        articles = payload.get("articles", [])
        if not isinstance(articles, list):
            return []
        parsed: list[GDELTArticle] = []
        for idx, item in enumerate(articles):
            if not isinstance(item, dict):
                continue
            parsed.append(
                GDELTArticle(
                    article_id=str(item.get("url") or item.get("title") or idx),
                    title=str(item.get("title", "")).strip(),
                    url=str(item.get("url", "")).strip(),
                    published_at=_gdelt_timestamp_to_iso(str(item.get("seendate", ""))),
                    domain=str(item.get("domain", "")).strip(),
                    language=str(item.get("language", "")).strip(),
                    source_country=str(item.get("sourcecountry", "")).strip(),
                )
            )
        return parsed


class NewsAPIArchiveClient(HistoricalNewsProvider):
    """Historical article provider backed by NewsAPI's Everything endpoint."""

    name = "newsapi"

    def __init__(
        self,
        api_key: str,
        *,
        cache_dir: str | Path | None = None,
        timeout: int = 30,
        max_retries: int = 3,
    ) -> None:
        self.api_key = api_key.strip()
        if not self.api_key:
            raise ValueError("NewsAPIArchiveClient requires a non-empty api_key.")
        self.cache_dir = (
            Path(cache_dir).resolve()
            if cache_dir is not None
            else Path(__file__).resolve().parents[2] / "data" / "newsapi_cache"
        )
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.timeout = int(timeout)
        self.max_retries = int(max_retries)

    def search_articles(
        self,
        query: str,
        *,
        start_time: datetime,
        end_time: datetime,
        max_records: int = 25,
    ) -> list[GDELTArticle]:
        cache_key = hashlib.sha1(
            f"newsapi|{query}|{start_time.isoformat()}|{end_time.isoformat()}|{max_records}".encode("utf-8")
        ).hexdigest()
        cache_path = self.cache_dir / f"{cache_key}.json"
        if cache_path.exists():
            payload = json.loads(cache_path.read_text(encoding="utf-8"))
            return self._parse_articles(payload)

        params = {
            "q": query,
            "from": start_time.astimezone(timezone.utc).isoformat().replace("+00:00", "Z"),
            "to": end_time.astimezone(timezone.utc).isoformat().replace("+00:00", "Z"),
            "language": "en",
            "sortBy": "relevancy",
            "pageSize": int(max_records),
        }
        attempts = max(self.max_retries, 1)
        last_error: Exception | None = None
        for _ in range(attempts):
            try:
                payload = _http_get_json(
                    "https://newsapi.org/v2/everything",
                    params=params,
                    timeout=self.timeout,
                    headers={"X-Api-Key": self.api_key},
                )
                cache_path.write_text(json.dumps(payload, ensure_ascii=False, sort_keys=True), encoding="utf-8")
                return self._parse_articles(payload)
            except Exception as exc:  # pragma: no cover - network only.
                last_error = exc
                time.sleep(1.0)
        if last_error is not None:
            raise last_error
        return []

    def _parse_articles(self, payload: Any) -> list[GDELTArticle]:
        if not isinstance(payload, dict):
            return []
        articles = payload.get("articles", [])
        if not isinstance(articles, list):
            return []
        parsed: list[GDELTArticle] = []
        for idx, item in enumerate(articles):
            if not isinstance(item, dict):
                continue
            url = str(item.get("url", "")).strip()
            parsed.append(
                GDELTArticle(
                    article_id=url or str(item.get("title") or idx),
                    title=str(item.get("title", "")).strip(),
                    url=url,
                    published_at=str(item.get("publishedAt", "")).strip(),
                    domain=urlparse(url).netloc,
                    language="english",
                    source_country="",
                )
            )
        return parsed


class GNewsArchiveClient(HistoricalNewsProvider):
    """Historical article provider backed by the GNews search endpoint."""

    name = "gnews"

    def __init__(
        self,
        api_key: str,
        *,
        cache_dir: str | Path | None = None,
        timeout: int = 30,
        max_retries: int = 3,
    ) -> None:
        self.api_key = api_key.strip()
        if not self.api_key:
            raise ValueError("GNewsArchiveClient requires a non-empty api_key.")
        self.cache_dir = (
            Path(cache_dir).resolve()
            if cache_dir is not None
            else Path(__file__).resolve().parents[2] / "data" / "gnews_cache"
        )
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.timeout = int(timeout)
        self.max_retries = int(max_retries)

    def search_articles(
        self,
        query: str,
        *,
        start_time: datetime,
        end_time: datetime,
        max_records: int = 25,
    ) -> list[GDELTArticle]:
        cache_key = hashlib.sha1(
            f"gnews|{query}|{start_time.isoformat()}|{end_time.isoformat()}|{max_records}".encode("utf-8")
        ).hexdigest()
        cache_path = self.cache_dir / f"{cache_key}.json"
        if cache_path.exists():
            payload = json.loads(cache_path.read_text(encoding="utf-8"))
            return self._parse_articles(payload)

        params = {
            "q": query,
            "lang": "en",
            "max": int(min(max_records, 100)),
            "from": start_time.astimezone(timezone.utc).isoformat().replace("+00:00", "Z"),
            "to": end_time.astimezone(timezone.utc).isoformat().replace("+00:00", "Z"),
            "sortby": "relevance",
            "apikey": self.api_key,
        }
        attempts = max(self.max_retries, 1)
        last_error: Exception | None = None
        for _ in range(attempts):
            try:
                payload = _http_get_json(
                    "https://gnews.io/api/v4/search",
                    params=params,
                    timeout=self.timeout,
                )
                cache_path.write_text(json.dumps(payload, ensure_ascii=False, sort_keys=True), encoding="utf-8")
                return self._parse_articles(payload)
            except Exception as exc:  # pragma: no cover - network only.
                last_error = exc
                time.sleep(1.0)
        if last_error is not None:
            raise last_error
        return []

    def _parse_articles(self, payload: Any) -> list[GDELTArticle]:
        if not isinstance(payload, dict):
            return []
        articles = payload.get("articles", [])
        if not isinstance(articles, list):
            return []
        parsed: list[GDELTArticle] = []
        for idx, item in enumerate(articles):
            if not isinstance(item, dict):
                continue
            url = str(item.get("url", "")).strip()
            source = item.get("source", {})
            source_url = str(source.get("url", "")).strip() if isinstance(source, dict) else ""
            parsed.append(
                GDELTArticle(
                    article_id=url or str(item.get("title") or idx),
                    title=str(item.get("title", "")).strip(),
                    url=url,
                    published_at=str(item.get("publishedAt", "")).strip(),
                    domain=urlparse(source_url or url).netloc,
                    language="english",
                    source_country="",
                )
            )
        return parsed


class ChainedHistoricalNewsProvider(HistoricalNewsProvider):
    """Try multiple historical news providers in order."""

    def __init__(self, providers: Sequence[HistoricalNewsProvider]) -> None:
        self.providers = list(providers)
        self.name = "+".join(provider.name for provider in self.providers)

    def search_articles(
        self,
        query: str,
        *,
        start_time: datetime,
        end_time: datetime,
        max_records: int = 25,
    ) -> list[GDELTArticle]:
        last_error: Exception | None = None
        for provider in self.providers:
            try:
                articles = provider.search_articles(
                    query,
                    start_time=start_time,
                    end_time=end_time,
                    max_records=max_records,
                )
                if articles:
                    return articles
            except Exception as exc:  # pragma: no cover - network only.
                last_error = exc
                continue
        if last_error is not None:
            raise last_error
        return []


def reconstruct_probability_path(market: dict[str, Any], bets: Sequence[MarketBet]) -> list[tuple[int, float]]:
    """Rebuild a market probability path from bets."""

    sorted_bets = sorted(bets, key=lambda bet: (bet.created_time, bet.bet_id))
    if sorted_bets:
        initial_probability = next(
            (bet.prob_before for bet in sorted_bets if bet.prob_before is not None),
            0.5,
        )
    else:
        initial_probability = 0.5
    path = [(int(market["createdTime"]), float(initial_probability))]
    for bet in sorted_bets:
        if bet.prob_after is None:
            continue
        path.append((bet.created_time, float(bet.prob_after)))
    deduped: list[tuple[int, float]] = []
    for timestamp_ms, probability in path:
        if deduped and deduped[-1][0] == timestamp_ms:
            deduped[-1] = (timestamp_ms, probability)
        else:
            deduped.append((timestamp_ms, probability))
    return deduped


def _probability_at_or_before(path: Sequence[tuple[int, float]], cutoff_time_ms: int) -> float:
    prior_probability = float(path[0][1])
    for timestamp_ms, probability in path:
        if timestamp_ms > cutoff_time_ms:
            break
        prior_probability = float(probability)
    return prior_probability


def _window_bets(bets: Sequence[MarketBet], cutoff_time_ms: int, window_days: int) -> list[MarketBet]:
    window_start = cutoff_time_ms - int(window_days * DAY_MS)
    return [
        bet
        for bet in bets
        if window_start < bet.created_time <= cutoff_time_ms
    ]


def compute_market_features(
    market: dict[str, Any],
    bets: Sequence[MarketBet],
    *,
    cutoff_time_ms: int,
    horizon_days: int,
) -> MarketFeatures:
    """Compute a compact market state vector at one cutoff."""

    path = reconstruct_probability_path(market, bets)
    cutoff_probability = _probability_at_or_before(path, cutoff_time_ms)
    probability_7d = _probability_at_or_before(path, cutoff_time_ms - 7 * DAY_MS)
    probability_30d = _probability_at_or_before(path, cutoff_time_ms - 30 * DAY_MS)
    bets_7d = _window_bets(bets, cutoff_time_ms, 7)
    bets_30d = _window_bets(bets, cutoff_time_ms, 30)
    flow_7d = sum(
        float((bet.prob_after or 0.0) - (bet.prob_before or 0.0))
        for bet in bets_7d
        if bet.prob_after is not None and bet.prob_before is not None
    )
    flow_30d = sum(
        float((bet.prob_after or 0.0) - (bet.prob_before or 0.0))
        for bet in bets_30d
        if bet.prob_after is not None and bet.prob_before is not None
    )
    turnover_7d = sum(abs(bet.amount) for bet in bets_7d)
    turnover_30d = sum(abs(bet.amount) for bet in bets_30d)
    age_days = max((cutoff_time_ms - int(market["createdTime"])) / DAY_MS, 0.0)
    if age_days < 7.0:
        probability_7d = cutoff_probability
    if age_days < 30.0:
        probability_30d = cutoff_probability
    return MarketFeatures(
        cutoff_probability=float(cutoff_probability),
        probability_7d=float(probability_7d),
        probability_30d=float(probability_30d),
        momentum_7d=float(cutoff_probability - probability_7d),
        momentum_30d=float(cutoff_probability - probability_30d),
        flow_7d=float(flow_7d),
        flow_30d=float(flow_30d),
        turnover_7d=float(turnover_7d),
        turnover_30d=float(turnover_30d),
        bets_total=int(len(bets)),
        bets_7d=int(len(bets_7d)),
        bets_30d=int(len(bets_30d)),
        liquidity=float(market.get("totalLiquidity", 0.0)),
        age_days=float(age_days),
        horizon_days=float(horizon_days),
        cutoff_time_ms=int(cutoff_time_ms),
    )


def build_binary_market_schema(
    market: dict[str, Any],
    features: MarketFeatures,
    *,
    news_edge: float = 0.0,
    domain_polarity: str | None = None,
) -> dict[str, Any]:
    """Translate one Manifold market snapshot into a compact Freeman schema."""

    market_logit = 0.5 * _logit(features.cutoff_probability)
    momentum_edge = _clip(4.0 * features.momentum_7d, -1.0, 1.0)
    flow_edge = _clip(4.0 * features.flow_7d, -1.0, 1.0)
    attention_heat = _clip(math.log1p(features.bets_30d) / 4.0, 0.0, 1.0)
    uncertainty_edge = _clip(1.0 - abs(2.0 * features.cutoff_probability - 1.0), 0.0, 1.0)
    directed_news_edge = _clip(float(news_edge), -1.0, 1.0)
    resolved_polarity = str(
        domain_polarity
        or _infer_domain_polarity(
            str(market.get("question", "")),
            str(market.get("textDescription", "") or ""),
        )
    ).lower()
    if resolved_polarity not in {"positive", "negative"}:
        resolved_polarity = "positive"

    def _resource(resource_id: str, value: float, *, lower: float = -4.0, upper: float = 4.0) -> dict[str, Any]:
        return {
            "id": resource_id,
            "name": resource_id.replace("_", " ").title(),
            "value": float(value),
            "unit": "signal",
            "min_value": float(lower),
            "max_value": float(upper),
            "evolution_type": "linear",
            "evolution_params": {
                "a": 1.0,
                "b": 0.0,
                "c": 0.0,
                "coupling_weights": {},
            },
        }

    return {
        "domain_id": f"manifold_{market['id']}",
        "name": market["question"],
        "description": market.get("textDescription", "") or market["question"],
        "domain_polarity": resolved_polarity,
        "modifier_mode": "probability_monotonic",
        "actors": [
            {
                "id": "market",
                "name": "Prediction Market",
                "state": {"influence": 0.8},
                "metadata": {"liquidity": features.liquidity},
            },
            {
                "id": "news",
                "name": "News Flow",
                "state": {"influence": 0.6},
                "metadata": {"news_edge": directed_news_edge},
            },
        ],
        "resources": [
            _resource("market_logit", market_logit),
            _resource("momentum_edge", momentum_edge),
            _resource("flow_edge", flow_edge),
            _resource("news_edge", directed_news_edge),
            _resource("attention_heat", attention_heat, lower=0.0, upper=4.0),
            _resource("uncertainty_edge", uncertainty_edge, lower=0.0, upper=4.0),
        ],
        "relations": [
            {
                "source_id": "market",
                "target_id": "news",
                "relation_type": "information_flow",
                "weights": {"salience": attention_heat},
            }
        ],
        "outcomes": [
            {
                "id": "yes",
                "label": "YES",
                "scoring_weights": {
                    "market_logit": 1.0,
                    "momentum_edge": 0.35,
                    "flow_edge": 0.25,
                    "news_edge": 0.35,
                },
                "description": "The market resolves YES.",
            },
            {
                "id": "no",
                "label": "NO",
                "scoring_weights": {
                    "market_logit": -1.0,
                    "momentum_edge": -0.35,
                    "flow_edge": -0.25,
                    "news_edge": -0.35,
                },
                "description": "The market resolves NO.",
            },
        ],
        "causal_dag": [
            {"source": "market_logit", "target": "momentum_edge", "expected_sign": "+", "strength": "weak"},
            {"source": "news_edge", "target": "momentum_edge", "expected_sign": "+", "strength": "weak"},
            {"source": "uncertainty_edge", "target": "market_logit", "expected_sign": "-", "strength": "weak"},
        ],
        "metadata": {
            "market_id": market["id"],
            "question": market["question"],
            "created_time": market["createdTime"],
            "close_time": market.get("closeTime"),
            "resolution_time": market.get("resolutionTime"),
            "cutoff_probability": features.cutoff_probability,
            "horizon_days": features.horizon_days,
            "domain_polarity": resolved_polarity,
            "modifier_mode": "probability_monotonic",
        },
    }


def freeman_probability_from_schema(schema: dict[str, Any]) -> float:
    """Run Freeman and return the resulting YES probability."""

    pipeline = AnalysisPipeline(
        knowledge_graph=KnowledgeGraph(auto_load=False, auto_save=False),
        sim_config=SimConfig(
            max_steps=1,
            level2_check_every=1,
            stop_on_hard_level2=False,
            convergence_check_steps=25,
            convergence_epsilon=1.0e-5,
            seed=11,
        )
    )
    session_log = SessionLog(session_id=f"manifold:{schema['domain_id']}")
    result = pipeline.run(schema, session_log=session_log)
    return float(result.simulation["final_outcome_probs"]["yes"])


def freeman_probability_with_llm_signal(
    schema: dict[str, Any],
    *,
    signal_text: str,
    llm_client: DeepSeekChatClient,
) -> tuple[float, dict[str, Any]]:
    """Run one LLM-backed Freeman update and return YES probability plus the parameter vector."""

    pipeline = AnalysisPipeline(
        knowledge_graph=KnowledgeGraph(auto_load=False, auto_save=False),
        sim_config=SimConfig(
            max_steps=1,
            level2_check_every=1,
            stop_on_hard_level2=False,
            convergence_check_steps=25,
            convergence_epsilon=1.0e-5,
            seed=11,
        )
    )
    previous_world = pipeline.compiler.compile(schema)
    estimator = ParameterEstimator(llm_client)
    parameter_vector = estimator.estimate(previous_world, signal_text)
    result = pipeline.update(
        previous_world,
        parameter_vector,
        signal_text=signal_text,
        session_log=SessionLog(session_id=f"manifold-llm:{schema['domain_id']}"),
    )
    return float(result.simulation["final_outcome_probs"]["yes"]), parameter_vector.snapshot()


def _text_overlap(question: str, text: str) -> float:
    question_tokens = _normalize_tokens(question)
    text_tokens = _normalize_tokens(text)
    if not question_tokens or not text_tokens:
        return 0.0
    overlap = len(question_tokens & text_tokens)
    climate_overlap = len(text_tokens & _CLIMATE_TERMS)
    if overlap == 0 and climate_overlap == 0:
        return 0.0
    score = overlap / math.sqrt(max(len(question_tokens), 1))
    if climate_overlap > 0:
        score += 0.25 * climate_overlap
    return float(score)


def _headline_overlap(question: str, headline: RSSHeadline) -> float:
    return _text_overlap(question, f"{headline.title} {headline.description}")


def _gdelt_query_from_market(market: dict[str, Any]) -> str:
    question = str(market.get("question", ""))
    description = str(market.get("textDescription", "") or "")
    tokens = _ordered_tokens(f"{question} {description}")[:8]
    if not any(token in _CLIMATE_MARKET_TERMS for token in tokens):
        tokens.append("climate")
    return " ".join(tokens)


def _historical_signal_text(
    market: dict[str, Any],
    *,
    features: MarketFeatures,
    articles: Sequence[GDELTArticle],
    news_edge: float,
) -> str:
    lines = [
        f"MANIFOLD QUESTION: {market['question']}",
        f"MARKET DESCRIPTION: {market.get('textDescription', '') or market['question']}",
        f"CUTOFF MARKET PROBABILITY: {features.cutoff_probability:.4f}",
        f"7D MOMENTUM: {features.momentum_7d:+.4f}",
        f"30D MOMENTUM: {features.momentum_30d:+.4f}",
        f"INDEPENDENT NEWS EDGE ESTIMATE: {news_edge:+.4f}",
        "HISTORICAL NEWS WINDOW ARTICLES:",
    ]
    for article in articles[:5]:
        lines.append(f"- [{article.published_at}] {article.title}")
    return "\n".join(lines)


def _load_deepseek_api_key(path: str | Path | None = None) -> str:
    env_key = ""
    try:
        env_key = os.getenv("DEEPSEEK_API_KEY", "").strip()
    except Exception:  # pragma: no cover - environment lookup should never fail.
        env_key = ""
    if env_key:
        return env_key
    key_path = Path(path or Path(__file__).resolve().parents[2] / "DS.txt").resolve()
    key = key_path.read_text(encoding="utf-8").strip()
    if not key:
        raise RuntimeError(f"DeepSeek API key file is empty: {key_path}")
    return key


def _load_optional_api_key(
    *,
    env_names: Sequence[str],
    file_candidates: Sequence[str | Path] = (),
) -> str | None:
    for env_name in env_names:
        value = os.getenv(env_name, "").strip()
        if value:
            return value
    for candidate in file_candidates:
        path = Path(candidate).resolve()
        if path.exists():
            value = path.read_text(encoding="utf-8").strip()
            if value:
                return value
    return None


def build_historical_news_provider(
    *,
    provider: str = "auto",
    gnews_api_key_path: str | Path | None = None,
    newsapi_api_key_path: str | Path | None = None,
    gdelt_rate_limit_seconds: float = 8.0,
) -> HistoricalNewsProvider:
    selected = provider.strip().lower()
    repo_root = Path(__file__).resolve().parents[2]
    gnews_key = _load_optional_api_key(
        env_names=("GNEWS_API_KEY",),
        file_candidates=(
            gnews_api_key_path or repo_root / "GNEWS.txt",
        ),
    )
    newsapi_key = _load_optional_api_key(
        env_names=("NEWSAPI_API_KEY", "NEWS_API_KEY"),
        file_candidates=(
            newsapi_api_key_path or repo_root / "NEWSAPI.txt",
            repo_root / "NEWS_API.txt",
        ),
    )
    if selected == "gnews":
        if not gnews_key:
            raise RuntimeError("GNews provider requested but no GNEWS key was found in env or GNEWS.txt.")
        return GNewsArchiveClient(gnews_key)
    if selected == "newsapi":
        if not newsapi_key:
            raise RuntimeError("NewsAPI provider requested but no NEWSAPI key was found in env or NEWSAPI.txt.")
        return NewsAPIArchiveClient(newsapi_key)
    if selected == "gdelt":
        return GDELTDocClient(rate_limit_seconds=gdelt_rate_limit_seconds)
    providers: list[HistoricalNewsProvider] = []
    if gnews_key:
        providers.append(GNewsArchiveClient(gnews_key))
    if newsapi_key:
        providers.append(NewsAPIArchiveClient(newsapi_key))
    if providers:
        providers.append(GDELTDocClient(rate_limit_seconds=gdelt_rate_limit_seconds))
        return ChainedHistoricalNewsProvider(providers)
    return GDELTDocClient(rate_limit_seconds=gdelt_rate_limit_seconds)


def _market_relevance_score(market: dict[str, Any]) -> float:
    question = str(market.get("question", ""))
    description = str(market.get("textDescription", "") or "")
    score = len(_normalize_tokens(f"{question} {description}") & _CLIMATE_MARKET_TERMS)
    group_slugs = {str(slug).lower() for slug in market.get("groupSlugs") or []}
    if "climate" in group_slugs:
        score += 3.0
    if "environment" in group_slugs:
        score += 1.0
    return float(score)


def _collect_resolved_markets(
    client: ManifoldClient,
    *,
    market_limit: int,
    horizon_days: int,
    search_terms: Sequence[str],
    per_term_limit: int = 100,
) -> list[dict[str, Any]]:
    by_id: dict[str, dict[str, Any]] = {}
    for term in search_terms:
        payload = client.search_markets(
            term=term,
            limit=per_term_limit,
            market_filter="resolved",
            contract_type="BINARY",
            sort="liquidity",
            topic_slug=None,
        )
        for market in payload:
            if _market_relevance_score(market) <= 0.0:
                continue
            market_id = str(market["id"])
            previous = by_id.get(market_id)
            if previous is None or float(market.get("totalLiquidity", 0.0)) > float(previous.get("totalLiquidity", 0.0)):
                by_id[market_id] = market
    selected = _select_resolved_markets(list(by_id.values()), horizon_days=horizon_days)
    selected.sort(
        key=lambda market: (
            _market_relevance_score(market),
            float(market.get("totalLiquidity", 0.0)),
            int(market.get("resolutionTime", 0) or 0),
        ),
        reverse=True,
    )
    return selected[:market_limit]


def _bootstrap_mean_ci(values: Sequence[float], *, bootstrap_samples: int = 2000, seed: int = 42) -> tuple[float | None, float | None]:
    if not values:
        return None, None
    samples = max(int(bootstrap_samples), 100)
    array = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(array), size=(samples, len(array)))
    means = array[draws].mean(axis=1)
    low, high = np.quantile(means, [0.025, 0.975])
    return float(low), float(high)


def _prefetch_bets(
    client: ManifoldClient,
    markets: Sequence[dict[str, Any]],
    *,
    max_workers: int = 8,
) -> tuple[dict[str, list[MarketBet]], list[str]]:
    notes: list[str] = []
    fetched: dict[str, list[MarketBet]] = {}
    worker_count = max(1, min(int(max_workers), len(markets) or 1))
    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        future_map = {
            executor.submit(client.get_bets, str(market["id"])): str(market["id"])
            for market in markets
        }
        for future in as_completed(future_map):
            market_id = future_map[future]
            try:
                fetched[market_id] = future.result()
            except Exception as exc:  # pragma: no cover - network failures only.
                fetched[market_id] = []
                notes.append(f"Failed to fetch bets for {market_id}: {exc}")
    return fetched, notes


def _bbc_news_edge(question: str, headlines: Sequence[RSSHeadline]) -> tuple[float, list[str], list[str]]:
    source = RSSSignalSource(
        [
            {
                "signal_id": headline.signal_id,
                "text": f"{headline.title}. {headline.description}".strip(),
                "topic": "bbc_rss",
                "timestamp": headline.published_at or _now_utc().isoformat(),
                "entities": sorted((_normalize_tokens(headline.title) | (_normalize_tokens(headline.description) & _CLIMATE_TERMS)))[:8],
                "metadata": {"feed": headline.feed, "link": headline.link},
            }
            for headline in headlines
        ]
    )
    triggers = SignalIngestionEngine().ingest(source, signal_memory=SignalMemory())
    trigger_map = {trigger.signal_id: trigger for trigger in triggers}
    ranked: list[tuple[float, RSSHeadline]] = []
    for headline in headlines:
        overlap = _headline_overlap(question, headline)
        if overlap <= 0.0:
            continue
        trigger = trigger_map.get(headline.signal_id)
        severity = trigger.classification.severity if trigger is not None else 0.0
        ranked.append((overlap * (1.0 + severity), headline))
    ranked.sort(key=lambda item: item[0], reverse=True)
    top_ranked = ranked[:3]
    direction = _question_direction(question)
    if direction == 0 or not top_ranked:
        return 0.0, [headline.title for _, headline in top_ranked], [trigger_map[headline.signal_id].mode for _, headline in top_ranked if headline.signal_id in trigger_map]
    average_score = sum(score for score, _ in top_ranked) / len(top_ranked)
    news_edge = direction * _clip(0.25 * average_score, -1.0, 1.0)
    trigger_modes = [
        trigger_map[headline.signal_id].mode
        for _, headline in top_ranked
        if headline.signal_id in trigger_map
    ]
    return float(news_edge), [headline.title for _, headline in top_ranked], trigger_modes


def _gdelt_news_edge(question: str, articles: Sequence[GDELTArticle]) -> tuple[float, list[str], list[str], list[GDELTArticle]]:
    english_articles = [article for article in articles if article.language.lower() in {"english", ""}]
    if not english_articles:
        return 0.0, [], [], []
    source = RSSSignalSource(
        [
            {
                "signal_id": f"gdelt:{hashlib.sha1(article.article_id.encode('utf-8')).hexdigest()[:12]}",
                "text": article.title,
                "topic": "gdelt_doc",
                "timestamp": article.published_at,
                "entities": sorted((_normalize_tokens(article.title) & _CLIMATE_TERMS))[:8],
                "metadata": {"domain": article.domain, "url": article.url, "source_country": article.source_country},
                "title": article.title,
            }
            for article in english_articles
        ]
    )
    triggers = SignalIngestionEngine().ingest(source, signal_memory=SignalMemory())
    trigger_map = {trigger.signal_id: trigger for trigger in triggers}
    ranked: list[tuple[float, GDELTArticle, str]] = []
    for article in english_articles:
        signal_id = f"gdelt:{hashlib.sha1(article.article_id.encode('utf-8')).hexdigest()[:12]}"
        overlap = _text_overlap(question, article.title)
        if overlap <= 0.0:
            continue
        trigger = trigger_map.get(signal_id)
        severity = trigger.classification.severity if trigger is not None else 0.0
        ranked.append((overlap * (1.0 + severity), article, trigger.mode if trigger is not None else "WATCH"))
    ranked.sort(key=lambda item: item[0], reverse=True)
    top_ranked = ranked[:3]
    direction = _question_direction(question)
    if direction == 0 or not top_ranked:
        return 0.0, [article.title for _, article, _ in top_ranked], [mode for _, _, mode in top_ranked], [article for _, article, _ in top_ranked]
    average_score = sum(score for score, _, _ in top_ranked) / len(top_ranked)
    news_edge = direction * _clip(0.20 * average_score, -1.0, 1.0)
    return (
        float(news_edge),
        [article.title for _, article, _ in top_ranked],
        [mode for _, _, mode in top_ranked],
        [article for _, article, _ in top_ranked],
    )


def _brier_score(predicted_probability: float, target_probability: float) -> float:
    return float((float(predicted_probability) - float(target_probability)) ** 2)


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True), encoding="utf-8")


def _write_csv(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    rows_list = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows_list:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows_list[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        import csv

        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows_list)


def _flatten_backtest_row(result: ManifoldBacktestResult) -> dict[str, Any]:
    row = {
        "market_id": result.market_id,
        "question": result.question,
        "target_probability": result.target_probability,
        "cutoff_probability": result.cutoff_probability,
        "freeman_probability": result.freeman_probability,
        "market_brier": result.market_brier,
        "freeman_brier": result.freeman_brier,
        "freeman_probability_with_news": result.freeman_probability_with_news,
        "freeman_with_news_brier": result.freeman_with_news_brier,
        "historical_news_edge": result.historical_news_edge,
        "historical_news_article_count": result.historical_news_article_count,
        "historical_news_query": result.historical_news_query,
        "historical_news_titles": " | ".join(result.historical_news_titles),
        "llm_rationale": result.llm_rationale,
        "llm_parameter_vector": (
            json.dumps(result.llm_parameter_vector, ensure_ascii=False, sort_keys=True)
            if result.llm_parameter_vector is not None
            else None
        ),
        "resolution": result.resolution,
        "resolution_time": _ms_to_iso(result.resolution_time),
        "cutoff_time": _ms_to_iso(result.cutoff_time),
    }
    row.update({f"feature_{key}": value for key, value in asdict(result.features).items()})
    return row


def _select_resolved_markets(
    markets: Sequence[dict[str, Any]],
    *,
    horizon_days: int,
) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    for market in markets:
        if market.get("outcomeType") != "BINARY":
            continue
        if not market.get("isResolved"):
            continue
        target_probability = _market_target_probability(market)
        resolution_time = market.get("resolutionTime")
        created_time = market.get("createdTime")
        if target_probability is None or resolution_time is None or created_time is None:
            continue
        if int(resolution_time) - int(created_time) <= horizon_days * DAY_MS:
            continue
        selected.append(market)
    return selected


def fetch_and_run_experiment(
    *,
    term: str = "climate",
    market_limit: int = 20,
    horizon_days: int = 14,
    open_market_limit: int = 5,
    search_terms: Sequence[str] | None = None,
    historical_news_limit: int = 0,
    historical_news_window_days: int = 7,
    gdelt_rate_limit_seconds: float = 8.0,
    historical_news_provider: str = "auto",
    deepseek_api_key_path: str | Path | None = None,
    gnews_api_key_path: str | Path | None = None,
    newsapi_api_key_path: str | Path | None = None,
) -> tuple[list[ManifoldBacktestResult], list[LiveMarketSnapshot], list[str]]:
    """Fetch Manifold/BBC data and execute one experiment run."""

    client = ManifoldClient()
    notes: list[str] = []
    active_terms = tuple(search_terms or DEFAULT_MARKET_SEARCH_TERMS)
    resolved_markets = _collect_resolved_markets(
        client,
        market_limit=market_limit,
        horizon_days=horizon_days,
        search_terms=active_terms,
    )
    prefetched_bets, prefetch_notes = _prefetch_bets(client, resolved_markets)
    notes.extend(prefetch_notes)
    backtest_results: list[ManifoldBacktestResult] = []
    registry = ForecastRegistry(auto_load=False, auto_save=False)
    archive_provider = (
        build_historical_news_provider(
            provider=historical_news_provider,
            gnews_api_key_path=gnews_api_key_path,
            newsapi_api_key_path=newsapi_api_key_path,
            gdelt_rate_limit_seconds=gdelt_rate_limit_seconds,
        )
        if historical_news_limit > 0
        else None
    )
    llm_client = None
    if historical_news_limit > 0:
        llm_client = DeepSeekChatClient(api_key=_load_deepseek_api_key(deepseek_api_key_path))
        notes.append(f"DeepSeek model enabled via DS.txt/env using model={llm_client.model}.")
        if archive_provider is not None:
            notes.append(f"Historical news provider enabled: {archive_provider.name}.")

    for idx, market in enumerate(resolved_markets):
        resolution_time = int(market["resolutionTime"])
        cutoff_time = resolution_time - int(horizon_days * DAY_MS)
        bets = prefetched_bets.get(str(market["id"]), [])
        if not bets:
            notes.append(f"Skipped {market['id']}: no bets returned.")
            continue
        features = compute_market_features(
            market,
            bets,
            cutoff_time_ms=cutoff_time,
            horizon_days=horizon_days,
        )
        if features.bets_total < 3 or features.age_days < horizon_days:
            notes.append(f"Skipped {market['id']}: too little pre-cutoff history.")
            continue
        schema = build_binary_market_schema(market, features)
        freeman_probability = freeman_probability_from_schema(schema)
        target_probability = float(_market_target_probability(market))
        freeman_probability_with_news: float | None = None
        freeman_with_news_brier: float | None = None
        historical_news_edge: float | None = None
        historical_news_titles: list[str] = []
        historical_news_query: str | None = None
        historical_news_article_count = 0
        llm_rationale: str | None = None
        llm_parameter_vector: dict[str, Any] | None = None
        if archive_provider is not None and llm_client is not None and idx < historical_news_limit:
            start_dt = datetime.fromtimestamp(
                (cutoff_time - int(historical_news_window_days * DAY_MS)) / 1000.0,
                tz=timezone.utc,
            )
            end_dt = datetime.fromtimestamp(cutoff_time / 1000.0, tz=timezone.utc)
            historical_news_query = _gdelt_query_from_market(market)
            try:
                historical_articles = archive_provider.search_articles(
                    historical_news_query,
                    start_time=start_dt,
                    end_time=end_dt,
                    max_records=25,
                )
                historical_news_article_count = len(historical_articles)
                historical_news_edge, historical_news_titles, _, ranked_articles = _gdelt_news_edge(
                    str(market["question"]),
                    historical_articles,
                )
                if ranked_articles:
                    llm_signal_text = _historical_signal_text(
                        market,
                        features=features,
                        articles=ranked_articles,
                        news_edge=historical_news_edge,
                    )
                    freeman_probability_with_news, parameter_vector = freeman_probability_with_llm_signal(
                        schema,
                        signal_text=llm_signal_text,
                        llm_client=llm_client,
                    )
                    freeman_with_news_brier = _brier_score(freeman_probability_with_news, target_probability)
                    llm_rationale = str(parameter_vector.get("rationale", "")).strip() or None
                    llm_parameter_vector = parameter_vector
            except Exception as exc:  # pragma: no cover - exercised only in live runs.
                notes.append(f"Historical news calibration failed for {market['id']}: {exc}")
        forecast = Forecast(
            forecast_id=f"manifold:{market['id']}:{cutoff_time}",
            domain_id=str(schema["domain_id"]),
            outcome_id="yes",
            predicted_prob=freeman_probability,
            session_id=f"manifold:{market['id']}",
            horizon_steps=horizon_days,
            created_at=datetime.fromtimestamp(cutoff_time / 1000.0, tz=timezone.utc),
            created_step=0,
        )
        registry.record(forecast)
        registry.verify(
            forecast.forecast_id,
            actual_prob=target_probability,
            verified_at=datetime.fromtimestamp(resolution_time / 1000.0, tz=timezone.utc),
        )
        backtest_results.append(
            ManifoldBacktestResult(
                market_id=str(market["id"]),
                question=str(market["question"]),
                target_probability=target_probability,
                cutoff_probability=float(features.cutoff_probability),
                freeman_probability=float(freeman_probability),
                market_brier=_brier_score(features.cutoff_probability, target_probability),
                freeman_brier=_brier_score(freeman_probability, target_probability),
                resolution=market.get("resolution"),
                resolution_time=resolution_time,
                cutoff_time=cutoff_time,
                features=features,
                freeman_probability_with_news=freeman_probability_with_news,
                freeman_with_news_brier=freeman_with_news_brier,
                historical_news_edge=historical_news_edge,
                historical_news_article_count=historical_news_article_count,
                historical_news_titles=historical_news_titles,
                historical_news_query=historical_news_query,
                llm_rationale=llm_rationale,
                llm_parameter_vector=llm_parameter_vector,
            )
        )

    live_snapshots: list[LiveMarketSnapshot] = []
    try:
        headlines = BBCRSSClient().fetch()
    except Exception as exc:  # pragma: no cover - network failures are nondeterministic.
        notes.append(f"BBC RSS fetch failed: {exc}")
        headlines = []

    open_markets = client.search_markets(
        term=term,
        limit=open_market_limit,
        market_filter="open",
        contract_type="BINARY",
        sort="liquidity",
        topic_slug="climate",
    )
    for market in open_markets:
        bets = client.get_bets(str(market["id"]))
        cutoff_time = int(_now_utc().timestamp() * 1000)
        features = compute_market_features(
            market,
            bets,
            cutoff_time_ms=cutoff_time,
            horizon_days=horizon_days,
        )
        news_edge, relevant_headlines, trigger_modes = _bbc_news_edge(str(market["question"]), headlines)
        schema = build_binary_market_schema(market, features, news_edge=news_edge)
        live_snapshots.append(
            LiveMarketSnapshot(
                market_id=str(market["id"]),
                question=str(market["question"]),
                market_probability=float(features.cutoff_probability),
                freeman_probability=float(freeman_probability_from_schema(schema)),
                news_edge=float(news_edge),
                relevant_headlines=relevant_headlines,
                trigger_modes=trigger_modes,
            )
        )

    return backtest_results, live_snapshots, notes


def run_manifold_climate_experiment(
    *,
    output_dir: str | Path,
    term: str = "climate",
    market_limit: int = 20,
    horizon_days: int = 14,
    open_market_limit: int = 5,
    search_terms: Sequence[str] | None = None,
    bootstrap_samples: int = 2000,
    historical_news_limit: int = 0,
    historical_news_window_days: int = 7,
    gdelt_rate_limit_seconds: float = 8.0,
    historical_news_provider: str = "auto",
    deepseek_api_key_path: str | Path | None = None,
    gnews_api_key_path: str | Path | None = None,
    newsapi_api_key_path: str | Path | None = None,
) -> dict[str, Any]:
    """Run the experiment and persist summary artifacts."""

    output_path = Path(output_dir).resolve()
    output_path.mkdir(parents=True, exist_ok=True)
    active_terms = tuple(search_terms or DEFAULT_MARKET_SEARCH_TERMS)
    backtest_results, live_snapshots, notes = fetch_and_run_experiment(
        term=term,
        market_limit=market_limit,
        horizon_days=horizon_days,
        open_market_limit=open_market_limit,
        search_terms=active_terms,
        historical_news_limit=historical_news_limit,
        historical_news_window_days=historical_news_window_days,
        gdelt_rate_limit_seconds=gdelt_rate_limit_seconds,
        historical_news_provider=historical_news_provider,
        deepseek_api_key_path=deepseek_api_key_path,
        gnews_api_key_path=gnews_api_key_path,
        newsapi_api_key_path=newsapi_api_key_path,
    )
    market_brier_mean = (
        sum(result.market_brier for result in backtest_results) / len(backtest_results)
        if backtest_results
        else None
    )
    freeman_brier_mean = (
        sum(result.freeman_brier for result in backtest_results) / len(backtest_results)
        if backtest_results
        else None
    )
    market_only_deltas = [result.freeman_brier - result.market_brier for result in backtest_results]
    market_only_ci_low, market_only_ci_high = _bootstrap_mean_ci(
        market_only_deltas,
        bootstrap_samples=bootstrap_samples,
    )
    historical_news_results = [result for result in backtest_results if result.freeman_with_news_brier is not None]
    freeman_with_news_brier_mean = (
        sum(result.freeman_with_news_brier for result in historical_news_results if result.freeman_with_news_brier is not None)
        / len(historical_news_results)
        if historical_news_results
        else None
    )
    historical_news_deltas = [
        float(result.freeman_with_news_brier - result.market_brier)
        for result in historical_news_results
        if result.freeman_with_news_brier is not None
    ]
    historical_ci_low, historical_ci_high = _bootstrap_mean_ci(
        historical_news_deltas,
        bootstrap_samples=bootstrap_samples,
    )
    report = ExperimentReport(
        created_at=_now_utc().isoformat(),
        term=term,
        market_terms=list(active_terms),
        horizon_days=int(horizon_days),
        market_limit=int(market_limit),
        bootstrap_samples=int(bootstrap_samples),
        resolved_evaluated=len(backtest_results),
        skipped_markets=max(int(market_limit) - len(backtest_results), 0),
        market_brier_mean=market_brier_mean,
        freeman_brier_mean=freeman_brier_mean,
        freeman_minus_market_brier=(
            float(freeman_brier_mean - market_brier_mean)
            if market_brier_mean is not None and freeman_brier_mean is not None
            else None
        ),
        freeman_minus_market_brier_ci_low=market_only_ci_low,
        freeman_minus_market_brier_ci_high=market_only_ci_high,
        historical_news_evaluated=len(historical_news_results),
        freeman_with_news_brier_mean=freeman_with_news_brier_mean,
        freeman_with_news_minus_market_brier=(
            float(freeman_with_news_brier_mean - market_brier_mean)
            if market_brier_mean is not None and freeman_with_news_brier_mean is not None
            else None
        ),
        freeman_with_news_minus_market_brier_ci_low=historical_ci_low,
        freeman_with_news_minus_market_brier_ci_high=historical_ci_high,
        live_market_count=len(live_snapshots),
        notes=notes,
    )
    _write_json(output_path / "summary.json", asdict(report))
    _write_json(
        output_path / "backtest.json",
        [asdict(result) for result in backtest_results],
    )
    _write_json(
        output_path / "live_snapshot.json",
        [asdict(snapshot) for snapshot in live_snapshots],
    )
    _write_csv(output_path / "backtest.csv", (_flatten_backtest_row(result) for result in backtest_results))
    return {
        "summary": asdict(report),
        "backtest": [asdict(result) for result in backtest_results],
        "live_snapshot": [asdict(snapshot) for snapshot in live_snapshots],
        "output_dir": str(output_path),
    }
