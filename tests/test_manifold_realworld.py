"""Tests for the Manifold real-world experiment helpers."""

from __future__ import annotations

import pytest

from freeman.core.scorer import score_outcomes
from freeman.core.types import ParameterVector
from freeman.domain.compiler import DomainCompiler
from freeman.realworld.manifold import (
    ChainedHistoricalNewsProvider,
    GDELTDocClient,
    GNewsArchiveClient,
    MarketBet,
    MarketFeatures,
    NewsAPIArchiveClient,
    build_historical_news_provider,
    build_binary_market_schema,
    compute_market_features,
    freeman_probability_from_schema,
    reconstruct_probability_path,
)


def test_reconstruct_probability_path_and_cutoff_features() -> None:
    market = {
        "id": "m1",
        "question": "Will X happen?",
        "createdTime": 1_000,
        "resolutionTime": 40 * 86_400_000,
        "totalLiquidity": 1_000,
    }
    bets = [
        MarketBet("b1", "m1", 2_000, 0.50, 0.60, 50.0, "YES"),
        MarketBet("b2", "m1", 3_000, 0.60, 0.70, 75.0, "YES"),
        MarketBet("b3", "m1", 4_000, 0.70, 0.65, 20.0, "NO"),
    ]

    path = reconstruct_probability_path(market, bets)
    features = compute_market_features(
        market,
        bets,
        cutoff_time_ms=3_500,
        horizon_days=14,
    )

    assert path == [(1_000, 0.50), (2_000, 0.60), (3_000, 0.70), (4_000, 0.65)]
    assert features.cutoff_probability == pytest.approx(0.70)
    assert features.flow_30d == pytest.approx(0.20)
    assert features.bets_total == 3


def test_freeman_schema_reproduces_market_probability_without_adjustments() -> None:
    market = {
        "id": "m2",
        "question": "Will X happen?",
        "createdTime": 1_000,
        "resolutionTime": 40 * 86_400_000,
        "totalLiquidity": 1_000,
    }
    features = MarketFeatures(
        cutoff_probability=0.73,
        probability_7d=0.73,
        probability_30d=0.73,
        momentum_7d=0.0,
        momentum_30d=0.0,
        flow_7d=0.0,
        flow_30d=0.0,
        turnover_7d=0.0,
        turnover_30d=0.0,
        bets_total=1,
        bets_7d=0,
        bets_30d=0,
        liquidity=1_000.0,
        age_days=40.0,
        horizon_days=14.0,
        cutoff_time_ms=2_500,
    )
    schema = build_binary_market_schema(market, features, news_edge=0.0)
    probability = freeman_probability_from_schema(schema)

    assert probability == pytest.approx(features.cutoff_probability, abs=1.0e-3)


def test_build_binary_market_schema_infers_domain_polarity() -> None:
    market = {
        "id": "risk",
        "question": "Will global CO2 emissions in 2024 deplete the remaining carbon budget to limit warming to 1.5°C?",
        "textDescription": "",
        "createdTime": 1_000,
        "resolutionTime": 40 * 86_400_000,
        "totalLiquidity": 1_000,
    }
    positive_market = {
        "id": "goal",
        "question": "Will annual US CO2 emissions be below 4.5 billion tons in 2030?",
        "textDescription": "",
        "createdTime": 1_000,
        "resolutionTime": 40 * 86_400_000,
        "totalLiquidity": 1_000,
    }
    features = MarketFeatures(
        cutoff_probability=0.40,
        probability_7d=0.39,
        probability_30d=0.38,
        momentum_7d=0.01,
        momentum_30d=0.02,
        flow_7d=0.01,
        flow_30d=0.02,
        turnover_7d=10.0,
        turnover_30d=20.0,
        bets_total=10,
        bets_7d=3,
        bets_30d=5,
        liquidity=1_000.0,
        age_days=40.0,
        horizon_days=14.0,
        cutoff_time_ms=2_500,
    )

    risk_schema = build_binary_market_schema(market, features)
    goal_schema = build_binary_market_schema(positive_market, features)

    assert risk_schema["domain_polarity"] == "negative"
    assert risk_schema["modifier_mode"] == "probability_monotonic"
    assert risk_schema["metadata"]["domain_polarity"] == "negative"
    assert goal_schema["domain_polarity"] == "positive"


def test_probability_monotonic_modifier_increases_risk_yes_probability() -> None:
    market = {
        "id": "risk2",
        "question": "Will global CO2 emissions in 2024 deplete the remaining carbon budget to limit warming to 1.5°C?",
        "textDescription": "",
        "createdTime": 1_000,
        "resolutionTime": 40 * 86_400_000,
        "totalLiquidity": 1_000,
    }
    features = MarketFeatures(
        cutoff_probability=0.28676691429401285,
        probability_7d=0.28676691429401285,
        probability_30d=0.28676691429401285,
        momentum_7d=0.0,
        momentum_30d=0.0,
        flow_7d=0.0,
        flow_30d=0.0,
        turnover_7d=0.0,
        turnover_30d=0.0,
        bets_total=10,
        bets_7d=0,
        bets_30d=0,
        liquidity=1_000.0,
        age_days=40.0,
        horizon_days=14.0,
        cutoff_time_ms=2_500,
    )

    schema = build_binary_market_schema(market, features)
    world = DomainCompiler().compile(schema)
    base_probability = score_outcomes(world)["yes"]

    world.parameter_vector = ParameterVector(outcome_modifiers={"yes": 1.5})
    updated_probability = score_outcomes(world)["yes"]

    assert schema["domain_polarity"] == "negative"
    assert updated_probability > base_probability


def test_build_historical_news_provider_defaults_to_gdelt_without_key(monkeypatch) -> None:
    monkeypatch.delenv("NEWSAPI_API_KEY", raising=False)
    monkeypatch.delenv("NEWS_API_KEY", raising=False)
    provider = build_historical_news_provider(
        provider="auto",
        newsapi_api_key_path="/tmp/definitely_missing_newsapi_key.txt",
    )

    assert isinstance(provider, GDELTDocClient)


def test_build_historical_news_provider_uses_newsapi_when_key_file_exists(tmp_path, monkeypatch) -> None:
    monkeypatch.delenv("NEWSAPI_API_KEY", raising=False)
    monkeypatch.delenv("NEWS_API_KEY", raising=False)
    key_path = tmp_path / "NEWSAPI.txt"
    key_path.write_text("demo-key", encoding="utf-8")

    provider = build_historical_news_provider(
        provider="newsapi",
        newsapi_api_key_path=key_path,
    )

    assert isinstance(provider, NewsAPIArchiveClient)


def test_build_historical_news_provider_uses_gnews_when_requested(tmp_path, monkeypatch) -> None:
    monkeypatch.delenv("GNEWS_API_KEY", raising=False)
    key_path = tmp_path / "GNEWS.txt"
    key_path.write_text("demo-key", encoding="utf-8")

    provider = build_historical_news_provider(
        provider="gnews",
        gnews_api_key_path=key_path,
    )

    assert isinstance(provider, GNewsArchiveClient)


def test_build_historical_news_provider_auto_prefers_gnews_then_gdelt(tmp_path, monkeypatch) -> None:
    monkeypatch.delenv("GNEWS_API_KEY", raising=False)
    monkeypatch.delenv("NEWSAPI_API_KEY", raising=False)
    monkeypatch.delenv("NEWS_API_KEY", raising=False)
    key_path = tmp_path / "GNEWS.txt"
    key_path.write_text("demo-key", encoding="utf-8")

    provider = build_historical_news_provider(
        provider="auto",
        gnews_api_key_path=key_path,
        newsapi_api_key_path=tmp_path / "missing_newsapi.txt",
    )

    assert isinstance(provider, ChainedHistoricalNewsProvider)
    assert [type(item) for item in provider.providers] == [GNewsArchiveClient, GDELTDocClient]
