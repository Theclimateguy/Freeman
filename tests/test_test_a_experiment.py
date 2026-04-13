"""Tests for formal Test A helpers."""

from __future__ import annotations

from pathlib import Path

from freeman.realworld.manifold import ManifoldBacktestResult, MarketFeatures
from freeman.realworld.test_a_experiment import filter_test_a_rows, load_market_ids


def _result(*, market_id: str, edge: float | None, article_count: int, with_news_brier: float | None) -> ManifoldBacktestResult:
    features = MarketFeatures(
        cutoff_probability=0.5,
        probability_7d=0.5,
        probability_30d=0.5,
        momentum_7d=0.0,
        momentum_30d=0.0,
        flow_7d=0.0,
        flow_30d=0.0,
        turnover_7d=0.0,
        turnover_30d=0.0,
        bets_total=10,
        bets_7d=1,
        bets_30d=2,
        liquidity=100.0,
        age_days=40.0,
        horizon_days=14.0,
        cutoff_time_ms=1,
    )
    return ManifoldBacktestResult(
        market_id=market_id,
        question=f"Question {market_id}",
        target_probability=1.0,
        cutoff_probability=0.5,
        freeman_probability=0.5,
        market_brier=0.25,
        freeman_brier=0.25,
        resolution="YES",
        resolution_time=2,
        cutoff_time=1,
        features=features,
        freeman_probability_with_news=0.6 if with_news_brier is not None else None,
        freeman_with_news_brier=with_news_brier,
        historical_news_edge=edge,
        historical_news_article_count=article_count,
    )


def test_filter_test_a_rows_requires_articles_nonzero_edge_and_llm_output() -> None:
    kept = filter_test_a_rows(
        [
            _result(market_id="keep", edge=0.2, article_count=4, with_news_brier=0.16),
            _result(market_id="zero_edge", edge=0.0, article_count=4, with_news_brier=0.16),
            _result(market_id="no_articles", edge=0.2, article_count=0, with_news_brier=0.16),
            _result(market_id="no_llm", edge=0.2, article_count=4, with_news_brier=None),
        ]
    )

    assert [row.market_id for row in kept] == ["keep"]


def test_load_market_ids_ignores_blank_lines(tmp_path: Path) -> None:
    path = tmp_path / "ids.txt"
    path.write_text("m1\n\nm2\n", encoding="utf-8")

    assert load_market_ids(path) == ["m1", "m2"]
