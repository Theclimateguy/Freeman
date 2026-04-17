"""Dedicated Test A runner over a curated inclusion list."""

from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Callable, Sequence

from freeman_librarian.llm.deepseek import DeepSeekChatClient
from freeman_librarian.realworld.manifold import (
    DAY_MS,
    ManifoldBacktestResult,
    ManifoldClient,
    _brier_score,
    _bootstrap_mean_ci,
    _flatten_backtest_row,
    _gdelt_news_edge,
    _gdelt_query_from_market,
    _historical_signal_text,
    _load_deepseek_api_key,
    _market_target_probability,
    _now_utc,
    _prefetch_bets,
    _write_csv,
    _write_json,
    build_historical_news_provider,
    build_binary_market_schema,
    compute_market_features,
    freeman_probability_from_schema,
    freeman_probability_with_llm_signal,
)


ProgressCallback = Callable[[str], None]


def load_market_ids(path: str | Path) -> list[str]:
    """Load one market id per line."""

    return [line.strip() for line in Path(path).resolve().read_text(encoding="utf-8").splitlines() if line.strip()]


def filter_test_a_rows(results: Sequence[ManifoldBacktestResult]) -> list[ManifoldBacktestResult]:
    """Return the formal Test A subset with an actual historical news signal."""

    return [
        result
        for result in results
        if result.freeman_with_news_brier is not None
        and result.historical_news_article_count > 0
        and result.historical_news_edge is not None
        and abs(float(result.historical_news_edge)) > 1.0e-12
    ]


def run_test_a_experiment(
    *,
    market_ids_path: str | Path,
    output_dir: str | Path,
    horizon_days: int = 14,
    historical_news_window_days: int = 30,
    gdelt_rate_limit_seconds: float = 8.0,
    historical_news_provider: str = "auto",
    bootstrap_samples: int = 500,
    deepseek_api_key_path: str | Path | None = None,
    gnews_api_key_path: str | Path | None = None,
    newsapi_api_key_path: str | Path | None = None,
    progress_callback: ProgressCallback | None = None,
) -> dict[str, Any]:
    """Run Test A on a curated set of market ids."""

    def emit(message: str) -> None:
        if progress_callback is not None:
            progress_callback(message)

    output_path = Path(output_dir).resolve()
    output_path.mkdir(parents=True, exist_ok=True)
    market_ids = load_market_ids(market_ids_path)
    client = ManifoldClient()
    archive_provider = build_historical_news_provider(
        provider=historical_news_provider,
        gnews_api_key_path=gnews_api_key_path,
        newsapi_api_key_path=newsapi_api_key_path,
        gdelt_rate_limit_seconds=gdelt_rate_limit_seconds,
    )
    llm_client = DeepSeekChatClient(api_key=_load_deepseek_api_key(deepseek_api_key_path))

    emit(f"Loaded {len(market_ids)} curated market ids.")
    markets: list[dict[str, Any]] = []
    for idx, market_id in enumerate(market_ids, start=1):
        market = client.get_market(market_id)
        markets.append(market)
        emit(f"[fetch {idx}/{len(market_ids)}] {market_id} :: {market.get('question', '')}")

    prefetched_bets, prefetch_notes = _prefetch_bets(client, markets)
    notes: list[str] = list(prefetch_notes)
    notes.append(f"DeepSeek model enabled via DS.txt/env using model={llm_client.model}.")
    notes.append(f"Historical news provider enabled: {archive_provider.name}.")
    notes.append(f"Test A uses a curated inclusion list of {len(market_ids)} market ids.")

    backtest_results: list[ManifoldBacktestResult] = []
    total_markets = len(markets)
    for idx, market in enumerate(markets, start=1):
        market_id = str(market["id"])
        question = str(market.get("question", ""))
        resolution_time = int(market["resolutionTime"])
        cutoff_time = resolution_time - int(horizon_days * DAY_MS)
        bets = prefetched_bets.get(market_id, [])
        if not bets:
            notes.append(f"Skipped {market_id}: no bets returned.")
            emit(f"[{idx}/{total_markets}] {market_id} skipped: no bets.")
            continue

        features = compute_market_features(
            market,
            bets,
            cutoff_time_ms=cutoff_time,
            horizon_days=horizon_days,
        )
        if features.bets_total < 3 or features.age_days < horizon_days:
            notes.append(f"Skipped {market_id}: too little pre-cutoff history.")
            emit(f"[{idx}/{total_markets}] {market_id} skipped: too little history.")
            continue

        schema = build_binary_market_schema(market, features)
        freeman_probability = freeman_probability_from_schema(schema)
        target_probability = float(_market_target_probability(market))
        freeman_probability_with_news: float | None = None
        freeman_with_news_brier: float | None = None
        historical_news_edge: float | None = None
        historical_news_titles: list[str] = []
        historical_news_query = _gdelt_query_from_market(market)
        historical_news_article_count = 0
        llm_rationale: str | None = None
        llm_parameter_vector: dict[str, Any] | None = None

        start_dt = datetime.fromtimestamp(
            (cutoff_time - int(historical_news_window_days * DAY_MS)) / 1000.0,
            tz=timezone.utc,
        )
        end_dt = datetime.fromtimestamp(cutoff_time / 1000.0, tz=timezone.utc)
        try:
            historical_articles = archive_provider.search_articles(
                historical_news_query,
                start_time=start_dt,
                end_time=end_dt,
                max_records=50,
            )
            historical_news_article_count = len(historical_articles)
            historical_news_edge, historical_news_titles, _, ranked_articles = _gdelt_news_edge(question, historical_articles)
            if ranked_articles and historical_news_edge is not None and abs(float(historical_news_edge)) > 1.0e-12:
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
            emit(
                f"[{idx}/{total_markets}] {market_id} articles={historical_news_article_count} "
                f"edge={0.0 if historical_news_edge is None else historical_news_edge:+.4f} "
                f"llm={'yes' if freeman_probability_with_news is not None else 'no'} :: {question}"
            )
        except Exception as exc:  # pragma: no cover - exercised only in live runs.
            notes.append(f"Historical news calibration failed for {market_id}: {exc}")
            emit(f"[{idx}/{total_markets}] {market_id} failed: {exc}")

        backtest_results.append(
            ManifoldBacktestResult(
                market_id=market_id,
                question=question,
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

    test_a_rows = filter_test_a_rows(backtest_results)
    deltas = [
        float(result.freeman_with_news_brier - result.market_brier)
        for result in test_a_rows
        if result.freeman_with_news_brier is not None
    ]
    delta_ci_low, delta_ci_high = _bootstrap_mean_ci(deltas, bootstrap_samples=bootstrap_samples)
    market_brier_mean = (
        sum(result.market_brier for result in test_a_rows) / len(test_a_rows)
        if test_a_rows
        else None
    )
    freeman_with_news_brier_mean = (
        sum(result.freeman_with_news_brier for result in test_a_rows if result.freeman_with_news_brier is not None)
        / len(test_a_rows)
        if test_a_rows
        else None
    )
    summary = {
        "created_at": _now_utc().isoformat(),
        "market_ids_path": str(Path(market_ids_path).resolve()),
        "candidate_market_count": len(market_ids),
        "evaluated_market_count": len(backtest_results),
        "historical_news_window_days": int(historical_news_window_days),
        "bootstrap_samples": int(bootstrap_samples),
        "article_positive_count": sum(result.historical_news_article_count > 0 for result in backtest_results),
        "news_signal_count": len(test_a_rows),
        "news_signal_market_ids": [result.market_id for result in test_a_rows],
        "market_brier_mean": market_brier_mean,
        "freeman_with_news_brier_mean": freeman_with_news_brier_mean,
        "delta_news_brier": (
            float(freeman_with_news_brier_mean - market_brier_mean)
            if market_brier_mean is not None and freeman_with_news_brier_mean is not None
            else None
        ),
        "delta_news_brier_ci_low": delta_ci_low,
        "delta_news_brier_ci_high": delta_ci_high,
        "passes_hypothesis": bool(
            test_a_rows
            and freeman_with_news_brier_mean is not None
            and market_brier_mean is not None
            and freeman_with_news_brier_mean < market_brier_mean
            and delta_ci_high is not None
            and delta_ci_high < 0.0
        ),
        "n_requirement_met": bool(len(test_a_rows) >= 30),
        "notes": notes + [
            "Formal Test A subset includes only markets with article_count > 0 and historical_news_edge != 0.",
            "DeepSeek DS.txt is used for every news-conditioned update.",
            f"Cutoff times are horizon-based: tau_i = resolution_time - {horizon_days} days.",
        ],
    }

    _write_json(output_path / "summary.json", summary)
    _write_json(output_path / "backtest.json", [asdict(result) for result in backtest_results])
    _write_json(output_path / "test_a_subset.json", [asdict(result) for result in test_a_rows])
    _write_csv(output_path / "backtest.csv", (_flatten_backtest_row(result) for result in backtest_results))
    _write_csv(output_path / "test_a_subset.csv", (_flatten_backtest_row(result) for result in test_a_rows))
    emit(
        "Completed Test A run: "
        f"news_signal_count={len(test_a_rows)}, delta_news_brier={summary['delta_news_brier']}, "
        f"ci=[{summary['delta_news_brier_ci_low']}, {summary['delta_news_brier_ci_high']}]"
    )
    return {
        "summary": summary,
        "backtest": [asdict(result) for result in backtest_results],
        "test_a_subset": [asdict(result) for result in test_a_rows],
        "output_dir": str(output_path),
    }


__all__ = [
    "filter_test_a_rows",
    "load_market_ids",
    "run_test_a_experiment",
]
