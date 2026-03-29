"""Real-world experiment adapters and utilities."""

from freeman.realworld.manifold import (
    BBCRSSClient,
    GDELTDocClient,
    LiveMarketSnapshot,
    ManifoldBacktestResult,
    ManifoldClient,
    MarketFeatures,
    build_binary_market_schema,
    compute_market_features,
    fetch_and_run_experiment,
    freeman_probability_from_schema,
    freeman_probability_with_llm_signal,
    run_manifold_climate_experiment,
)
from freeman.realworld.test_a_preflight import (
    TestAMarketRow,
    build_recommended_inclusion_list,
    classify_test_a_market,
    run_test_a_preflight,
    stratify_backtest_rows,
)
from freeman.realworld.test_a_experiment import (
    filter_test_a_rows,
    load_market_ids,
    run_test_a_experiment,
)
from freeman.realworld.test_c_cross_domain import (
    CrossDomainResult,
    CrossDomainTarget,
    build_cross_domain_schema,
    run_cross_domain_causal_test,
)

__all__ = [
    "BBCRSSClient",
    "GDELTDocClient",
    "LiveMarketSnapshot",
    "ManifoldBacktestResult",
    "ManifoldClient",
    "MarketFeatures",
    "TestAMarketRow",
    "build_binary_market_schema",
    "build_recommended_inclusion_list",
    "classify_test_a_market",
    "compute_market_features",
    "fetch_and_run_experiment",
    "filter_test_a_rows",
    "freeman_probability_from_schema",
    "freeman_probability_with_llm_signal",
    "load_market_ids",
    "CrossDomainResult",
    "CrossDomainTarget",
    "build_cross_domain_schema",
    "run_manifold_climate_experiment",
    "run_test_a_experiment",
    "run_test_a_preflight",
    "run_cross_domain_causal_test",
    "stratify_backtest_rows",
]
