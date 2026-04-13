"""Tests for Test A preflight stratification."""

from __future__ import annotations

from freeman.realworld.test_a_preflight import (
    build_recommended_inclusion_list,
    classify_test_a_market,
    stratify_backtest_rows,
)


def test_classify_test_a_market_assigns_expected_strata() -> None:
    assert classify_test_a_market(
        "Carbon Brief Forecast: If Biden wins, will annual US CO2 emissions be below 4.5 billion tons in 2030?"
    )[:3] == ("policy_to_sector", "core_non_obvious", True)

    assert classify_test_a_market(
        "Will the World Bank or IMF establish a dedicated climate disaster relief fund of at least $50 billion by the end of 2025"
    )[:3] == ("policy_to_policy_finance", "core_non_obvious", True)

    assert classify_test_a_market(
        "Will climate change plaintiffs win in Held v Montana?"
    )[:3] == ("legal_to_policy", "core_non_obvious", True)

    assert classify_test_a_market(
        "Will the ocean surface temperature (SST) record set in August 2023 be exceeded in the next year?"
    )[:3] == ("physical_climate", "control", True)

    assert classify_test_a_market(
        'Will the "First Room-Temperature Ambient-Pressure Superconductor" paper be replicated by October 1st?'
    )[:3] == ("exclude_noise", "exclude", False)

    assert classify_test_a_market(
        "Will the highest temperature in NYC in Mar 28, 2024 be higher than the previous day?"
    )[:3] == ("exclude_local_weather", "exclude", False)


def test_build_recommended_inclusion_list_prioritizes_non_obvious_core_rows() -> None:
    rows = [
        {"market_id": "m1", "question": "If Trump is elected president again, will the US withdraw from the Paris Agreement a second time?"},
        {"market_id": "m2", "question": "Carbon Brief Forecast: If Biden wins, will annual US CO2 emissions be below 4.5 billion tons in 2030?"},
        {"market_id": "m3", "question": "Will global CO2 emissions decrease in 2024?"},
        {"market_id": "m4", "question": "Will the ocean surface temperature (SST) record set in August 2023 be exceeded in the next year?"},
        {"market_id": "m5", "question": 'Will the "First Room-Temperature Ambient-Pressure Superconductor" paper be replicated by October 1st?'},
    ]

    stratified = stratify_backtest_rows(rows)
    selected = build_recommended_inclusion_list(stratified, physical_control_limit=1)

    assert [row.market_id for row in selected] == ["m2", "m1", "m3", "m4"]
