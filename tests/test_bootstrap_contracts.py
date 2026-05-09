from freeman.runtime.bootstrap_contracts import build_bootstrap_contract, resolve_bootstrap_strategy


def test_resolve_bootstrap_strategy_prefers_seed_schema_for_schema_path() -> None:
    strategy = resolve_bootstrap_strategy(
        bootstrap_mode="schema_path",
        llm_provider="ollama",
        has_fallback_schema=False,
    )
    assert strategy.strategy_id == "seed_schema"
    assert strategy.entrypoint == "schema_path"


def test_resolve_bootstrap_strategy_distinguishes_local_remote_and_fallback() -> None:
    local = resolve_bootstrap_strategy(
        bootstrap_mode="llm_synthesize",
        llm_provider="ollama",
        has_fallback_schema=False,
    )
    remote = resolve_bootstrap_strategy(
        bootstrap_mode="llm_synthesize",
        llm_provider="openai",
        has_fallback_schema=False,
    )
    remote_with_fallback = resolve_bootstrap_strategy(
        bootstrap_mode="llm_synthesize",
        llm_provider="deepseek",
        has_fallback_schema=True,
    )

    assert local.strategy_id == "brief_local_etl"
    assert remote.strategy_id == "brief_remote_etl"
    assert remote_with_fallback.strategy_id == "brief_remote_etl_with_fallback_seed"


def test_build_bootstrap_contract_records_actual_materialization_path() -> None:
    contract = build_bootstrap_contract(
        bootstrap_mode="llm_synthesize",
        llm_provider="ollama",
        model_name="qwen2.5-coder:14b",
        has_fallback_schema=True,
        actual_bootstrap_path="fallback_schema_seed",
        fallback_schema_path="/tmp/fallback.json",
        domain_brief_supplied=True,
    )

    assert contract["strategy_id"] == "brief_local_etl_with_fallback_seed"
    assert contract["actual_bootstrap_path"] == "fallback_schema_seed"
    assert contract["fallback_schema_path"] == "/tmp/fallback.json"
    assert contract["domain_brief_supplied"] is True
