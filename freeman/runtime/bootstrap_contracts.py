"""Catalog and serialization helpers for ontology bootstrap strategies."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class BootstrapStrategy:
    """Machine-readable description of a supported ontology bootstrap strategy."""

    strategy_id: str
    entrypoint: str
    summary: str
    input_requirements: tuple[str, ...]
    recommended_for: tuple[str, ...]
    limitations: tuple[str, ...]


_STRATEGIES: dict[str, BootstrapStrategy] = {
    "seed_schema": BootstrapStrategy(
        strategy_id="seed_schema",
        entrypoint="schema_path",
        summary="Start from an explicit seed ontology or hand-authored graph.",
        input_requirements=(
            "Requires an existing Freeman schema JSON.",
            "Best results when actor/resource/outcome ids are already stable.",
        ),
        recommended_for=(
            "production baselines",
            "high reproducibility",
            "seed-graph-first ontology work",
        ),
        limitations=(
            "Coverage is capped by the seed ontology until later ontology repair expands it.",
            "Weak choice when no usable schema or seed graph exists.",
        ),
    ),
    "brief_local_etl": BootstrapStrategy(
        strategy_id="brief_local_etl",
        entrypoint="llm_synthesize",
        summary="Generate the ontology from a natural-language brief using a local model.",
        input_requirements=(
            "Requires a compact domain brief with explicit actors, resources, and outcomes.",
            "Requires a local chat model that supports schema repair calls.",
        ),
        recommended_for=(
            "local-only runs",
            "privacy-sensitive domains",
            "cheap iterative prototyping",
        ),
        limitations=(
            "Bootstrap quality is sensitive to brief quality and local-model capability.",
            "Small or slow local models may timeout or under-specify the ontology.",
        ),
    ),
    "brief_local_etl_with_fallback_seed": BootstrapStrategy(
        strategy_id="brief_local_etl_with_fallback_seed",
        entrypoint="llm_synthesize",
        summary="Try local ETL from a brief, but preserve continuity with an explicit fallback seed graph.",
        input_requirements=(
            "Requires a domain brief plus a fallback Freeman schema JSON.",
            "Requires a local chat model that supports schema repair calls.",
        ),
        recommended_for=(
            "local-first production runs",
            "bootstraps that must always return a graph",
            "seed-assisted ontology induction",
        ),
        limitations=(
            "If ETL fails, the resulting graph may reflect the fallback seed more than the brief.",
            "A poor fallback schema can hide ETL failures behind a superficially valid graph.",
        ),
    ),
    "brief_remote_etl": BootstrapStrategy(
        strategy_id="brief_remote_etl",
        entrypoint="llm_synthesize",
        summary="Generate the ontology from a natural-language brief using a remote API model.",
        input_requirements=(
            "Requires a compact domain brief with explicit actors, resources, and outcomes.",
            "Requires remote API credentials and network access.",
        ),
        recommended_for=(
            "complex ontology induction",
            "higher-fidelity first-pass bootstraps",
            "research-oriented schema synthesis",
        ),
        limitations=(
            "Introduces API cost, latency variance, and data-governance constraints.",
            "Reproducibility is weaker unless prompts, model version, and brief are pinned.",
        ),
    ),
    "brief_remote_etl_with_fallback_seed": BootstrapStrategy(
        strategy_id="brief_remote_etl_with_fallback_seed",
        entrypoint="llm_synthesize",
        summary="Try remote ETL from a brief, but preserve continuity with an explicit fallback seed graph.",
        input_requirements=(
            "Requires a domain brief, remote API credentials, and a fallback Freeman schema JSON.",
            "Requires network access to the configured provider.",
        ),
        recommended_for=(
            "high-value ontology bootstraps",
            "API-assisted production runs that still need a deterministic safety net",
            "complex domains with an existing seed graph",
        ),
        limitations=(
            "If ETL fails, the final graph still collapses to the fallback seed ontology.",
            "Operational risk spans both remote dependency risk and fallback-schema drift.",
        ),
    ),
}


def list_bootstrap_strategies() -> list[dict[str, Any]]:
    """Return the supported ontology bootstrap strategies as JSON-serializable payloads."""

    return [asdict(strategy) for strategy in _STRATEGIES.values()]


def resolve_bootstrap_strategy(
    *,
    bootstrap_mode: str,
    llm_provider: str,
    has_fallback_schema: bool,
) -> BootstrapStrategy:
    """Resolve the configured bootstrap strategy from mode/provider/fallback flags."""

    normalized_mode = str(bootstrap_mode or "").strip().lower()
    normalized_provider = str(llm_provider or "").strip().lower()
    if normalized_mode == "schema_path":
        return _STRATEGIES["seed_schema"]
    if normalized_mode != "llm_synthesize":
        raise ValueError(f"Unsupported bootstrap mode for strategy resolution: {bootstrap_mode}")
    remote_provider = normalized_provider in {"openai", "deepseek"}
    if remote_provider and has_fallback_schema:
        return _STRATEGIES["brief_remote_etl_with_fallback_seed"]
    if remote_provider:
        return _STRATEGIES["brief_remote_etl"]
    if has_fallback_schema:
        return _STRATEGIES["brief_local_etl_with_fallback_seed"]
    return _STRATEGIES["brief_local_etl"]


def build_bootstrap_contract(
    *,
    bootstrap_mode: str,
    llm_provider: str,
    model_name: str,
    has_fallback_schema: bool,
    actual_bootstrap_path: str,
    schema_path: str | None = None,
    fallback_schema_path: str | None = None,
    domain_brief_supplied: bool = False,
) -> dict[str, Any]:
    """Build a persisted contract describing how the current ontology was bootstrapped."""

    strategy = resolve_bootstrap_strategy(
        bootstrap_mode=bootstrap_mode,
        llm_provider=llm_provider,
        has_fallback_schema=has_fallback_schema,
    )
    contract = asdict(strategy)
    contract.update(
        {
            "actual_bootstrap_path": str(actual_bootstrap_path),
            "llm_provider": str(llm_provider),
            "model_name": str(model_name),
            "domain_brief_supplied": bool(domain_brief_supplied),
            "schema_path": str(schema_path) if schema_path else None,
            "fallback_schema_path": str(fallback_schema_path) if fallback_schema_path else None,
        }
    )
    return contract
