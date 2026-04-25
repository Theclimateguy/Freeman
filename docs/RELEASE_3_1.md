# Freeman 3.1

`3.1.0` promoted the new ETL bootstrap to the primary runtime path for building a Freeman state vector from a natural-language domain brief.

`3.1.1` hardens that release by making ontology-ingestion provenance explicit, fixing ETL resource normalization, and adding a transparent live-news benchmark harness.

## What Changed

- `llm_synthesize` in runtime now uses a two-phase ETL flow instead of one monolithic synthesis prompt:
  - `skeleton` extracts only actors, resources, and outcomes
  - `edges` adds `causal_dag`, `actor_update_rules`, policies, and assumptions against a verified node set
- Resource-target causal edges are materialized into bounded deterministic coupling weights during ETL so the compiled world is executable before runtime estimation.
- Level-2 sign failures now go through `repair_sign_edges()` first, which limits repair scope to failing edges before falling back to full-package repair.
- `bootstrap_attempts` now records `etl_phase` values so failed bootstraps are diagnosable without reading the full prompt history.
- Interactive single-call synthesis remains available for backward compatibility in `run()` / `synthesize_package()`, while daemon bootstrap uses ETL by default.

## Release Notes

- The practical effect is lower bootstrap variance on small models and clearer provenance for generated causal structure.
- Compile, level1, and level0 trial failures still use the existing verifier-guided full-package repair path.
- Runtime cost estimation now reflects that `llm_synthesize` usually spends two LLM calls, not one.
- Repository licensing is now explicit under Apache License 2.0.

## Validation

- `pytest -q` -> `234 passed`

## 3.1.1 Follow-up

### What Changed

- Runtime bootstrap artifacts now persist `bootstrap_contract`, which makes the ontology-ingestion strategy explicit:
  - `seed_schema`
  - `brief_local_etl`
  - `brief_local_etl_with_fallback_seed`
  - `brief_remote_etl`
  - `brief_remote_etl_with_fallback_seed`
- `bootstrap_contract.actual_bootstrap_path` now records whether the graph actually came from:
  - `schema_seed`
  - `etl_from_brief`
  - `fallback_schema_seed`
- Resource normalization in ETL now coerces `evolution_params` to the operator contract so `logistic` resources cannot leak incompatible linear parameters into compiled worlds.
- A transparent benchmark runner for random live RSS samples is now tracked in the repository:
  - [scripts/run_random_news_etl_benchmark.py](/Users/theclimateguy/Documents/science/Freeman/scripts/run_random_news_etl_benchmark.py)

### Why It Matters

- Freeman should now be understood as an ontology-construction runtime with multiple explicit ingestion strategies, not as a single "news to graph" path.
- Fallback-backed runs remain operationally useful, but the saved runtime package now makes it obvious when the final graph reflects a fallback seed rather than a successful ETL synthesis.

### Validation

- `pytest -q` -> `238 passed`
