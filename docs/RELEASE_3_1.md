# Freeman 3.1

`3.1.0` promotes the new ETL bootstrap to the primary runtime path for building a Freeman state vector from a natural-language domain brief.

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
