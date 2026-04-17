# Freeman 3.0

`3.0.0` is the first Freeman release where the local operational loop is closed end-to-end.

## What Changed

- Universal semantic retrieval is now the default retrieval contract across CLI and runtime artifacts.
- `domain_brief -> llm_synthesize` is now the primary bootstrap path; `schema_path` remains as a secondary/fallback operational mode.
- LLM bootstrap normalizes common relation aliases (`source/target/type`) into Freeman's canonical `source_id/target_id/relation_type` schema contract before compile/repair.
- `ask` / semantic `query` run over persisted KG nodes, forecasts, causal edges, and world state, not only raw graph nodes.
- `schema_path` ontology repair is autonomous by default: runtime can extend schema metadata and infer new causal edges without a manual approval step.
- Budget/cost governance is enforced in runtime, not only estimated offline:
  - `signal_processing`
  - `ontology_repair`
  - `answer_generation`
- Every governed task is written to `runtime/cost_ledger.jsonl` and surfaced in `freeman status`.

## Operational Contract

Freeman `3.0` supports the full local cycle:

1. ingest a stream of signals into a persistent runtime
2. update a world model and verify forecasts over monotonic `runtime_step`
3. detect ontology gaps from anomaly pressure
4. repair the active schema/world model automatically for schema-backed runtimes
5. answer semantic questions from persisted runtime evidence
6. enforce compute budget with downgrade/stop policy and durable accounting

## Remaining Non-Blockers

These are quality/tuning issues, not architectural gaps:

- provider-specific calibration of cost coefficients
- domain-specific retrieval thresholds on very noisy streams
- autonomous relation induction remains safest when new links stay within an existing schema vocabulary
