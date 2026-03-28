# Changelog

## Unreleased

### Added

- `ParameterVector` as a universal dynamic calibration layer over static world schemas.
- `ParameterEstimator` for LLM-driven T1 calibration of `outcome_modifiers`, `shock_decay`, and `edge_weight_deltas`.
- Stateful shock accumulation in `WorldGraph.apply_shocks()` with decayed baseline-relative updates and persisted `_shock_state`.
- Conditional `Outcome.regime_shifts` with safe boolean condition parsing for nonlinear outcome flips under large shocks.
- FAAB benchmark package under `scripts/benchmark_faab/` with dummy dataset generation, mode-specific metric exports, and recorded longitudinal run artifacts.
- `docs/FAAB.md` describing benchmark modes, formulas, commands, and the recorded post-regime-shift run.
- Obligation-driven attention scheduling via `ForecastDebt`, `ConflictDebt`, `AnomalyDebt`, and `ObligationQueue`.
- `ForecastRegistry` with horizon tracking, pending/due queries, verification, and optional JSON persistence.
- Self-model reconciliation path that writes `self_observation` KG nodes from verified forecast errors.
- `ProactiveEmitter` for structured `alert`, `forecast_update`, and `question_to_human` interface events.
- `SignalMemory` with cross-session deduplication, exponential decay, and replay-oriented signal ingestion support.
- Deterministic `AgentHarness` plus JSONL replay fixtures for end-to-end behavioral tests.
- `HashingEmbeddingAdapter` for local semantic retrieval without an external embedding API.
- `OllamaEmbeddingClient` with batched `/api/embed` support, compatibility fallback to `/api/embeddings`, and model-aware prompt preparation for `nomic-embed-text` and `mxbai-embed-large`.
- `scripts/run_real_llm_e2e.py` for live DeepSeek evaluation across multiple domains with semantic-memory follow-up.
- `docs/REAL_LLM_E2E.md` documenting the recorded live E2E run and results.

### Changed

- `WorldState` / `WorldGraph` snapshots now persist `parameter_vector`, preserving dynamic calibration state across clone / JSON boundaries.
- Resource evolution, actor-state updates, and outcome scoring now read `ParameterVector` so T1 updates can modify simulation sensitivity without rewriting the base ontology.
- `AnalysisPipeline` now exposes `update(previous_world, parameter_vector, ...)` and writes `parameter_vector` into KG analysis metadata for auditability.
- `MODE_A_FULL` in FAAB now uses `ParameterEstimator.estimate()` plus `AnalysisPipeline.update()` on `T1`.
- Outcome scoring now evaluates regime-shift conditions against accumulated shock state rather than only current absolute levels.
- The FAAB benchmark runner now applies stateful `T_1` updates through `WorldGraph.apply_shocks(..., time_decay=0.5)` and includes macro / film regime-shift rules in its domain templates.
- Repository documentation now covers decayed state updates, nonlinear scoring, and the FAAB benchmark workflow and results.
- `ManualSignalSource` now accepts replay mappings with extra top-level fields and folds them into `Signal.metadata`.
- Repository documentation now covers obligation pressure, forecast verification, self-model feedback, and replay-based testing flows.
- Default embedding configuration now targets a local Ollama daemon with `nomic-embed-text`, while preserving hashing and OpenAI fallback paths.
- `freeman kg-reindex` now batches re-embedding through `embed_many()` when the provider supports it.

### Validation

- `pytest tests/` -> `86 passed`
- `python scripts/benchmark_faab/run_benchmark.py --dry-run --dataset scripts/benchmark_faab/dataset/cases.jsonl --output-dir runs/faab_universal_dryrun`
- Universal dry-run FAAB means:
  - `MODE_A_FULL`: `t0_mean=0.50`, `t1_mean=1.00`
  - `MODE_B_AMNESIC`: `t0_mean=0.50`, `t1_mean=0.50`
  - `MODE_C_HASH`: `t0_mean=0.50`, `t1_mean=0.75`
  - `MODE_D_LLMONLY`: `t0_mean=0.25`, `t1_mean=1.00`
- Real FAAB benchmark run completed at `runs/faab_real_regime_v1/`.
- Recorded FAAB result means:
  - `MODE_A_FULL`: `t0_mean=0.50`, `t1_mean=0.75`
  - `MODE_B_AMNESIC`: `t0_mean=0.50`, `t1_mean=0.50`
  - `MODE_C_HASH`: `t0_mean=0.50`, `t1_mean=0.50`
  - `MODE_D_LLMONLY`: `t0_mean=0.50`, `t1_mean=1.00`
- Live DeepSeek + Ollama E2E run completed across 3 domains plus repeated social-memory probe.
- Local Ollama smoke tests completed for `nomic-embed-text` and `mxbai-embed-large`.

## v1.1.0

This release brings the repository to a working USIM-AGENT baseline covering the v0.1 and v0.2 technical specifications.

### Added

- `WorldGraph` / `OutcomeRegistry` compatibility layer on top of the existing `WorldState`-based simulator.
- `EvolutionRegistry` and explicit additive operator view.
- Aggregate verifier API, fixed-point result model, and spectral-radius guard.
- Knowledge graph, session log, reconciler, confidence thresholds, persistence, and graph export.
- Agent layer: analysis pipeline, signal ingestion, UCB scheduler, cost governance.
- Interface layer: CLI, minimal REST API, KG export, model override API, simulation diff.
- v0.2 modules: compile validation, ensemble sign consensus, historical fit scoring, uncertainty propagation.
- Full end-to-end integration coverage.
- Repository documentation:
  - architecture
  - API map

### Changed

- `pytest tests/` now works directly in the repo without manual `PYTHONPATH` manipulation.
- Default memory backend is explicitly configured as NetworkX + JSON.
- Packaging metadata now points to `README.md`.

### Validation

- `pytest tests/` -> `44 passed`
