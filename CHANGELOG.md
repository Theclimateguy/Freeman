# Changelog

## Unreleased

### Added

### Changed

- No unreleased changes.

## v3.3.1 - 2026-06-01

Release focused on promoting the stabilized hive-mind runtime to `main` and
making regional geospatial structure active in compiled simulations.

### Added

- Role-scoped hive runtime entrypoint: `freeman-hive` /
  `python -m freeman.runtime.hive_runtime`.
- Hive role contracts for `ingestor`, `repairer`, `planner`, `narrator`, and
  `verifier`, with per-role trail scopes and optional LLM policies.
- Cooperative KG-node locking with an optional Redis backend through the
  `redis` extra.
- Runtime structural protocols in `freeman.runtime.contracts`.
- Regional geospatial adapter in `freeman.realworld.spatial_adapter` using
  optional GeoPandas/Shapely dependencies through the `geo` extra.
- Compile-time spatial materialization from `world.metadata["spatial"]` into
  actor `Relation(..., relation_type="spatial_neighbor")` records.
- Dedicated regional analytics documentation in `docs/GEO_ANALYTICS.md`.

### Changed

- `main` now contains the release-grade hive-mind runtime and docs from the
  3.2 line, without generated experiment artifacts from that branch.
- `DomainCompiler.compile()` initializes spatial relations automatically; the
  real-world manifold path performs the same initialization idempotently before
  simulation.
- README, architecture docs, release notes, and packaging metadata now describe
  `3.3.1` as the active release.
- Core package version bumped to `3.3.1`; `freeman-connectors` now targets
  `freeman>=3.3.1,<4.0.0`.

### Fixed

- Large regional schemas now emit a warning when spatial actor-pair
  materialization is likely to become expensive.

### Validation

- `PYTHONPATH="$PWD/packages/freeman-connectors:$PYTHONPATH" python -m pytest -q` -> `287 passed`
- `python -m compileall -q freeman packages/freeman-connectors/freeman_connectors` -> passed
- Core and connector sdist/wheel builds completed for `3.3.1`; see `docs/RELEASE_3_3_1.md`.

## v3.2.2 - 2026-05-10

Hive-mind bugfix prerelease focused on Redis-backed horizontal scaling.

### Fixed

- Redis lock release now checks lock ownership atomically, preventing one hive
  worker from unlocking another worker's active KG-node lock.

### Validation

- `pytest tests/test_hive_mind.py` -> `19 passed`

## v3.2.1 - 2026-05-10

Hive-mind prerelease focused on making the role stack operational rather than
conceptual.

### Added

- Executable hive runtime dispatcher over role-scoped KG frontiers.
- Memory and Redis lock backends.
- Optional role LLM policy and OpenAI-compatible structured client support.
- Hive runtime documentation and role docs under `docs/`.

### Validation

- `pytest tests/test_hive_mind.py tests/test_openai_client.py tests/test_runtime.py` -> `48 passed`

## v3.2.0 - 2026-05-10

Official 3.2 release cut from the stabilized `hive_mind` branch.

### Added

- Explicit layer contracts around world, KG, and consciousness state.
- Signal conflict preservation instead of suppressing contradictory evidence.
- Flat graph export and interactive ask/what-if surfaces over persisted runtime
  state.
- Release CI for full pytest, CLI smoke tests, and build artifacts.

### Changed

- Ontology repair and runtime query paths were tightened around explicit
  contracts and persisted evidence.

## v3.1.2 - 2026-05-01

Patch release focused on forecast-verification clock semantics.

### Changed

- Forecast horizons now use the simulator/domain clock: `Forecast.deadline_step = created_step + horizon_steps`, where `created_step` is `world.t`.
- Runtime forecast verification now checks due forecasts against `current_world.t`; `runtime_step` remains the agent-level event/audit clock for queues, snapshots, fallback continuity, and provenance.
- Forecast query/explanation fields now report `created_at_step` and `due_at_step` on the same domain-step axis.
- README, API map, and full architecture documentation now describe the domain-step forecast horizon contract.
- Core package version bumped to `3.1.2`; `freeman-connectors` now targets `freeman>=3.1.2,<4.0.0`.

### Fixed

- Forecasts no longer expire early when the agent runtime advances faster than the simulated domain, when one agent tick includes multiple runtime events, or when a domain is loaded with non-zero `world.t`.

### Validation

- `pytest -q` -> `239 passed`

## v3.1.1 - 2026-04-25

Patch release focused on release hygiene, explicit ontology-ingestion provenance, and ETL hardening.

### Added

- `bootstrap_contract` persisted in `bootstrap_package.json` so every runtime bootstrap records:
  - strategy id
  - actual materialization path (`schema_seed`, `etl_from_brief`, `fallback_schema_seed`)
  - input requirements
  - recommended use cases
  - explicit limitations
- New ontology-ingestion reference document: `docs/ONTOLOGY_INGESTION.md`.
- Transparent live-news benchmark harness: `scripts/run_random_news_etl_benchmark.py`.
- Dedicated bootstrap-contract test coverage in `tests/test_bootstrap_contracts.py`.

### Changed

- Freeman now treats ontology creation as a catalog of explicit ingestion strategies rather than a single ETL path with implicit fallbacks.
- `.gitignore` now keeps runtime artifacts and generated data ignored, while allowing the tracked benchmark runner and release changelog to remain versioned.
- Core package version bumped to `3.1.1`; `freeman-connectors` now targets `freeman>=3.1.1,<4.0.0`.

### Fixed

- ETL resource normalization now coerces `evolution_params` to the active operator contract, preventing invalid `logistic` resources from carrying linear-only parameters into compiled worlds.
- Runtime bootstrap tests now assert ontology-ingestion provenance for both successful ETL and fallback-backed paths.

### Validation

- `pytest -q` -> `238 passed`

## v3.1.0 - 2026-04-25

Minor release focused on replacing monolithic LLM bootstrap with the new two-phase ETL path and formalizing repository licensing.

### Added

- Two-phase ETL bootstrap in `FreemanOrchestrator`:
  - `skeleton` phase extracts only actors, resources, and outcomes
  - `edges` phase adds `causal_dag`, `actor_update_rules`, policies, and assumptions against a validated node set
- Surgical level-2 sign repair via `repair_sign_edges()`, reducing full-package churn when only a few causal signs fail.
- `bootstrap_attempts[].etl_phase` persisted in runtime bootstrap artifacts for `skeleton`, `edges`, and `sign_repair` diagnostics.
- Repository license files under Apache License 2.0 for both the root package and `freeman-connectors`.

### Changed

- `llm_synthesize` is now the primary ETL bootstrap path for runtime state-vector assembly; interactive single-call synthesis remains available for backward compatibility.
- Resource-target causal edges now receive deterministic bounded coupling materialization during ETL (`weight_source=\"etl_deterministic\"`), leaving numeric estimation to runtime calibration.
- Runtime cost estimation for bootstrap/repair now accounts for the two-call ETL path rather than a single monolithic synthesis call.
- Architecture docs, API map, README, and the 3D architecture visualization now describe the ETL bootstrap instead of the old single-shot synthesis flow.
- Core package version bumped to `3.1.0`; `freeman-connectors` now targets `freeman>=3.1.0,<4.0.0`.

### Validation

- `pytest -q` -> `234 passed`

## v2.0.2 - 2026-04-13

Patch release focused on release hygiene, CI, and explicit connector coverage.

### Added

- GitHub Actions test workflow in `.github/workflows/tests.yml` running `pytest -q` and release builds on push / pull request.
- Dedicated connector test slice in `tests/test_connectors.py` covering factory construction and the `rss`, `http_json`, and `web` source adapters.

### Changed

- Moved the climate example brief out of the repository root into `examples/domain_brief_climate_news.md`.
- Updated `config.climate.yaml` and user-facing docs to point to the new `examples/` location.

### Validation

- `pytest -q` -> `204 passed`

## v2.0.1 - 2026-04-13

Patch release focused on anomaly handling, ontology self-repair, and read-only runtime queries over saved state.

### Added

- Ontology-blind soft-filtered signals are now preserved as `anomaly_candidate` nodes in the KG instead of being silently discarded.
- `ConsciousnessEngine` now reviews accumulated anomaly candidates, escalates repeated clusters into `identity_trait` signals with `trait_key=ontology_gap`, and closes stale singletons as `noise`.
- Self-healing bootstrap loop: repeated `ontology_gap` traits now emit `ontology_repair_request`, append gap topics to `domain_brief_history.jsonl`, and trigger a verifier-guided re-bootstrap while preserving monotonic `runtime_step`.
- Read-only runtime query mode in `freeman.runtime.stream_runtime`:
  - `--query forecasts`
  - `--query explain --forecast-id <id>`
  - `--query anomalies`
  - `--query causal --limit <n>`
- Forecast explanation API with `ForecastExplanation`, `CausalStep`, `AnalysisPipeline.list_forecasts()`, `AnalysisPipeline.explain_forecast()`, and `KnowledgeGraph.explain_causal_path()`.

- Agent-side stream filtering now splits low-relevance signals into `FILTERED_OUT` versus `ANOMALY_CANDIDATE` based on ontology and hypothesis overlap.

### Validation

- `pytest -q` -> `200 passed`

## v2.0.0 - 2026-04-12

Second major release of Freeman. This version promotes the repository from a deterministic world-simulation toolkit to a domain-agnostic daemon runtime with deterministic consciousness, persistent stream learning, and trajectory-level self-verification.

### Added

- Deterministic consciousness layer with `ConsciousState`, `SelfModelGraph`, `ConsciousnessEngine`, `IdleScheduler`, read-only `IdentityNarrator`, and `ExplanationRenderer`.
- Generic domain-agnostic daemon runtime in `freeman.runtime.stream_runtime` with checkpoint/resume, persistent pending queue, durable event log, and synchronous idle deliberation.
- Verifier-guided `llm_synthesize` bootstrap with persisted `bootstrap_attempts` and explicit fallback artifact retention.
- Causal-path export from simulation runs into KG edges (`causes`, `propagates_to`, `threshold_exceeded`) and forecast-level `causal_path` storage.
- Causal trajectory verification in `Reconciler.verify_causal_path()` with `self_observation` grounded in both scalar error and path confirmation/refutation.

### Changed

- Long-running local execution is now documented around a single runtime entrypoint: `python -m freeman.runtime.stream_runtime`.
- `runtime_step` is now the monotonic verification clock for forecasts, separate from simulator `world.t`.
- Runtime stream filtering is two-phase: config hard filter before queueing and agent-side soft reject after self-calibration.
- `stream_runtime.py` now uses explicit runtime decomposition around `RuntimeContext` instead of a large closure-driven `main()`.

### Documentation

- Refreshed `README.md`, `docs/ARCHITECTURE.md`, `docs/API_MAP.md`, and `docs/CONSCIOUSNESS_ARCHITECTURE.md` to match the current runtime and consciousness implementation.
- Marked research-only or historical documents explicitly as legacy artifacts.

### Validation

- `pytest -q` -> `190 passed`

## v1.0.0 - 2026-04-11

First public clean release of Freeman. This version freezes the repository in a stable v1 state around four core capabilities:

- domain-agnostic world simulation with explicit operators and bounded verification
- persistent epistemic memory, reconciliation, and signal ingestion
- attention and budget-aware agent orchestration
- bounded counterfactual policy evaluation that reuses simulation prep to keep compute practical

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

- `python -m pytest tests/ -q` -> `143 passed`
- `python -m build --outdir /tmp/freeman-release-dist` -> built `freeman-1.0.0.tar.gz` and `freeman-1.0.0-py3-none-any.whl`
- `python -m build packages/freeman-connectors --outdir /tmp/freeman-connectors-dist` -> built `freeman_connectors-1.0.0.tar.gz` and `freeman_connectors-1.0.0-py3-none-any.whl`
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

## Legacy Internal Baseline (pre-v1)

Internal milestone recorded before formal public release tags existed.

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
