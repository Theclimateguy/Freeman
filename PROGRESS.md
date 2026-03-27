# USIM-AGENT Progress

## Semantic Memory Layer (v1.2.0)

- Added semantic retrieval over the knowledge graph with optional ChromaDB persistence in `freeman/memory/vectorstore.py`.
- Extended `KGNode` and `KnowledgeGraph` to persist embeddings, lazy-reembed legacy JSON nodes, auto-sync semantic state on add/update/archive, and expose `semantic_query()` with top-K + 1-hop neighbor expansion.
- Updated the reconciler to force re-embedding on merged nodes when semantic indexing is active, so vector state stays aligned with merged claims.
- Updated the analysis pipeline to use retrieval-bounded context selection instead of exposing the full active KG to downstream LLM-facing paths.
- Extended the LLM layer with an embedding interface, a minimal OpenAI embedding client, and an offline deterministic embedding stub used for tests and local reindexing when no API key is available.
- Added `freeman kg-reindex`, config support for `memory.vector_store`, and embedding token tracking in the cost model.
- Added semantic-memory tests in `tests/test_vectorstore.py` and expanded reconciler/cost tests to cover archive/delete, merge/re-embed, lazy embedding, CLI reindex, and retrieval fallback behavior.

### Deviations from spec

- When `OPENAI_API_KEY` is not present, CLI reindexing falls back to a deterministic stub embedding adapter so offline validation still works; real semantic quality requires a real embedding backend.
- ChromaDB remains optional at runtime and is only constructed when `memory.vector_store.enabled: true`.

### Validation

- `pytest tests/` -> 54 passed
- `freeman --config-path <temp-config> kg-reindex --use-stub-embeddings` -> created `chroma_db/` and indexed 1 node

## Phase 1

- Completed the core API foundation in `freeman/core/`.
- Added `WorldGraph` and `OutcomeRegistry` in `freeman/core/world.py` while keeping `WorldState` as a backward-compatible alias used by the existing codebase.
- Added a formal `EvolutionRegistry` in `freeman/core/evolution.py` and exposed the operator increment `delta` so the transition layer now has an explicit `F(D, S(t), t)` view without changing legacy step semantics.
- Split scorer internals into `raw_outcome_scores` and `softmax_distribution`, keeping `score_outcomes` as the public softmax scorer.
- Added Phase 1 compatibility tests in `tests/test_core_phase1.py`.
- Fixed the test bootstrap so the required command `pytest tests/` works in this repo without setting `PYTHONPATH` manually.

### Deviations from spec

- The repo already used `WorldState` pervasively, so Phase 1 preserves that interface and implements `WorldGraph` as the spec-facing canonical class with `WorldState` aliasing it for compatibility.
- Existing linear/logistic/stock-flow operator semantics were preserved to avoid regressions; the spec-level additive operator form is now exposed through `EvolutionOperator.delta`.

### Validation

- `pytest tests/` -> 26 passed

## Phase 2

- Added `freeman/verifier/verifier.py` with `Verifier` and `VerifierConfig` as the aggregate API over Levels 0, 1, and 2.
- Added `freeman/verifier/fixedpoint.py` with `FixedPointResult`, iterative correction history, and an explicit spectral-radius guard.
- Kept `freeman/verifier/fixed_point.py` as a backward-compatible wrapper so existing imports and `GameRunner` continue to work.
- Added new Phase 2 tests in `tests/test_verifier.py` and `tests/test_fixedpoint.py`.
- Reworked `freeman/verifier/__init__.py` to use lazy exports and avoid circular imports during package initialization.

### Deviations from spec

- Existing `level0.py`, `level1.py`, and `level2.py` remain the low-level implementation modules; `verifier.py` is the new spec-facing aggregation layer on top of them.
- Level 1 sign-consistency is implemented as a precheck that reuses the existing local DAG sign test, while Level 2 adds bounded correction iterations and the spectral-radius guard.

### Validation

- `pytest tests/` -> 31 passed

## Phase 3

- Added `freeman/memory/knowledgegraph.py` with `KGNode`, `KGEdge`, `KnowledgeGraph`, NetworkX backend, JSON persistence, query, split, archive, and HTML/JSON/DOT export.
- Added `freeman/memory/sessionlog.py` with `SessionLog`, `TaskRecord`, `AttentionStep`, and `KGDelta`.
- Added `freeman/memory/reconciler.py` with beta-style confidence updates, status classification, merge logic, conflict-driven node splitting, and archival below threshold.
- Added `freeman/memory/__init__.py` exports, extended `config.yaml` with `memory.json_path`, and declared `networkx`/`PyYAML` in `pyproject.toml`.
- Added `tests/test_reconciler.py` covering archival, conflict resolution, persistence, and export.

### Deviations from spec

- The reconciler implements a pragmatic claim-key based conflict detector because no higher-level ontology or NLI layer exists in the current repo yet.
- JSON persistence is immediate via the in-repo configured path; no external graph backend is used, per the environment constraint.

### Validation

- `pytest tests/` -> 33 passed

## Phase 4

- Added `freeman/agent/domainregistry.py` with `DomainTemplate`, template registry, and wrappers around shared-resource multi-domain composition.
- Added `freeman/agent/analysispipeline.py` implementing compile -> verify -> simulate -> score -> KG update -> reconcile flow.
- Added `freeman/agent/signalingestion.py` with normalized signal models, Mahalanobis anomaly scoring, heuristic/LLM shock classification, and WATCH/ANALYZE/DEEP_DIVE trigger logic.
- Added `freeman/agent/attentionscheduler.py` with finite-budget UCB allocation and the `PENDING -> ACTIVE -> SUSPENDED -> COMPLETED -> ARCHIVED` task state machine.
- Added `freeman/agent/__init__.py` exports and `tests/test_attention.py`.

### Deviations from spec

- Tavily/RSS integrations are implemented as normalized source adapters over already fetched records because this environment has no guaranteed external signal service wiring yet.
- The attention interest score currently uses an additive `(expected_information_gain + anomaly + semantic_gap + confidence_gap) / cost` form, with UCB selection exactly following the spec over that interest term.

### Validation

- `pytest tests/` -> 35 passed

## Phase 5

- Added `freeman/interface/cli.py` with `run`, `query`, `export-kg`, `status`, `reconcile`, and `kg-archive` commands.
- Added `freeman/interface/api.py` with minimal REST handlers for `GET /status` and `POST /query`.
- Added `freeman/interface/kgexport.py` with HTML, DOT, and JSON-LD export wrappers.
- Added `freeman/interface/__init__.py` exports and validated the CLI `status` command directly.

### Deviations from spec

- The REST interface is implemented on the Python standard library HTTP server instead of a full ASGI framework to avoid unnecessary dependencies in the current repo.
- Query/status endpoints are currently KG-centered read APIs, which matches the v0.1 read-only interface requirement and leaves override/edit flows for v0.2.

### Validation

- `python -m freeman.interface.cli status` -> OK
- `pytest tests/` -> 35 passed

## Phase 6

- Added `freeman/core/compilevalidator.py` with `CompileCandidate`, `CompileValidationReport`, `HistoricalFitScore`, backtesting, ensemble validation, and sign voting consensus.
- Added `freeman/core/uncertainty.py` with Monte Carlo sampling over parameter distributions and confidence-from-variance reporting.
- Added `freeman/agent/costmodel.py` with explicit task cost estimation, budget policy, downgrade logic, and stop reasons.
- Added `freeman/interface/modeloverride.py` with parameter and edge overrides, rerun support, immutable machine baseline preservation, and audit logging.
- Added `freeman/interface/simulationdiff.py` with structured diff export and override history.
- Added tests: `tests/test_compilevalidator.py`, `tests/test_uncertainty.py`, `tests/test_costmodel.py`, `tests/test_humanoverride.py`.

### Deviations from spec

- Compile validation uses a deterministic callable-based ensemble interface rather than binding directly to one provider, so it stays compatible with the existing LLM adapter abstraction.
- Human override endpoints are implemented as a reusable API class and diff utility rather than being wired into the stdlib HTTP server routes yet; the contracts are in place and tested.

### Validation

- `pytest tests/` -> 42 passed

## Phase 7

- Added `tests/test_integration.py` covering a 30-step pipeline run, reconciliation, KG export, and end-to-end invariant checks.
- Completed v0.2 interface wiring for override/diff CLI commands and REST routes:
  - `override-param`
  - `override-sign`
  - `rerun-domain`
  - `diff-domain`
  - `PATCH /domain/{id}/params`
  - `PATCH /domain/{id}/edges/{edge_id}`
  - `POST /domain/{id}/rerun`
  - `GET /domain/{id}/diff`
- Added full repository documentation in `README.md`, `docs/ARCHITECTURE.md`, `docs/API_MAP.md`, and `CHANGELOG.md`.

### Validation

- `pytest tests/test_integration.py` -> 1 passed
- `pytest tests/` -> 44 passed
