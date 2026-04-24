# Freeman

Freeman is an analytical agent for situations that change over time. You describe a system as actors, resources, causal links, and possible outcomes; Freeman simulates that system, stores the result in memory, and helps you update the picture when new signals arrive.

The point of Freeman is not "chatting about a topic". The point is to keep a structured world model, run repeatable scenario analysis, remember past cases, and give you an audit trail of what changed and why.

Current release: `3.0.0`

Freeman `3.0` closes the operational local-agent loop:

- continuous ingestion into a persistent world model
- universal semantic retrieval over persisted runtime artifacts
- autonomous ontology repair for schema-backed runtimes
- budget/cost governance with persisted runtime ledger
- evidence-grounded `ask` / `query` against the saved daemon state

## Why Use It

Use Freeman when you need an agent that can:

- turn domain knowledge into an explicit model instead of an opaque prompt
- simulate how shocks propagate through a causal system
- keep memory across runs instead of starting from scratch every time
- answer follow-up questions from stored evidence
- let a human override assumptions, rerun the model, and compare results

Typical fits:

- climate, macro, policy, and risk monitoring
- repeated scenario analysis over an evolving system
- research workflows where traceability matters as much as the final answer
- any domain that can be written as variables + causal links + outcomes

## What Freeman Can Do

- Build a world from a structured schema: actors, resources, causal graph, outcomes, and optional policies.
- Run deterministic multi-step simulation and score the most likely outcomes.
- Carry state forward when new events arrive, including shock decay and dynamic recalibration of outcome weights or causal strengths.
- Check the world for invalid, unstable, or contradictory behavior before and during simulation.
- Store analyses in a persistent knowledge graph.
- Use universal semantic retrieval over that graph; ChromaDB is optional acceleration, not a requirement.
- Answer questions against stored memory with `freeman ask`.
- Track forecasts, conflicts, anomalies, and proactive events in the agent layer.
- Export the knowledge graph as HTML, JSON-LD, DOT, or interactive 3D HTML.
- Render graph evolution from ordered runtime snapshots as a standalone timeline viewer.
- Expose Freeman through MCP so external agents can query the daemon as a stateful knowledge service.
- Apply manual parameter or sign overrides, rerun the world, and diff the result.
- Connect external feeds through the separate `freeman-connectors` package.

Important design choice:

- The simulator and core memory are deterministic.
- LLMs are optional and are mainly useful for summarization, embedding-based retrieval, or advanced recalibration flows.
- Source adapters are not bundled into the core runtime; they live in `packages/freeman-connectors/`.

## How It Works

At a high level, Freeman runs this loop:

1. Compile a domain schema into an internal world model.
2. Verify that the model is numerically and causally sane.
3. Simulate the world over time and score outcomes.
4. Write the result, confidence, and context into the knowledge graph.
5. On the next run, reuse memory and optionally update the world with new evidence.

This makes Freeman useful for "stateful analysis": the system can accumulate prior cases instead of treating every question as isolated.

Current operational path:

- `python -m freeman.runtime.stream_runtime` is the single daemon-like runtime entrypoint.
- Domain behavior is supplied by `agent.sources`, `agent.bootstrap`, and the schema or natural-language brief, not by separate runtime modules.
- The consciousness layer is deterministic and replayable; LLMs may translate state, but do not mutate it.
- Runtime cost governance is now part of the operational path, not an external benchmark-only concern.

## Install

Core package:

```bash
pip install .
```

Optional semantic-memory extras:

```bash
pip install ".[semantic]"
```

Optional causal-estimation extras:

```bash
pip install ".[causal]"
```

Development extras:

```bash
pip install ".[dev]"
```

Optional MCP server extra:

```bash
pip install ".[mcp]"
```

Connector package for RSS / HTTP / web-page ingestion:

```bash
pip install ./packages/freeman-connectors
```

Install directly from GitHub:

```bash
pip install git+https://github.com/Theclimateguy/Freeman.git
pip install "git+https://github.com/Theclimateguy/Freeman.git#subdirectory=packages/freeman-connectors"
```

## Quick Start

Initialize a fresh agent workspace:

```bash
freeman init --config-path config.yaml
```

Run a bundled example schema:

```bash
freeman run --config-path config.yaml --schema-path freeman/domain/profiles/gim15.json
```

Inspect the latest stored analysis node:

```bash
freeman query --config-path config.yaml --node-type analysis_run --status archived
```

Export the knowledge graph:

```bash
freeman export-kg html runs/kg.html --config-path config.yaml
freeman export-kg html-3d runs/kg_3d.html --config-path config.yaml
freeman export-kg json-ld runs/kg.jsonld --config-path config.yaml
freeman export-kg dot runs/kg.dot --config-path config.yaml
freeman export-kg-evolution data/runtime/kg_snapshots runs/kg_evolution.html
```

The `html-3d` export is the current high-density graph viewer: force-directed 3D layout, relation/node filters, search, and side-panel inspection. `export-kg-evolution` is the corresponding timeline viewer over ordered KG snapshots.

Generate a deterministic climate seed graph without committing local memory artifacts:

```bash
python scripts/seed_climate_kg.py --write-memory
```

Ask a follow-up question after you have accumulated relevant memory and configured an LLM summarizer:

```bash
freeman ask "What changed most since the previous similar case?" --config-path config.yaml
```

Run direct semantic lookup over graph memory:

```bash
freeman query --config-path config.yaml --text "power outage extreme weather" --limit 5
freeman query --config-path config.climate.clean.yaml --text "heat adaptation migration" --limit 5
```

`freeman ask` and `freeman query --text ...` now use the same retrieval stack:

- embedding retrieval through the configured adapter when available
- ChromaDB nearest-neighbor lookup when the optional vector store is enabled
- deterministic lexical-semantic ranking when no vector store is configured
- strict no-match semantics instead of falling back to unrelated high-confidence nodes

They now answer against persisted runtime context, not only raw KG nodes:

- semantic KG matches
- saved forecasts
- current causal edges
- current world snapshot
- self-model and anomaly nodes already stored in the KG

This lets a clean local instance answer semantically meaningful questions against a trained graph even before `kg-reindex` or Chroma setup.

`freeman ask` also returns live budget telemetry sourced from the runtime ledger:

- `spent_usd`
- `remaining_usd`
- `allowed_count`
- `blocked_count`
- `by_task_type`

Human-in-the-loop workflow:

```bash
freeman override-param world.json resources.x.value 12.0 --output-path world_override.json --config-path config.yaml
freeman override-sign world.json x->y + --output-path world_override.json --config-path config.yaml
freeman rerun-domain world_override.json --max-steps 10 --output-path rerun.json --config-path config.yaml
freeman diff-domain world.json rerun.json --output-path diff.json --config-path config.yaml
```

Long local stream run (generic daemon-like runtime):

```bash
python -m freeman.runtime.stream_runtime \
  --config-path config.yaml \
  --schema-path freeman/domain/profiles/gim15.json \
  --hours 8 \
  --poll-seconds 600 \
  --analysis-interval-seconds 1.0 \
  --resume \
  --model auto
```

Domain-agnostic bootstrap from a natural-language brief:

```bash
python -m freeman.runtime.stream_runtime \
  --config-path config.yaml \
  --bootstrap-mode llm_synthesize \
  --domain-brief-path domain_brief.md \
  --hours 8 \
  --poll-seconds 600 \
  --analysis-interval-seconds 1.0 \
  --resume \
  --model auto
```

The same daemon runtime can be tested on climate RSS by swapping only the config:

```bash
python -m freeman.runtime.stream_runtime \
  --config-path config.climate.yaml \
  --hours 8 \
  --poll-seconds 600 \
  --analysis-interval-seconds 1.0 \
  --resume \
  --model auto
```

The daemon runtime reads `agent.sources` from config, polls them on the configured interval, and processes a persistent pending queue between polls (plus ex-post forecast verification on each world update). Domain behavior now lives in config and schema/bootstrap inputs, not in separate runtime entrypoints.

If `runtime.kg_snapshots.enabled: true`, the same runtime also writes ordered KG snapshots under the configured snapshot path. Those snapshots can be rendered later with:

```bash
python -m freeman.interface.cli export-kg-evolution data/runtime_climate/kg_snapshots runs/kg_climate_evolution.html
```

Repository CI now runs `pytest -q` plus release builds on every push to `main` and on pull requests through `.github/workflows/tests.yml`.

It supports two bootstrap modes:

- `llm_synthesize`: the primary path; use the built-in Freeman orchestrator to build the initial state vector from a natural-language brief through a two-phase ETL bootstrap before the stream loop starts
- `schema_path`: the secondary path; start from an explicit schema such as `freeman/domain/profiles/gim15.json`

`llm_synthesize` now runs `skeleton -> edges -> verifier` instead of one monolithic synthesis call. `bootstrap_package.json` persists `bootstrap_attempts` with `etl_phase` markers (`skeleton`, `edges`, `sign_repair`) plus verifier errors; level2 sign failures use targeted edge repair, while compile/level1/level0 failures still use full-package repair. The default clean config is local-model-first (`ollama` + `qwen2.5-coder:14b`) with `fallback_schema_path` only as a secondary recovery path.

The runtime also carries a monotonic `runtime_step`, separate from simulator `world.t`. Forecast deadlines and verification use `runtime_step`, so fallback updates do not starve ex-post verification.

Fallback updates now also preserve monotonic simulator time: runtime may retry an update from a safe schema/base world, but it preserves the live `world.t` / `runtime_step` clocks and rejects any fallback result that would rewind `world.t`.

For long local domain runs, keep a dedicated `runtime.runtime_path` (and matching `event_log_path` / KG path) per instance. Reusing the same runtime directory with `--resume` intentionally appends to the same longitudinal trace.

Longitudinal self-verification now happens on two levels:

- scalar outcome verification against the realized posterior at the verification horizon
- causal-path verification against KG trajectory edges exported from the simulation

The current causal path uses three edge relations:

- `causes`
- `propagates_to`
- `threshold_exceeded`

Saved runtime state is also queryable without starting the daemon loop:

```bash
python -m freeman.runtime.stream_runtime --config-path config.yaml --query forecasts
python -m freeman.runtime.stream_runtime --config-path config.yaml --query explain --forecast-id <forecast_id>
python -m freeman.runtime.stream_runtime --config-path config.yaml --query anomalies
python -m freeman.runtime.stream_runtime --config-path config.yaml --query causal --limit 20
python -m freeman.runtime.stream_runtime --config-path config.yaml --query semantic --text "greenhouse warming" --limit 10
python -m freeman.runtime.stream_runtime --config-path config.yaml --query answer --text "What is the current warming risk?" --limit 10
```

If you want other agents to call Freeman as a stateful knowledge daemon, run the MCP wrapper:

```bash
freeman-mcp --transport stdio
```

The MCP server exposes the in-memory simulation tools plus persistent runtime query tools:

- `freeman_get_runtime_summary`
- `freeman_query_forecasts`
- `freeman_explain_forecast`
- `freeman_query_anomalies`
- `freeman_query_causal_edges`
- `freeman_trace_relation_learning`
- `freeman_query_runtime_context`
- `freeman_answer_query`

`freeman_trace_relation_learning` reads recent KG snapshots and is the right tool when an external agent asks what Freeman learned about a relation `X -> Y` over the last `N` runtime steps.

A minimal local MCP launch looks like:

```bash
freeman-mcp --transport stdio
```

For remote attachment during development, the same server can run over HTTP transports:

```bash
freeman-mcp --transport streamable-http --host 127.0.0.1 --port 8000
```

Stream ingestion is two-phase:

- phase 1 hard filter from config: `agent.stream_keywords` + `agent.stream_filter.min_keyword_matches` + `agent.stream_filter.min_relevance_score`
- phase 2 agent-side relevance filter after self-calibration begins (`self_observation` exists), with soft-reject trace events and `stream_relevance` tracked as a `self_capability`

The phase-2 threshold is domain-sensitive. For broad climate/news streams, `agent.stream_filter.agent_min_relevance_score` often needs to stay near the empirical relevance-score range, otherwise the runtime can starve itself after self-calibration.

Ontology-blind low-relevance signals are not dropped silently anymore:

- if both ontology overlap and hypothesis overlap are zero, the signal is written to KG as `anomaly_candidate`
- `ConsciousnessEngine` reviews repeated anomaly clusters and escalates them into `identity_trait` nodes with `trait_key=ontology_gap`
- once enough unhandled ontology gaps accumulate, runtime triggers ontology repair automatically
- for `schema_path` / fallback runtimes, repair writes an overlay `bootstrap_package.json`, auto-applies inferred causal edges to the live world and base template, and persists an audit queue in `ontology_repair_queue.jsonl`
- for `llm_synthesize` runtimes, repair appends topics to the current domain brief, writes `domain_brief_history.jsonl`, and re-runs verifier-guided bootstrap while preserving monotonic `runtime_step`

Runtime budget governance is now also first-class:

- `signal_processing`, `ontology_repair`, and `answer_generation` all pass through the same cost policy
- decisions are persisted into `runtime/cost_ledger.jsonl`
- hard limits can downgrade `DEEP_DIVE -> ANALYZE -> WATCH` or stop execution
- `freeman status` surfaces the live ledger summary from disk

The reconciler is also configurable through `memory.reconciler`:

- `merge_threshold`: cosine-similarity threshold for semantic merge instead of eager claim splitting
- `compaction_interval`: periodic `__split_*` compaction cadence

`checkpoint.json` now includes `runtime_metadata.kg_health`, including split-node count, average live-node degree, and the last compaction step.

The consciousness layer stays deterministic. It does not read narrative text back into state. Instead it projects:

- `goal_state` from schema semantics and world polarity
- `active_hypothesis` from the current posterior over outcomes
- `identity_trait` and `self_capability` from verified forecast error (`self_observation`)

This keeps the separation strict: `ConsciousState -> LLM -> external world`, never the reverse.

## Main Commands

- `freeman init`: create a config file and empty storage for the knowledge graph and session logs.
- `freeman run`: compile a schema and run the full analysis pipeline.
- `freeman ask`: retrieve relevant runtime evidence and, when an LLM is configured, answer a question from stored KG/forecast/causal/world context.
- `freeman status`: inspect configured storage paths, current active-memory counts, and persisted budget ledger state.
- `freeman query`: query graph nodes directly; `--text` uses runtime semantic retrieval over KG + forecasts + causal edges + world state.
- `freeman export-kg`: export the graph for inspection or downstream tooling.
- `freeman reconcile`: merge one saved session log back into the long-term graph.
- `freeman kg-archive`: archive low-confidence or obsolete graph nodes.
- `freeman kg-reindex`: rebuild embeddings and sync them into ChromaDB for faster semantic retrieval on large graphs.
- `freeman override-param`, `override-sign`, `rerun-domain`, `diff-domain`: edit assumptions and compare before/after simulations.

## Advanced Use Cases

If you are integrating Freeman into Python code rather than only using the CLI, the main entry points are:

- `freeman.agent.AnalysisPipeline`: compile -> simulate -> verify -> write to memory.
- `freeman.agent.SignalIngestionEngine`: normalize incoming signals and decide whether they deserve attention.
- `freeman.agent.ParameterEstimator`: ask an LLM to adjust an existing world when new evidence changes the regime.
- `freeman.memory.KnowledgeGraph`: persistent graph memory with universal semantic retrieval, optional Chroma acceleration, and strict no-match behavior.
- `freeman.runtime.queryengine`: shared runtime semantic retrieval and answer synthesis over persisted runtime artifacts.
- `freeman.agent.costmodel`: persisted budget ledger plus shared downgrade/stop policy across runtime tasks.

## Operational Readiness

For `3.0`, the remaining issues are no longer architecture blockers for local deployment/training. The loop is closed end-to-end:

- signal ingestion persists state and resumes cleanly
- the world model can expand from anomaly pressure without human review for schema-backed runtimes
- runtime query/answer uses the same semantic retrieval layer as the CLI
- budget/cost accounting is visible and enforced

What remains is mostly tuning, not missing infrastructure:

- budget coefficients may need calibration per provider or hardware profile
- semantic thresholds may need per-domain tuning on very noisy streams
- autonomous ontology induction is safest when new relations stay within an existing schema vocabulary
- `freeman.verifier.Verifier`: structural and causal checks for world consistency.

This advanced path is the right place if you want an agent that reacts to incoming news/data streams, updates its internal state, and keeps learning from prior forecast errors.

## Repository Layout

- `freeman/core/`: world state, transition logic, scoring, uncertainty, compile validation
- `freeman/verifier/`: invariant checks, structural checks, sign consistency, fixed-point repair
- `freeman/memory/`: knowledge graph, vector store, reconciliation, session logs
- `freeman/agent/`: analysis pipeline, signal ingestion, attention, forecasts, proactive events, recalibration
- `freeman/interface/`: CLI, minimal API, export, override, diff helpers
- `freeman/domain/`: schema compiler and bundled profiles
- `freeman/llm/`: optional LLM and embedding adapters
- `packages/freeman-connectors/`: external ingestion adapters kept outside the core package
- `scripts/`: end-to-end and benchmark runners
- `tests/`: regression, integration, and behavior tests

## Documentation

Actual operational docs:

- Architecture: [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)
- API map: [docs/API_MAP.md](docs/API_MAP.md)
- Consciousness design and invariants: [docs/CONSCIOUSNESS_ARCHITECTURE.md](docs/CONSCIOUSNESS_ARCHITECTURE.md)

Historical / legacy artifacts:

- Benchmark notes: [docs/FAAB.md](docs/FAAB.md)
- Legacy live end-to-end evaluation artifact: [docs/REAL_LLM_E2E.md](docs/REAL_LLM_E2E.md)
- Historical implementation log: [PROGRESS.md](PROGRESS.md)

Example assets:

- Climate example config: [config.climate.yaml](config.climate.yaml)
- Climate example brief for `llm_synthesize`: [examples/domain_brief_climate_news.md](examples/domain_brief_climate_news.md)

## Notes

- Default long-term memory is a JSON-backed NetworkX graph.
- Semantic retrieval is optional and uses ChromaDB when installed with the `semantic` extra.
- `freeman init` creates an empty agent workspace; it does not preload domain knowledge.
- The core package is source-agnostic. If you need RSS, HTTP/JSON, HTML page, or arXiv ingestion, use `freeman-connectors`.
