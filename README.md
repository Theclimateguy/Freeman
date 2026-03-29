# Freeman

Freeman is a domain-agnostic simulation and agent framework for structured reasoning over actors, resources, causal graphs, uncertainty, semantic memory, forecast verification, and proactive agent behavior. The current codebase implements the USIM-AGENT roadmap through v0.2, plus semantic KG retrieval, obligation-driven scheduling, forecast tracking, self-model updates, replay-based behavioral tests, a stateful regime-shift simulator update, and a universal dynamic `ParameterVector` layer for T1 recalibration.

## What Is Implemented

- Core simulator: `WorldGraph` / `WorldState`, transition operators, outcome scoring, multi-domain shared-resource worlds.
- Stateful regime updates: decayed shock accumulation via `WorldGraph.apply_shocks(..., time_decay=...)` plus conditional outcome multipliers through `Outcome.regime_shifts`.
- Universal dynamic calibration: `ParameterVector` with `outcome_modifiers`, `shock_decay`, `edge_weight_deltas`, and LLM-generated rationale.
- Verifier: Level 0 invariants, Level 1 structural checks, Level 2 sign consistency, fixed-point repair, spectral-radius guard.
- Memory: NetworkX + JSON knowledge graph, ChromaDB semantic retrieval, session logs, confidence reconciliation, self-model nodes, export to HTML / JSON-LD / DOT.
- Agent layer: domain template registry, analysis pipeline, signal ingestion, signal memory, UCB attention scheduler, obligation queue, forecast registry, proactive emitter, formal cost model.
- T1 recalibration path: `ParameterEstimator` + `AnalysisPipeline.update()` for ontology-preserving world updates.
- Interface layer: CLI commands, minimal REST API, KG export, human override API, simulation diff.
- v0.2 extensions: compile validation, historical fit scoring, ensemble sign consensus, Monte Carlo uncertainty, cost governance, override audit trail.

## Install

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

Separate connector package:

```bash
pip install ./packages/freeman-connectors
pip install "git+https://github.com/Theclimateguy/Freeman.git#subdirectory=packages/freeman-connectors"
```

Development extras:

```bash
pip install ".[dev]"
```

GitHub install:

```bash
pip install git+https://github.com/Theclimateguy/Freeman.git
```

## Quick Start

Initialize an empty agent in the current directory:

```bash
freeman init
```

Or create the config manually from the bundled template:

```bash
cp config.yaml.example config.yaml
```

Check the empty knowledge graph status:

```bash
freeman status --config-path config.yaml
```

Ask a question against the accumulated KG:

```bash
freeman ask "What is the current strongest belief conflict?" --config-path config.yaml
```

Run a config-driven bootstrap cycle:

```bash
freeman run --config-path config.yaml
```

Run a one-shot bootstrap from a schema configured in `agent.bootstrap.schema_path`, or override it explicitly:

```bash
freeman run --config-path config.yaml --schema-path path/to/schema.json
```

Export the knowledge graph:

```bash
freeman export-kg html runs/kg.html
freeman export-kg json-ld runs/kg.jsonld
freeman export-kg dot runs/kg.dot
```

Apply human overrides:

```bash
freeman override-param world.json resources.x.value 12.0 --output-path world_override.json
freeman override-sign world.json x->y + --output-path world_override.json
freeman rerun-domain world_override.json --max-steps 10 --output-path rerun.json
freeman diff-domain world.json rerun.json --output-path diff.json
```

## Main Entry Points

- Core Python API:
- `freeman.core`
  - includes stateful shock decay, regime-shift scoring, and `ParameterVector`
- `freeman.verifier`
- `freeman.memory`
- `freeman.agent`
- `freeman.interface`
- CLI:
  - `freeman init`
  - `freeman run --config-path config.yaml`
  - `freeman ask "..."`
  - `freeman status`
- Minimal REST server:
  - `freeman.interface.api.run_server()`
- LLM orchestration:
  - `freeman.llm.orchestrator.DeepSeekFreemanOrchestrator`
  - `freeman.llm.OllamaEmbeddingClient`

## Documentation

- Architecture and workflows: [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)
- API map: [docs/API_MAP.md](docs/API_MAP.md)
- Universal parameter-vector update path: [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)
- Live E2E evaluation report: [docs/REAL_LLM_E2E.md](docs/REAL_LLM_E2E.md)

## Repository Layout

- `freeman/core/` — simulator state, evolution operators, transition logic, scoring, compile validation, uncertainty.
- `freeman/verifier/` — verifier levels, aggregate verifier, fixed-point correction.
- `freeman/memory/` — knowledge graph, semantic vector store, reconciliation, session logs.
- `freeman/agent/` — analysis pipeline, scheduling, signal ingestion, forecast tracking, proactive emission, cost governance.
- `freeman/interface/` — CLI, REST API, export, override and diff helpers.
- `freeman/domain/` — schema compiler and bundled profiles.
- `freeman/llm/` — provider-facing LLM orchestration and repair loop.

## Notes

- Default long-term memory backend is NetworkX + JSON via `config.yaml -> memory.json_path`.
- Semantic retrieval is optional and uses ChromaDB when installed with the `semantic` extra.
- The default packaged agent starts empty: `freeman init` creates a blank KG plus storage directories, not a prefilled memory.
- The core package stays source-agnostic. Live ingestion adapters live in the separate `freeman-connectors` package, not in the core runtime.
- The stdlib REST layer is intentionally minimal; the override and diff logic already exists as reusable Python API classes and can be mounted behind a richer HTTP framework later.
