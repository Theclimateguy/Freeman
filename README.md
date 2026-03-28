# Freeman

Freeman is a domain-agnostic simulation and agent framework for structured reasoning over actors, resources, causal graphs, uncertainty, semantic memory, forecast verification, and proactive agent behavior. The current codebase implements the USIM-AGENT roadmap through v0.2, plus semantic KG retrieval, obligation-driven scheduling, forecast tracking, self-model updates, replay-based behavioral tests, and a stateful regime-shift simulator update for longitudinal benchmark cases.

## What Is Implemented

- Core simulator: `WorldGraph` / `WorldState`, transition operators, outcome scoring, multi-domain shared-resource worlds.
- Stateful regime updates: decayed shock accumulation via `WorldGraph.apply_shocks(..., time_decay=...)` plus conditional outcome multipliers through `Outcome.regime_shifts`.
- Verifier: Level 0 invariants, Level 1 structural checks, Level 2 sign consistency, fixed-point repair, spectral-radius guard.
- Memory: NetworkX + JSON knowledge graph, ChromaDB semantic retrieval, session logs, confidence reconciliation, self-model nodes, export to HTML / JSON-LD / DOT.
- Agent layer: domain template registry, analysis pipeline, signal ingestion, signal memory, UCB attention scheduler, obligation queue, forecast registry, proactive emitter, formal cost model.
- Interface layer: CLI commands, minimal REST API, KG export, human override API, simulation diff.
- v0.2 extensions: compile validation, historical fit scoring, ensemble sign consensus, Monte Carlo uncertainty, cost governance, override audit trail.
- Behavioral validation: deterministic `AgentHarness`, replay fixtures, and end-to-end stimulus tests for watch/analyze/escalation behavior.
- Live evaluation tooling: real-LLM scenario runner over economic, social, and media-release domains with semantic memory recall.
- Benchmark tooling: `scripts/benchmark_faab/` for longitudinal A/B evaluation of full-memory Freeman vs amnesic/hash/LLM-only baselines.

## Install

```bash
pip install -e .
```

Optional semantic-memory extras:

```bash
pip install -e ".[semantic]"
```

Benchmark extras:

```bash
pip install -e ".[benchmark]"
```

Local embedding backend with Ollama:

```bash
brew install ollama
brew services start ollama
ollama pull nomic-embed-text
ollama pull mxbai-embed-large
```

The default `config.yaml` uses `memory.embedding_provider: ollama` and `memory.embedding_model: nomic-embed-text`. To switch to the larger local model, set `memory.embedding_model: mxbai-embed-large` or export `FREEMAN_EMBEDDING_MODEL=mxbai-embed-large` for the live runner.

## Quick Start

Run the full test suite:

```bash
pytest tests/
```

Run the replay-driven behavioral suite only:

```bash
pytest tests/test_agent_behavior.py
```

Run the live DeepSeek + Ollama end-to-end evaluation:

```bash
python scripts/run_real_llm_e2e.py --output-dir runs/real_llm_e2e --max-steps 12 --top-k 6
```

Run the FAAB longitudinal benchmark:

```bash
python scripts/benchmark_faab/run_benchmark.py --dataset scripts/benchmark_faab/dataset/cases.jsonl --output-dir runs/faab_real_regime_v1
```

Inspect the current knowledge graph status:

```bash
python -m freeman.interface.cli status
```

Run a domain schema end-to-end:

```bash
python -m freeman.interface.cli run path/to/schema.json
```

Query the knowledge graph:

```bash
python -m freeman.interface.cli query --text "water stress"
```

Reindex legacy KG nodes into the semantic vector store:

```bash
python -m freeman.interface.cli kg-reindex --batch-size 100
```

Export the knowledge graph:

```bash
python -m freeman.interface.cli export-kg html runs/kg.html
python -m freeman.interface.cli export-kg json-ld runs/kg.jsonld
python -m freeman.interface.cli export-kg dot runs/kg.dot
```

Apply human overrides:

```bash
python -m freeman.interface.cli override-param world.json resources.x.value 12.0 --output-path world_override.json
python -m freeman.interface.cli override-sign world.json x->y + --output-path world_override.json
python -m freeman.interface.cli rerun-domain world_override.json --max-steps 10 --output-path rerun.json
python -m freeman.interface.cli diff-domain world.json rerun.json --output-path diff.json
```

## Main Entry Points

- Core Python API:
- `freeman.core`
  - includes stateful shock decay and regime-shift scoring
  - `freeman.verifier`
  - `freeman.memory`
  - `freeman.agent`
  - `freeman.interface`
- CLI:
  - `python -m freeman.interface.cli`
- Minimal REST server:
  - `freeman.interface.api.run_server()`
- LLM orchestration:
  - `freeman.llm.orchestrator.DeepSeekFreemanOrchestrator`
  - `freeman.llm.OllamaEmbeddingClient`
- Deterministic replay harness:
  - `tests/harness.py::AgentHarness`

## Documentation

- Architecture and workflows: [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)
- API map: [docs/API_MAP.md](docs/API_MAP.md)
- FAAB benchmark and recorded regime-shift run: [docs/FAAB.md](docs/FAAB.md)
- Release notes: [CHANGELOG.md](CHANGELOG.md)
- Live E2E evaluation report: [docs/REAL_LLM_E2E.md](docs/REAL_LLM_E2E.md)

## Repository Layout

- `freeman/core/` — simulator state, evolution operators, transition logic, scoring, compile validation, uncertainty.
- `scripts/benchmark_faab/` — longitudinal benchmark runner, dummy dataset generator, and evaluation fixtures.
- `freeman/verifier/` — verifier levels, aggregate verifier, fixed-point correction.
- `freeman/memory/` — knowledge graph, semantic vector store, reconciliation, session logs.
- `freeman/agent/` — analysis pipeline, scheduling, signal ingestion, forecast tracking, proactive emission, cost governance.
- `freeman/interface/` — CLI, REST API, export, override and diff helpers.
- `freeman/domain/` — schema compiler and bundled profiles.
- `freeman/llm/` — provider-facing LLM orchestration and repair loop.
- `tests/` — unit, integration, replay fixtures, and behavioral agent harness coverage.

## Notes

- Default long-term memory backend is NetworkX + JSON via `config.yaml -> memory.json_path`.
- Semantic retrieval is optional and uses ChromaDB when installed with the `semantic` extra.
- Local semantic retrieval defaults to Ollama with `nomic-embed-text`; `mxbai-embed-large` is supported through the same adapter.
- The regime-shift benchmark baseline recorded in-repo lives under `runs/faab_real_regime_v1/`.
- Signal replay fixtures live under `tests/fixtures/signals/` and are used by `tests/test_agent_behavior.py`.
- `pytest tests/` is the required validation command and is wired to work without manual `PYTHONPATH` changes.
- The stdlib REST layer is intentionally minimal; the override and diff logic already exists as reusable Python API classes and can be mounted behind a richer HTTP framework later.
