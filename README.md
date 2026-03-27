# Freeman

Freeman is a domain-agnostic simulation and agent framework for structured reasoning over actors, resources, causal graphs, uncertainty, verification, and long-term memory. The current codebase implements the USIM-AGENT roadmap through v0.2, including the deterministic simulator, three-level verifier, knowledge graph, agent scheduling, uncertainty propagation, cost governance, human overrides, and end-to-end tests.

## What Is Implemented

- Core simulator: `WorldGraph` / `WorldState`, transition operators, outcome scoring, multi-domain shared-resource worlds.
- Verifier: Level 0 invariants, Level 1 structural checks, Level 2 sign consistency, fixed-point repair, spectral-radius guard.
- Memory: NetworkX + JSON knowledge graph, session logs, confidence reconciliation, export to HTML / JSON-LD / DOT.
- Agent layer: domain template registry, analysis pipeline, signal ingestion, UCB attention scheduler, formal cost model.
- Interface layer: CLI commands, minimal REST API, KG export, human override API, simulation diff.
- v0.2 extensions: compile validation, historical fit scoring, ensemble sign consensus, Monte Carlo uncertainty, cost governance, override audit trail.

## Install

```bash
pip install -e .
```

## Quick Start

Run the full test suite:

```bash
pytest tests/
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

## Documentation

- Architecture and workflows: [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)
- API map: [docs/API_MAP.md](docs/API_MAP.md)
- Release notes: [CHANGELOG.md](CHANGELOG.md)
- Implementation progress by phase: [PROGRESS.md](PROGRESS.md)

## Repository Layout

- `freeman/core/` — simulator state, evolution operators, transition logic, scoring, compile validation, uncertainty.
- `freeman/verifier/` — verifier levels, aggregate verifier, fixed-point correction.
- `freeman/memory/` — knowledge graph, reconciliation, session logs.
- `freeman/agent/` — analysis pipeline, scheduling, signal ingestion, cost governance.
- `freeman/interface/` — CLI, REST API, export, override and diff helpers.
- `freeman/domain/` — schema compiler and bundled profiles.
- `freeman/llm/` — provider-facing LLM orchestration and repair loop.
- `tests/` — unit and integration coverage.

## Notes

- Default long-term memory backend is NetworkX + JSON via `config.yaml -> memory.json_path`.
- `pytest tests/` is the required validation command and is wired to work without manual `PYTHONPATH` changes.
- The stdlib REST layer is intentionally minimal; the override and diff logic already exists as reusable Python API classes and can be mounted behind a richer HTTP framework later.
