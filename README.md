# Freeman Hive Mind

`hive_mind` is the compact production branch of Freeman: deterministic world simulation, persistent knowledge graph, stream runtime, and a role-scoped multi-agent layer on top of the same state.

Current package version: `3.1.2`.

## Scope

Freeman on this branch is built around four persistent objects:

\[
R_k = (W_k, K_k, C_k, F_k),
\]

where

- `W_k` is the domain world state,
- `K_k` is the persistent knowledge graph,
- `C_k` is the deterministic consciousness / self-model state,
- `F_k` is the forecast registry and verification state.

The hive-mind layer adds coordination on top of that base:

- cooperative node locking: `KnowledgeGraph.try_lock(...)`, `unlock(...)`
- node trail metadata: `trail_type`, `trail_intensity`
- edge trail weights: `KGEdge.trail_weight`, `deposit_trail(...)`
- role-scoped consciousness writes via `ConsciousState.agent_role`
- role-scoped attention routing via `AttentionScheduler.eligible_tasks(trail_scope=...)`

## What It Does

- compile a domain schema into a causal world model
- simulate shock propagation and outcome probabilities
- persist analyses, forecasts, causal paths, and self-model state
- update the world from new signals in a local stream runtime
- route work across `ingestor`, `repairer`, `planner`, `narrator`, and `verifier`
- expose the same state through CLI and MCP

## Install

Core package:

```bash
pip install .
```

Optional extras:

```bash
pip install ".[semantic]"
pip install ".[causal]"
pip install ".[dev]"
pip install ".[mcp]"
pip install ./packages/freeman-connectors
```

## Quick Start

Initialize a fresh workspace:

```bash
freeman init --config-path config.yaml
```

Run a schema directly:

```bash
freeman run --config-path config.yaml --schema-path freeman/domain/profiles/gim15.json
```

Run the stream runtime:

```bash
python -m freeman.runtime.stream_runtime \
  --config-path config.yaml \
  --bootstrap-mode llm_synthesize \
  --domain-brief-path examples/domain_brief_climate_news.md \
  --hours 8 \
  --poll-seconds 600 \
  --resume \
  --model auto
```

Query persisted state:

```bash
freeman query --config-path config.yaml --text "heat adaptation migration" --limit 5
freeman ask "What changed most since the previous similar case?" --config-path config.yaml
```

Export the graph:

```bash
freeman export-kg html runs/kg.html --config-path config.yaml
freeman export-kg json-ld runs/kg.jsonld --config-path config.yaml
freeman export-kg dot runs/kg.dot --config-path config.yaml
```

## Hive-Mind Roles

The agent-layer handoff is:

```text
ingestor -> [ingest] -> repairer -> [repair] -> planner
planner -> [read_plan] -> narrator -> [llm_propose] -> verifier
verifier -> [verified] -> planner
```

Role contracts live in:

- [docs/agents/README.md](/Users/theclimateguy/Documents/science/Freeman/docs/agents/README.md)
- [docs/agents/ingestor.md](/Users/theclimateguy/Documents/science/Freeman/docs/agents/ingestor.md)
- [docs/agents/repairer.md](/Users/theclimateguy/Documents/science/Freeman/docs/agents/repairer.md)
- [docs/agents/planner.md](/Users/theclimateguy/Documents/science/Freeman/docs/agents/planner.md)
- [docs/agents/narrator.md](/Users/theclimateguy/Documents/science/Freeman/docs/agents/narrator.md)
- [docs/agents/verifier.md](/Users/theclimateguy/Documents/science/Freeman/docs/agents/verifier.md)

## Core Files

- `freeman/agent/analysispipeline.py`
- `freeman/agent/consciousness.py`
- `freeman/agent/attentionscheduler.py`
- `freeman/agent/domainregistry.py`
- `freeman/memory/knowledgegraph.py`
- `freeman/memory/reconciler.py`
- `freeman/runtime/stream_runtime.py`

## Documentation

- Architecture: [docs/ARCHITECTURE.md](/Users/theclimateguy/Documents/science/Freeman/docs/ARCHITECTURE.md)
- Consciousness layer: [docs/CONSCIOUSNESS_ARCHITECTURE.md](/Users/theclimateguy/Documents/science/Freeman/docs/CONSCIOUSNESS_ARCHITECTURE.md)
- Ontology ingestion: [docs/ONTOLOGY_INGESTION.md](/Users/theclimateguy/Documents/science/Freeman/docs/ONTOLOGY_INGESTION.md)

## Notes

- `runtime_step` is the agent clock; forecast horizons are verified on domain time `world.t`.
- LLMs are optional and do not own mutable internal state.
- `freeman-connectors` stays separate from the core runtime.
- License: Apache License 2.0.
