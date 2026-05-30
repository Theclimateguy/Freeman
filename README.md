# Freeman Hive Mind

[![Tests](https://github.com/Theclimateguy/Freeman/actions/workflows/tests.yml/badge.svg?branch=hive_mind)](https://github.com/Theclimateguy/Freeman/actions/workflows/tests.yml?query=branch%3Ahive_mind)

`hive_mind` is the compact production branch of Freeman: deterministic world simulation, persistent knowledge graph, stream runtime, and a role-scoped multi-agent layer on top of the same state.

Current package version: `3.2.0`.

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
- safe ontology repair: inferred causal edges are review-queued by default, not silently appended
- structural runtime contracts for `WorldState`, `KnowledgeGraph`, and `ConsciousState`

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

Run the hive-mind role dispatcher:

```bash
python -m freeman.runtime.hive_runtime --config-path config.yaml --cycles 1
# or, after installation:
freeman-hive --config-path config.yaml --cycles 1
```

Query persisted state:

```bash
freeman query --config-path config.yaml --text "heat adaptation migration" --limit 5
freeman ask "What changed most since the previous similar case?" --config-path config.yaml
freeman what-if "What if Country A increases water releases?" \
  --config-path config.yaml \
  --policies-path examples/scenario_policies.json \
  --max-steps 8
```

`ask` answers from persisted runtime evidence only. `what-if` first simulates the current `world_state.json` under scenario policies, then uses runtime evidence as mechanism-level calibration for the final answer.

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
- Full architecture: [docs/FREEMAN_FULL_ARCHITECTURE.md](/Users/theclimateguy/Documents/science/Freeman/docs/FREEMAN_FULL_ARCHITECTURE.md)
- 3D architecture: [docs/FREEMAN_ARCHITECTURE_3D.html](/Users/theclimateguy/Documents/science/Freeman/docs/FREEMAN_ARCHITECTURE_3D.html)
- Consciousness layer: [docs/CONSCIOUSNESS_ARCHITECTURE.md](/Users/theclimateguy/Documents/science/Freeman/docs/CONSCIOUSNESS_ARCHITECTURE.md)
- Ontology ingestion: [docs/ONTOLOGY_INGESTION.md](/Users/theclimateguy/Documents/science/Freeman/docs/ONTOLOGY_INGESTION.md)
- Release 3.2 notes: [docs/RELEASE_3_2.md](/Users/theclimateguy/Documents/science/Freeman/docs/RELEASE_3_2.md)

## Notes

- `runtime_step` is the agent clock; forecast horizons are verified on domain time `world.t`.
- contradictory retained signals are marked with conflict metadata before budgeting and downstream belief updates.
- compiled worlds created through `freeman.api.tool_api` persist under `runtime/compiled_worlds.json`, so tool-driven simulations survive process restarts.
- `llm.provider: openai-compatible` is accepted as an alias of `openai`; use `llm.base_url` to point at any compatible endpoint.
- `agent_stack.llm.role_models` can bind individual hive roles to local Qwen/Ollama or OpenAI-compatible chat endpoints.
- LLMs are optional and do not own mutable internal state.
- `freeman-connectors` stays separate from the core runtime.
- License: Apache License 2.0.
