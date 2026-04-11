# Freeman

Freeman is an analytical agent for situations that change over time. You describe a system as actors, resources, causal links, and possible outcomes; Freeman simulates that system, stores the result in memory, and helps you update the picture when new signals arrive.

The point of Freeman is not "chatting about a topic". The point is to keep a structured world model, run repeatable scenario analysis, remember past cases, and give you an audit trail of what changed and why.

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
- Optionally add semantic retrieval over that graph with ChromaDB.
- Answer questions against stored memory with `freeman ask`.
- Track forecasts, conflicts, anomalies, and proactive events in the agent layer.
- Export the knowledge graph as HTML, JSON-LD, or DOT.
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
freeman export-kg json-ld runs/kg.jsonld --config-path config.yaml
freeman export-kg dot runs/kg.dot --config-path config.yaml
```

Ask a follow-up question after you have accumulated relevant memory and configured an LLM summarizer:

```bash
freeman ask "What changed most since the previous similar case?" --config-path config.yaml
```

Human-in-the-loop workflow:

```bash
freeman override-param world.json resources.x.value 12.0 --output-path world_override.json --config-path config.yaml
freeman override-sign world.json x->y + --output-path world_override.json --config-path config.yaml
freeman rerun-domain world_override.json --max-steps 10 --output-path rerun.json --config-path config.yaml
freeman diff-domain world.json rerun.json --output-path diff.json --config-path config.yaml
```

Long local stream run (climate RSS + Ollama):

```bash
python -m freeman.runtime.climate_stream \
  --config-path config.climate.yaml \
  --schema-path freeman/domain/profiles/gim15.json \
  --hours 8 \
  --resume \
  --model auto
```

## Main Commands

- `freeman init`: create a config file and empty storage for the knowledge graph and session logs.
- `freeman run`: compile a schema and run the full analysis pipeline.
- `freeman ask`: retrieve relevant memory and, when an LLM is configured, answer a question from stored graph context.
- `freeman status`: inspect configured storage paths and current active-memory counts.
- `freeman query`: query graph nodes directly without summarization.
- `freeman export-kg`: export the graph for inspection or downstream tooling.
- `freeman reconcile`: merge one saved session log back into the long-term graph.
- `freeman kg-archive`: archive low-confidence or obsolete graph nodes.
- `freeman kg-reindex`: rebuild embeddings for semantic retrieval.
- `freeman override-param`, `override-sign`, `rerun-domain`, `diff-domain`: edit assumptions and compare before/after simulations.

## Advanced Use Cases

If you are integrating Freeman into Python code rather than only using the CLI, the main entry points are:

- `freeman.agent.AnalysisPipeline`: compile -> simulate -> verify -> write to memory.
- `freeman.agent.SignalIngestionEngine`: normalize incoming signals and decide whether they deserve attention.
- `freeman.agent.ParameterEstimator`: ask an LLM to adjust an existing world when new evidence changes the regime.
- `freeman.memory.KnowledgeGraph`: persistent graph memory with optional semantic retrieval.
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

- Architecture: [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)
- API map: [docs/API_MAP.md](docs/API_MAP.md)
- Benchmark notes: [docs/FAAB.md](docs/FAAB.md)
- Live end-to-end evaluation: [docs/REAL_LLM_E2E.md](docs/REAL_LLM_E2E.md)

## Notes

- Default long-term memory is a JSON-backed NetworkX graph.
- Semantic retrieval is optional and uses ChromaDB when installed with the `semantic` extra.
- `freeman init` creates an empty agent workspace; it does not preload domain knowledge.
- The core package is source-agnostic. If you need RSS, HTTP/JSON, HTML page, or arXiv ingestion, use `freeman-connectors`.
