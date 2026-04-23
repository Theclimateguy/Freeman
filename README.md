# Freeman Lite

`freeman-lite` is a reduced Freeman branch for one job:

\[
\text{news / brief} \rightarrow \text{causal model} \rightarrow \text{verify} \rightarrow \text{simulate} \rightarrow \text{memory}
\]

This branch removes proactive layers, consciousness, multi-domain scheduling, REST/MCP/web surfaces, and budget ledgers. The public surface is now:

- CLI: `run`, `query`, `export-kg`
- Python API: `compile`, `update`, `query`
- Persistence: `kg_state.json`, `forecasts.json`, `world_state.json`, `errors.jsonl`

## Pipeline

For a base compile:

1. Compile schema directly or synthesize it from a brief with one LLM client.
2. Verify with `L0 + L1` and optional `L2`.
3. Simulate one world.
4. Score outcomes and record forecasts.
5. Export causal edges and reconcile the knowledge graph.

For an update:

1. Deduplicate the signal.
2. Apply a keyword relevance gate.
3. Classify `WATCH` or `ANALYZE`.
4. For `ANALYZE`, estimate one `ParameterVector` and rerun the pipeline.
5. Verify due forecasts against the new world state and persist the result.

## Installation

```bash
pip install -e .
```

Console entrypoint:

```bash
freeman-lite --help
```

## Minimal Config

See [config.yaml.example](config.yaml.example).

Only four groups remain:

- `paths`
- `llm`
- `limits`
- `signals`

## CLI

Compile from an existing schema:

```bash
freeman-lite run --schema examples/water_market.yaml
```

Compile from a domain brief:

```bash
freeman-lite run --brief examples/domain_brief_climate_news.md
```

Update the current world with a signal:

```bash
freeman-lite run --signal "Severe drought cuts reservoir inflows across the basin."
```

Query the KG:

```bash
freeman-lite query "water crisis"
```

Export the KG:

```bash
freeman-lite export-kg --output artifacts/kg.json
```

## Python API

```python
from freeman import compile, query, update

compile("examples/water_market.yaml")
update("Severe drought cuts reservoir inflows across the basin.")
result = query("water crisis")
```

## Notes

- The lite runtime keeps one world in memory.
- ChromaDB and vector retrieval are no longer part of the default runtime path.
- If a limit is exceeded, the runtime stops with an explicit error instead of trying to self-manage compute.
- Legacy experimental modules may still exist in the repo history, but they are no longer part of the supported surface for this branch.
