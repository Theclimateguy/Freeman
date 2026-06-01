# Freeman 3.2

`3.2.0` is the official 3.2 release cut from the stabilized `hive_mind` branch. It keeps the E2E runtime compact while closing the main architecture-review gaps around ontology repair safety, layer contracts, signal conflicts, and release CI.

## What Changed

- Ontology repair is now metadata-first by default:
  - schema topics, aliases, relation candidates, and proposal nodes are persisted automatically
  - inferred causal DAG edges are queued for review unless `auto_apply_relation_candidates: true`
  - auto-apply uses a higher default confidence gate (`auto_apply_min_confidence: 0.75`)
- Runtime layer contracts are explicit in `freeman.runtime.contracts`:
  - `WorldStateContract`
  - `KnowledgeGraphContract`
  - `ConsciousStateContract`
- Signal ingestion now marks contradictory retained signals:
  - `conflict_score`
  - `conflict_reason`
  - `conflicts_with`
- The stream runtime checks the current signal against the pending queue, so single-signal processing still surfaces near-term contradictions.
- GitHub Actions now cover full pytest, CLI smoke tests, build artifacts, `hive_mind-*` prereleases, and official `v*` release tags.

## Why It Matters

- Freeman no longer silently converts semantic co-mentions into causal DAG mutations in the default path.
- The core state interfaces are documented and tested without coupling the implementation to inheritance.
- Contradictory evidence remains visible to downstream belief-conflict handling instead of being hidden by ingestion order.
- The `hive_mind` branch now serves as the release base for a repeatable, artifact-backed 3.2 line.

## Validation

- `pytest -q` -> `265 passed`
- `python -m compileall -q freeman`
- `python -m build --sdist --wheel --outdir /tmp/freeman-3.2.0-dist .`
- `python -m build --sdist --wheel --outdir /tmp/freeman-connectors-3.2.0-dist packages/freeman-connectors`

## Upgrade Notes

- Default ontology repair behavior changed. If a deployment intentionally wants automatic weak relation insertion, set:

```yaml
agent:
  ontology_repair:
    auto_apply_relation_candidates: true
    auto_apply_min_confidence: 0.75
```

- Forecast horizons remain evaluated on domain time (`world.t`), not `runtime_step`.
