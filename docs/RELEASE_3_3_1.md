# Freeman 3.3.1

`3.3.1` is the full `main` release that promotes the stabilized 3.2
`hive_mind` line and adds the regional geospatial layer.

## Release Scope

- Bring the release-grade hive runtime from the 3.2 line into `main`.
- Keep generated experiment/runtime artifacts out of the release commit.
- Make spatial metadata active in compiled worlds rather than passive schema
  metadata.
- Refresh README, architecture docs, release notes, packaging metadata, and CI
  expectations around the new runtime surface.

## Hive Runtime

The hive runtime adds a role-scoped dispatcher over the persistent KG:

```text
ingestor -> repairer -> planner -> narrator -> verifier
```

Each role reads only the trail states in its scope, acquires a cooperative
KG-node lock, optionally uses its configured LLM policy, writes the next trail
state, and persists `hive_checkpoint.json` / `hive_event_log.jsonl`.

Operational entrypoints:

```bash
freeman-hive --config-path config.yaml --cycles 1
python -m freeman.runtime.hive_runtime --config-path config.yaml --cycles 1
```

Optional Redis locking is available with:

```bash
pip install ".[redis]"
```

## Geospatial Layer

`world.metadata["spatial"]` is now read during compilation. Regional adjacency
is materialized into actor-level relations:

```text
spatial.adjacency -> Relation(relation_type="spatial_neighbor")
```

This means the simulator, query layer, and KG-facing reasoning paths see
geography as explicit graph structure without manually declaring every regional
neighbor relation.

Optional vector geometry support is available with:

```bash
pip install ".[geo]"
```

See [GEO_ANALYTICS.md](GEO_ANALYTICS.md) for schema contracts, supported
adjacency forms, warnings for large actor-pair expansion, and
GeoPandas/Shapely adapter usage.

## Package Changes

- `freeman`: `3.3.1`
- `freeman-connectors`: `3.3.1`
- New optional extras:
  - `geo`: GeoPandas/Shapely regional adapter.
  - `redis`: Redis-backed hive lock backend.
- New console script:
  - `freeman-hive`

## Upgrade Notes

- Existing schemas continue to compile unchanged when `spatial` metadata is
  absent.
- If `spatial.adjacency` is present, actor metadata should contain one of the
  supported region keys such as `geo_id`, `region_id`, `spatial_region`, or
  explicit `spatial.actor_regions`.
- For more than roughly 100 mapped regions or more than 10000 estimated
  materialized actor relations, Freeman logs a warning. Aggregate actors or
  prune weak adjacency edges if compilation becomes too dense.
- The default hive lock backend is process-local memory. Use Redis before
  running multiple hive workers against the same mutable KG from separate
  processes or hosts.

## Validation

Release validation is run from the repository root:

```bash
PYTHONPATH="$PWD/packages/freeman-connectors:$PYTHONPATH" python -m pytest -q
python -m compileall -q freeman packages/freeman-connectors/freeman_connectors
python -m build --sdist --wheel --outdir /tmp/freeman-3.3.1-dist .
python -m build --sdist --wheel --outdir /tmp/freeman-connectors-3.3.1-dist packages/freeman-connectors
```

Results for the release cut:

- `PYTHONPATH="$PWD/packages/freeman-connectors:$PYTHONPATH" python -m pytest -q` -> `287 passed`
- `python -m compileall -q freeman packages/freeman-connectors/freeman_connectors` -> passed
- Core package build -> `freeman-3.3.1.tar.gz`, `freeman-3.3.1-py3-none-any.whl`
- Connector package build -> `freeman_connectors-3.3.1.tar.gz`, `freeman_connectors-3.3.1-py3-none-any.whl`
