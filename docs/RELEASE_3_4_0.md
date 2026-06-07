# Freeman 3.4.0

`3.4.0` is a production infrastructure release. It keeps the domain model and
agent behavior compatible with the 3.3 line while hardening runtime structure,
persistence, observability, container deployment, and startup safety.

## Release Scope

- Make `stream_runtime` maintainable by moving behavior out of the monolithic
  entrypoint into focused runtime modules.
- Add production guardrails: fail-fast startup validation, structured logging,
  health, Prometheus metrics, circuit breaker, budget `fsync`, and queue
  backpressure.
- Add a pluggable KG storage layer with SQLite as the production backend.
- Make Docker/Compose deployment reproducible with healthchecks and
  `FREEMAN_CONFIG` wiring.

## Operational Additions

Readiness and metrics:

```bash
freeman health --config config.yaml
freeman metrics --config config.yaml
```

Container run path:

```bash
docker build -t freeman:local .
docker compose up
```

The container copies `config.yaml.example` to `/app/config.yaml` for safe
defaults. In normal operation, mount your own `config.yaml` as Compose does:

```yaml
volumes:
  - ./data:/app/data
  - ./config.yaml:/app/config.yaml:ro
```

Without a mounted config, the image starts from the example config with no
remote LLM provider configured.

## Storage

The KnowledgeGraph backend is selected through config:

```yaml
memory:
  backend: sqlite   # json | sqlite
  json_path: ./data/kg_state.json
  sqlite_path: ./data/kg.db
```

Both backends share the same public `KnowledgeGraph` interface and pass
round-trip plus rollback parity tests. Explicit `export_json()` remains
available for inspection and downstream tooling.

## Package Changes

- `freeman`: `3.4.0`
- `freeman-connectors`: `3.4.0`
- `freeman-connectors` dependency: `freeman>=3.4.0,<4.0.0`
- New files:
  - `Dockerfile`
  - `docker-compose.yml`
  - `.dockerignore`
  - `freeman/runtime/metrics.py`
  - `freeman/memory/backends/*`
  - `freeman/logging_config.py`
  - `freeman/llm/circuit_breaker.py`

## Upgrade Notes

- Existing JSON KG users can continue using `memory.backend: json`.
- To switch to SQLite, set `memory.backend: sqlite` and `memory.sqlite_path`.
  JSON export is still explicit via `KnowledgeGraph.export_json()`.
- Import `freeman.verifier.fixedpoint` directly; the old
  `freeman.verifier.fixed_point` compatibility wrapper is removed.
- Production real-world runners were renamed:
  - `test_a_experiment.py` -> `manifold_experiment.py`
  - `test_a_preflight.py` -> `manifold_preflight.py`
  - `test_c_cross_domain.py` -> `cross_domain_runner.py`
- `configure_logging(json_mode=...)` is accepted as an alias for
  `json_logs=...`.

## Validation

Release validation from the repository root:

```bash
pytest
docker build -t freeman:local .
docker compose config
docker run --rm --entrypoint freeman freeman:local health --config /app/config.yaml
docker run --rm --entrypoint python freeman:local -c "import freeman, freeman_connectors"
docker run --rm freeman:local --help
docker compose run --rm --entrypoint freeman freeman health --config /app/config.yaml
docker compose run --rm freeman --help
docker compose up -d redis
docker compose down
```

Observed results:

- `pytest` -> `307 passed`
- Docker image -> `sha256:699fb4d36ce224c4b4344bd53fe0e28e4ebf04d5c8333ce7f79565110d286207`
- Docker image size -> `280276557` bytes
- Container and Compose smoke checks passed.
