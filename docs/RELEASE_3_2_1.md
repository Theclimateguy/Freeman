# Freeman Hive Mind 3.2.1

`3.2.1` is a `hive_mind` prerelease focused on making the agent stack operational rather than conceptual. It keeps the existing agent roles intact and adds runtime wiring, controlled LLM policy, and parallel node work through pluggable lock backends.

## What Changed

- Added the `freeman-hive` runtime entry point and `agent_stack` config for role-scoped execution across:
  - `ingestor`
  - `repairer`
  - `planner`
  - `narrator`
  - `verifier`
- Added OpenAI-compatible LLM configuration for hosted APIs and local Qwen-family deployments while preserving deterministic fallbacks.
- Kept verifier execution deterministic; LLM use is concentrated in bounded planning/narration paths.
- Added `LockBackend` with three concrete backends:
  - `memory` for local single-process development
  - `filesystem` for cooperative multi-process runs on shared storage
  - `redis` for horizontal deployments
- `KnowledgeGraph.try_lock()` and `unlock()` now delegate to the selected backend.
- Deployment docs now cover intra-role parallelism, horizontal scaling, and lock-backend tradeoffs.

## Operational Model

Parallelism is intentionally scoped to multiple workers competing for nodes within the same role:

\[
  \text{parallelism} = \{(role, node_i)\}_{i=1}^{N}, \quad node_i \neq node_j
\]

Role order and trail routing remain unchanged. This avoids cross-role races on the same node while allowing a deployment to scale ingestion, repair, planning, narration, or verification throughput independently.

## Deployment Notes

- Use `agent_stack.lock_backend: redis` for production horizontal scaling.
- Use `agent_stack.lock_backend: filesystem` only when all workers share one low-latency filesystem.
- Use `agent_stack.lock_backend: memory` for local development and tests.
- Install Redis support with:

```bash
pip install "freeman[redis]"
```

## Validation

- `pytest tests/test_hive_mind.py` -> `18 passed`
- `pytest tests/test_hive_mind.py tests/test_openai_client.py tests/test_runtime.py` -> `48 passed`
- Hive runtime coverage gate -> `86.19%`
- GitHub Actions `hive_mind Tests` run `26691472006` -> success
