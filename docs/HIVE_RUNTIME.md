# Hive Runtime

The hive runtime turns the existing `hive_mind` role contracts into an executable dispatcher without changing the roles:

\[
Q_r(k)=\{v \in K_k:\operatorname{trail}(v)\in S_r,\ n_r(v)<N_r\},
\]

where `S_r` is `trail_scope_for_role(r)` and `n_r(v)` is the persisted per-node visit count for role `r`.

## Runtime Cycle

For each cycle and each role in

```text
ingestor -> repairer -> planner -> narrator -> verifier
```

the dispatcher:

1. selects KG nodes whose `trail_type` is visible to that role;
2. acquires `KnowledgeGraph.try_lock(node_id, runtime_id:role)`;
3. optionally calls the configured role LLM;
4. writes the next trail:

```text
ingestor -> ingest
repairer -> repair
planner  -> read_plan
narrator -> llm_propose
verifier -> verified
```

5. appends a `TraceEvent`;
6. persists `hive_checkpoint.json` and `hive_event_log.jsonl`.

The dispatcher records `hive_runtime.role_counts` and defaults to one visit per role per node, which prevents a verified node from looping indefinitely back through planner/narrator/verifier.

## LLM Profiles

LLM calls are optional and disabled by default:

```yaml
agent_stack:
  llm:
    enabled: false
    roles: ["narrator", "planner"]
```

When LLM calls are enabled, the default augmented roles are `planner` and `narrator`. The planner is the highest-value LLM target because it ranks repaired/verified evidence and writes the `read_plan` handoff. The narrator can turn a stable plan into a structured proposal. The verifier should remain deterministic.

Local Qwen models are resolved through Ollama:

```yaml
agent_stack:
  llm:
    enabled: true
    roles: ["narrator", "planner"]
    role_models:
      default:
        provider: "ollama"
        model: "qwen2.5-coder:14b"
        base_url: "http://127.0.0.1:11434"
```

OpenAI-compatible endpoints use the same chat-completions client:

```yaml
agent_stack:
  llm:
    enabled: true
    roles: ["narrator", "planner"]
    role_models:
      planner:
        provider: "openai-compatible"
        model: "gpt-4o-mini"
        base_url: "https://api.openai.com/v1"
        api_key_env: "OPENAI_API_KEY"
```

For local OpenAI-compatible servers on `localhost` or `127.0.0.1`, the runtime accepts a missing API key and sends `EMPTY`.

## Role LLM Policy

| Role | Recommended mode | Rationale |
| --- | --- | --- |
| `ingestor` | deterministic | It normalizes raw signals into first KG candidates; adding generation here increases schema drift and duplicate claims. |
| `repairer` | deterministic | It guards ontology integrity and repair queues; nondeterministic writes can hide consistency violations. |
| `planner` | LLM-recommended when enabled | It reads repaired/verified frontiers and benefits from synthesis, ranking, and explicit hypothesis framing. |
| `narrator` | LLM-optional | It externalizes planner state into proposal text; useful for analyst-facing handoffs, but not required for state progression. |
| `verifier` | deterministic | It is the trust gate. If it becomes generative, verification confidence and replayability become model-dependent. |

Latency risk: the runtime can call one model per eligible role/node pair. With Ollama defaults, the worst-case wall time is roughly `120s * frontier_limit_per_role * len(llm_roles)` per cycle unless the model returns earlier. Keep `frontier_limit_per_role` small when local models are slow.

Cost risk: OpenAI-compatible providers charge per token and per role frontier. Restrict `agent_stack.llm.roles`, lower `max_tokens`, and prefer planner-only augmentation for production runs with remote endpoints.

## Known Limitations

Distributed locking is backend-dependent. The default `memory` backend preserves the original cooperative KG-node lock fields and is suitable for a single runtime process, a single host, or carefully serialized operational runs, but it is not a distributed consensus lock. Do not run multiple hive runtimes against the same mutable KG path from different hosts or unsynchronized processes with `lock_backend: "memory"` and assume exactly-once role execution.

`filesystem` is a zero-dependency single-host multi-process backend based on atomic lock files. `redis` is the first cross-host backend and uses Redis `SET NX PX` leases. Install it with `pip install ".[redis]"` or provide the `redis` package in the runtime image. Production deployments that need concurrent workers should use `redis` or put an equivalent lease layer in front of the KG, for example Postgres advisory locks, etcd leases, or a queue with visibility timeouts.

## Horizontal Scaling

The runtime preserves role order per node by routing only nodes whose current `trail_type` is visible to a role. Parallelism is intra-role: multiple `planner` workers may process different `repair` or `verified` nodes, but a node cannot reach `narrator` until one planner has written `trail_type: read_plan`.

Lock backend options:

| Backend | Scope | Config | Use when |
| --- | --- | --- | --- |
| `memory` | one process / serialized local runs | `lock_backend: "memory"` | Default compatibility mode and tests. |
| `filesystem` | one host, multiple processes | `lock_backend: "filesystem"` | Local worker fan-out without Redis. |
| `redis` | multiple hosts/processes | `lock_backend: "redis"` plus `lock_redis_url` or `FREEMAN_REDIS_URL` | Production horizontal scaling. |

Redis deployment example:

```yaml
agent_stack:
  lock_backend: "redis"
  lock_redis_url: "redis://redis:6379/0"
  lock_ttl_seconds: 120
```

Filesystem deployment example:

```yaml
agent_stack:
  lock_backend: "filesystem"
  lock_filesystem_path: "/var/lib/freeman/node_locks"
  lock_ttl_seconds: 120
```

Production baseline:

```yaml
agent_stack:
  enabled: true
  runtime_id: "hive-prod"
  role_order: ["ingestor", "repairer", "planner", "narrator", "verifier"]
  frontier_limit_per_role: 3
  max_actions_per_cycle: 15
  max_role_revisits_per_node: 1
  lock_backend: "redis"
  lock_redis_url: "redis://redis:6379/0"
  lock_ttl_seconds: 120
  llm:
    enabled: true
    roles: ["planner"]
    temperature: 0.1
    max_tokens: 384
    role_models:
      default:
        provider: "ollama"
        model: "qwen2.5-coder:14b"
        base_url: "http://127.0.0.1:11434"
        timeout_seconds: 120
```

## Run

```bash
python -m freeman.runtime.hive_runtime --config-path config.yaml --cycles 1
```

or after package installation:

```bash
freeman-hive --config-path config.yaml --cycles 1
```
