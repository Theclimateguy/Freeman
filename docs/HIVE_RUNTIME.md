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
    roles: ["narrator"]
```

Local Qwen models are resolved through Ollama:

```yaml
agent_stack:
  llm:
    enabled: true
    roles: ["narrator"]
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
    roles: ["narrator", "verifier"]
    role_models:
      verifier:
        provider: "openai-compatible"
        model: "gpt-4o-mini"
        base_url: "https://api.openai.com/v1"
        api_key_env: "OPENAI_API_KEY"
```

For local OpenAI-compatible servers on `localhost` or `127.0.0.1`, the runtime accepts a missing API key and sends `EMPTY`.

## Run

```bash
python -m freeman.runtime.hive_runtime --config-path config.yaml --cycles 1
```

or after package installation:

```bash
freeman-hive --config-path config.yaml --cycles 1
```
