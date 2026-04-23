# Freeman Lite Architecture

## Core State

\[
\mathcal{S}_t = \left(W_t, K_t, F_t, C_t\right)
\]

where:

- \(W_t\): current `WorldState`
- \(K_t\): persistent `KnowledgeGraph`
- \(F_t\): `ForecastRegistry`
- \(C_t\): runtime counters for LLM calls and simulation steps

Persisted files:

- `kg_state.json`
- `forecasts.json`
- `world_state.json`
- `errors.jsonl`

## Compile Path

\[
\text{brief/schema} \xrightarrow{\text{compile}} W_0
\xrightarrow{\text{verify }(L0,L1,L2?)} \hat{W}_0
\xrightarrow{\text{simulate}} W_T
\xrightarrow{\text{score}} p_T(y)
\xrightarrow{\text{record}} (K_T, F_T)
\]

## Update Path

Signal routing is intentionally minimal:

\[
\text{signal} \xrightarrow{\text{dedupe + keyword gate}} \{\text{WATCH}, \text{ANALYZE}\}
\]

If `ANALYZE`:

\[
\text{signal}, W_t \xrightarrow{\text{LLM}} \theta_t
\xrightarrow{\text{update}} W_{t+1}
\]

where \(\theta_t\) is one `ParameterVector`.

## Conflict Heuristic

Instead of a separate belief-conflict subsystem, lite uses a direct posterior shift threshold:

\[
\Delta p_i = p_{t+1}(y_i) - p_t(y_i)
\]

and flags an outcome when

\[
|\Delta p_i| \ge \tau
\]

with default \(\tau = 0.25\).

## Query

Retrieval is lexical only. Node ranking uses:

\[
\text{score}(n \mid q) = s_{\text{lex}}(q,n) + 0.15 \cdot c_n + b_{\text{recency}}(n)
\]

where:

- \(s_{\text{lex}}\): token / n-gram overlap score
- \(c_n\): node confidence
- \(b_{\text{recency}}\): bounded freshness bonus from `updated_at`

## Removed from the supported surface

- Consciousness engine
- Self-model writes
- Idle / attention schedulers
- Proactive emitter
- Policy evaluator
- REST API
- MCP server
- Web and evolution viewers
- Multi-domain runtime
- Cost ledger / detailed budgeting
