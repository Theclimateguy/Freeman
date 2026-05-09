# Freeman Architecture

This document describes the compact operational architecture of the `hive_mind` branch.

## Formal Runtime State

At runtime Freeman evolves

\[
R_k = (W_k, K_k, C_k, F_k, Q_k, B_k),
\]

where

- `W_k`: `WorldState` / `WorldGraph`
- `K_k`: persistent `KnowledgeGraph`
- `C_k`: `ConsciousState` and `SelfModelGraph`
- `F_k`: `ForecastRegistry`
- `Q_k`: pending signals, cursor state, and attention frontier
- `B_k`: budget and cost state

The outer transition is

\[
R_{k+1} = \Phi(R_k, x_k, u_k),
\]

with external signal `x_k` and optional internal act `u_k`.

## Layer Map

1. `freeman.core`
World model, transition operators, scoring, uncertainty, compile validation.

2. `freeman.verifier`
Structural, numerical, and sign checks over the world state.

3. `freeman.memory`
Persistent knowledge graph, reconciliation, causal-path history, self-model storage.

4. `freeman.agent`
Signal ingestion, attention scheduling, analysis pipeline, consciousness, role routing.

5. `freeman.runtime`
Checkpointing, stream execution, forecast verification, query/answer surface.

6. `freeman.interface`
CLI, exports, identity/explanation views, overrides, MCP entrypoints.

## Simulation Core

The inner world step is

\[
W_{t+1} = W_t + F_{\theta,D}(W_t, \pi_t),
\]

with dynamic calibration

\[
\theta_t = (\alpha_t, \lambda_t, \Delta W_t),
\]

implemented as:

- `outcome_modifiers`
- `shock_decay`
- `edge_weight_deltas`

Outcome probabilities are computed from raw scores

\[
z_o = W_o^\top S_t, \qquad
p(o_t) = \frac{e^{z_o}}{\sum_j e^{z_j}}.
\]

Forecast deadlines are evaluated on domain time:

\[
\text{deadline_step} = \text{created_step} + \text{horizon_steps},
\]

where `created_step = world.t`.

## Memory and Trails

Freeman stores both node-level and edge-level routing state.

### Node trails

Each node may carry:

- `metadata["trail_type"]`
- `metadata["trail_intensity"]`

On `KnowledgeGraph.update_node(...)`:

\[
\text{trail\_intensity} \leftarrow 0.9 \cdot \text{trail\_intensity},
\]

and the trail is removed when intensity falls below `0.05`.

### Edge trails

Causal edges may carry `trail_weight`. Successful updates call:

\[
\text{trail\_weight}_e \leftarrow \text{trail\_weight}_e + q,
\]

where `q` is update quality / confidence. Reconciliation evaporates that weight as

\[
\text{trail\_weight}_e \leftarrow e^{-\gamma}\text{trail\_weight}_e.
\]

### Locks

Shared-node coordination uses cooperative locks:

- `try_lock(node_id, agent_id, lock_ttl_seconds=...)`
- `unlock(node_id, agent_id=...)`

This prevents concurrent multi-agent writes to the same frontier node.

## Hive-Mind Roles

Roles are explicit:

- `ingestor`
- `repairer`
- `planner`
- `narrator`
- `verifier`

The role field lives in `ConsciousState.agent_role`. Mutating acts are checked through `ConsciousnessEngine._require_permission(...)`, raising `RolePermissionError` on forbidden writes.

Trail scopes are:

- `ingestor -> {None}`
- `repairer -> {"ingest", "llm_propose"}`
- `planner -> {"verified", "repair"}`
- `narrator -> {"read_plan"}`
- `verifier -> {"llm_propose"}`

These are registered in `freeman.agent.domainregistry.ROLE_TRAIL_SCOPE`.

## Attention Routing

Each task carries information features plus optional trail metadata:

\[
I_i =
\frac{
\text{eig}_i
+ \text{anomaly}_i
+ \text{semanticGap}_i
+ \text{confidenceGap}_i
+ \text{obligation}_i
+ \text{trailWeight}_i
}{\text{cost}_i}.
\]

`trail_type` then shifts the frontier:

- `ingest`: boosts anomaly pressure
- `repair`: cools semantic-gap urgency
- `llm_propose`: boosts confidence-gap urgency
- `verified`: downweights the full task
- `read_plan`: applies a mild cooldown

`AttentionScheduler.eligible_tasks(trail_scope=...)` filters tasks so different roles see disjoint frontiers when their scopes do not overlap.

## Runtime Flow

The main runtime path is:

```text
sources -> SignalIngestionEngine -> AttentionScheduler -> AnalysisPipeline
-> KnowledgeGraph / ForecastRegistry / ConsciousnessEngine
-> Reconciler -> checkpoint / query / export
```

Bootstrap supports:

- `schema_path`
- `llm_synthesize`

and persists the chosen ingestion contract in `bootstrap_package.json`.

## Invariants

- world updates are deterministic given the same structured inputs
- mutable agent identity lives in structured state, not in prompts
- LLMs are read-only interpreters, not the source of truth
- forecast verification is anchored to `world.t`, not `runtime_step`
- role permissions and trail scopes constrain who can write and what each role can target

## Adjacent Docs

- [docs/agents/README.md](/Users/theclimateguy/Documents/science/Freeman/docs/agents/README.md)
- [docs/CONSCIOUSNESS_ARCHITECTURE.md](/Users/theclimateguy/Documents/science/Freeman/docs/CONSCIOUSNESS_ARCHITECTURE.md)
- [docs/ONTOLOGY_INGESTION.md](/Users/theclimateguy/Documents/science/Freeman/docs/ONTOLOGY_INGESTION.md)
