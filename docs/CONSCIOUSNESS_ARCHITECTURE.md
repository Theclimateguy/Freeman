# Consciousness Architecture for Freeman

This document defines the next development stage for Freeman: a deterministic consciousness layer built on top of the existing world model, knowledge graph, and reconciliation machinery.

The design goal is to keep reasoning, identity, and continuity inside structured state rather than inside an LLM prompt. LLMs remain optional read-only interpreters of that state.

## Design Principle

The target architecture is:

```text
ConsciousState (thinks) -> LLM (translates) -> external world
```

instead of:

```text
LLM (thinks) -> tools (execute) -> LLM (explains)
```

This preserves:

- reproducibility of internal state
- auditability of belief changes
- replaceability of the LLM layer without identity loss

## Formal State

Define the agent state at time `t` as:

$$
C_t = (W_t, M_t, G_t, A_t, E_t)
$$

where:

- $W_t$: deterministic external world model (`WorldState`)
- $M_t$: self-model / metacognitive state
- $G_t$: active goal state
- $A_t$: attention allocation state
- $E_t$: event and transition trace

The state transition is:

$$
C_{t+1} = \Phi(C_t, x_{t+1}, u_t)
$$

where:

- $x_{t+1}$ is an external signal
- $u_t$ is an internal deliberation act

Two cases:

1. External update:

$$
C_{t+1} = \Phi(C_t, x_{t+1}, \emptyset)
$$

2. Idle deliberation:

$$
C_{t+1} = \Phi(C_t, \emptyset, u_t)
$$

## Determinism and Path Dependence

The architecture distinguishes three objects:

1. `WorldState`:
   Deterministic given the same structured inputs and simulator settings.

2. `SelfModel`:
   Path-dependent. The final state may differ across runs if timing, ordering, or trace history differ.

3. `Trace`:
   Fully reproducible. Replaying the same trace must reconstruct the same final `ConsciousState`.

This gives the correct invariant:

```text
replay(trace) == C_T
```

rather than:

```text
same raw history == same C_T
```

## Core Separation of Roles

### Structured reasoning layer

Primary stateful reasoning must remain in:

- `WorldState`
- `KnowledgeGraph`
- `Reconciler`
- new `SelfModelGraph`
- new `ConsciousnessEngine`

### LLM layer

LLMs are restricted to read-only interpretation tasks:

- unstructured signal -> structured candidate signal
- structured state -> human-readable identity snapshot
- trace slice -> human-readable explanation

LLMs must not directly mutate `ConsciousState`.

## Proposed Runtime Components

### 1. `ConsciousState`

New top-level state container:

```text
ConsciousState
  world_ref
  self_model_ref
  goal_state
  attention_state
  trace_state
  runtime_metadata
```

Recommended location:

- `freeman/agent/consciousness.py`

### 2. `SelfModelGraph`

Structured metacognitive memory built on top of KG-compatible concepts.

Recommended location:

- `freeman/memory/selfmodel.py`

This can initially wrap the existing `KnowledgeGraph` and reserve a namespace for self-model nodes.

Write access must be restricted by design:

- only `ConsciousnessEngine` may mutate `SelfModelGraph`
- any other write path must raise `SelfModelAccessError`

This write guard prevents accidental identity drift from CLI helpers, ingestion adapters, or LLM-facing paths.

### 3. `IdleScheduler`

Deterministic scorer that decides whether the agent should perform internal deliberation.

Recommended location:

- `freeman/agent/consciousness.py`

Idle trigger:

$$
\text{idle\_score}_t =
w_{\mathrm{time}} z_{\mathrm{time},t}
+ w_{\mathrm{gap}} z_{\mathrm{gap},t}
+ w_{\mathrm{age}} z_{\mathrm{age},t}
+ w_{\mathrm{attn}} z_{\mathrm{attn},t}
$$

where:

- $z_{\mathrm{time},t}$ is normalized `time_since_last_update`
- $z_{\mathrm{gap},t}$ is normalized confidence gap
- $z_{\mathrm{age},t}$ is normalized active-hypothesis age pressure
- $z_{\mathrm{attn},t}$ is normalized attention deficit

Default weights:

- $w_{\mathrm{time}} = 0.25$
- $w_{\mathrm{gap}} = 0.35$
- $w_{\mathrm{age}} = 0.20$
- $w_{\mathrm{attn}} = 0.20$

If:

$$
\text{idle\_score}_t > \theta
$$

the engine schedules one internal act.

Recommended config:

```yaml
consciousness:
  idle_scheduler:
    threshold: 0.60
    weights:
      time_since_last_update: 0.25
      confidence_gap: 0.35
      hypothesis_age: 0.20
      attention_deficit: 0.20
```

### 4. `IdentityNarrator`

Read-only translator from `ConsciousState` to human narrative.

Recommended location:

- `freeman/llm/identity_narrator.py`

### 5. `ExplanationRenderer`

Read-only renderer from transition trace to explanation text.

Recommended location:

- `freeman/llm/explanation_renderer.py`

## Runtime Persistence and Resume

The agent also needs a production runtime layer that can:

- consume a local signal stream
- update state incrementally
- stop safely
- restart from the same development level

Persisting only the knowledge graph is not sufficient. To resume the agent faithfully, Freeman must persist the full runtime checkpoint:

$$
R_t = (C_t, O_t, S_t, P_t)
$$

where:

- $C_t$: current `ConsciousState`
- $O_t$: source offsets / cursors for each stream
- $S_t$: scheduler state, including pending obligations and next idle-review times
- $P_t$: process metadata, including runtime version and config hash

### Required persisted objects

Minimal persisted state:

- `WorldState` snapshot
- self-model snapshot
- attention / goal snapshot
- transition trace cursor
- source offsets
- `SignalMemory` snapshot
- pending obligations or deferred internal acts
- session metadata

### Checkpoint semantics

Use two persistence layers:

1. Event log:
   append-only stream of processed inputs and internal acts

2. Snapshot checkpoints:
   periodic materialized `ConsciousState` images for fast restart

Formally:

$$
\text{checkpoint}_k = \Psi(C_{t_k})
$$

and recovery is:

$$
C_T = \text{replay}(\text{checkpoint}_k, \text{events}_{t_k+1:T})
$$

This gives both:

- fast startup
- exact replay from a stable boundary

### Resume invariants

After a clean stop and restart:

- no already-processed signal is re-applied
- no pending internal act is lost
- the next `idle_score` is computed from restored state, not from a reset timer
- `ConsciousState` after replay matches the pre-shutdown state up to the last committed event

### Runtime components

Recommended new modules:

- `freeman/runtime/agent_runtime.py`
- `freeman/runtime/checkpoint.py`
- `freeman/runtime/stream.py`

Suggested responsibilities:

#### `AgentRuntime`

Single-process orchestrator:

```text
poll stream -> normalize signal -> update ConsciousState -> append event -> maybe checkpoint
```

No background daemon is required for MVP; a foreground loop with graceful shutdown is enough.

#### `CheckpointManager`

Responsible for:

- save checkpoint atomically
- load latest valid checkpoint
- rotate old checkpoints
- validate schema/runtime version compatibility

#### `StreamCursorStore`

Responsible for:

- storing per-source offsets
- marking events committed only after state checkpoint or event-log append succeeds
- providing `at-least-once` delivery semantics
- preventing duplicate state mutation via idempotent dedup keyed by `signal_id`

Delivery semantics:

- source delivery is `at-least-once`
- state mutation must be idempotent on `signal_id`

Minimal rules:

- every accepted signal must carry a stable `signal_id`
- committed `signal_id` values must survive restart
- a replayed source event with an already committed `signal_id` must not mutate `ConsciousState` a second time

### Local operation modes

The runtime should support three modes:

1. `oneshot`
   Process available signals once and exit.

2. `follow`
   Poll configured sources in a foreground loop.

3. `resume`
   Restore from latest checkpoint and continue consuming from saved offsets.

### CLI surface

Recommended commands:

- `freeman agent-start`
- `freeman agent-stop`
- `freeman agent-status`
- `freeman agent-resume`
- `freeman checkpoint-list`
- `freeman checkpoint-inspect`

For MVP, `agent-stop` can be implemented as graceful termination on `SIGINT` or `SIGTERM`, which forces:

1. flush pending event log
2. save checkpoint
3. persist cursor state
4. exit

### Storage layout

Suggested directory layout:

```text
data/
  kg_state.json
  sessions/
  runtime/
    checkpoints/
      checkpoint-000001.json
      checkpoint-000002.json
    cursors.json
    event_log.jsonl
    runtime_state.json
```

### Config extensions

Recommended additions to `config.yaml`:

```yaml
runtime:
  mode: "follow"
  poll_interval_seconds: 30
  checkpoint_every_n_events: 25
  checkpoint_every_n_seconds: 300
  runtime_path: "./data/runtime"
  resume_on_start: true
consciousness:
  idle_scheduler:
    threshold: 0.60
    weights:
      time_since_last_update: 0.25
      confidence_gap: 0.35
      hypothesis_age: 0.20
      attention_deficit: 0.20
```

### Acceptance tests for runtime

- start -> process signals -> stop -> resume preserves offsets
- replay from latest checkpoint plus event log reconstructs the same final state
- duplicate stream items are not re-applied after restart
- interrupted shutdown does not corrupt the last valid checkpoint
- upgrading LLM backend does not affect resumed structured state

## Self-Model Schema

The existing `self_observation` node type is too narrow. It should expand into a small controlled ontology.

### Node types

- `self_observation`
  - rolling forecast errors
  - calibration residuals
  - recent anomalies
- `self_capability`
  - domain competence estimates
  - task-specific reliability
  - known failure modes
- `self_uncertainty`
  - unresolved uncertainty sources
  - missing evidence
  - model instability flags
- `active_hypothesis`
  - current working explanations
  - contradiction status
  - age and support
- `goal_state`
  - active goals
  - urgency
  - blocking conditions
- `attention_focus`
  - current focus allocation
  - salience
  - neglected domains
- `identity_trait`
  - persistent reasoning preferences
  - policy constraints
  - communication profile

### Edge types

- `supports`
- `contradicts`
- `depends_on`
- `derived_from`
- `focuses_on`
- `serves_goal`
- `revises`

## Internal Deliberation Acts

Internal acts are deterministic operators over the self-model.

### `hypothesis_aging`

Decay stale hypotheses when no new support arrives:

$$
c_{h,t+1} = \lambda_h c_{h,t}
$$

where `lambda_h` depends on hypothesis age and recent evidence.

### `consistency_check`

Scan the self-model for incompatible node pairs or cycles and emit contradiction nodes or conflict edges.

### `attention_rebalance`

Recompute attention if current attention is misaligned with active goals or rising uncertainty:

$$
a_{d,t+1} \propto
w_g g_{d,t} +
w_u u_{d,t} +
w_e e_{d,t} -
w_s s_{d,t}
$$

where:

- $g_{d,t}$ is goal urgency
- $u_{d,t}$ is uncertainty
- $e_{d,t}$ is recent error pressure
- $s_{d,t}$ is current saturation

### `capability_review`

Update domain competence from rolling errors:

$$
\text{capability}_{d,t} = \sigma(\alpha - \beta \cdot \text{MAE}_{d,t})
$$

This turns forecast quality into a stable metacognitive estimate.

### `trait_consolidation`

`identity_trait` must be grounded in observed performance rather than free-form narrative residue.

For MVP, use MAE-driven consolidation:

$$
\text{trait\_support}_{k,t+1}
=
\lambda_k \text{trait\_support}_{k,t}
+ \eta_k \mathbf{1}\{\text{pattern}_k \text{ observed}\}
- \beta_k \Delta \text{MAE}_{k,t}^{+}
$$

where:

- `pattern_k` is a repeated reasoning pattern associated with trait `k`
- $\Delta \text{MAE}_{k,t}^{+}$ is the positive deterioration in forecast error associated with that trait

Interpretation:

- traits repeatedly associated with better calibration are consolidated
- traits associated with persistent MAE deterioration are weakened or moved to review

## Transition Trace

Every change to `ConsciousState` must write a trace event.

Minimal event schema:

```json
{
  "event_id": "trace:...",
  "timestamp": "2026-04-11T12:00:00Z",
  "transition_type": "external|internal",
  "trigger_type": "signal|idle_threshold|manual",
  "operator": "hypothesis_aging",
  "pre_state_ref": "state:...",
  "post_state_ref": "state:...",
  "input_refs": ["signal:...", "node:..."],
  "diff": {},
  "rationale": "deterministic explanation string"
}
```

Required invariants:

- every operator writes a trace event
- every trace event has a deterministic diff
- replaying trace events reconstructs the same final state

## Integration with Existing Freeman Flow

Current high-level pipeline:

```text
signal -> compile/update world -> verify -> simulate -> score -> write KG -> reconcile
```

Proposed pipeline:

```text
signal
-> structured interpretation
-> world update
-> KG/reconciliation update
-> self-model update
-> attention recompute
-> optional idle check
-> optional internal deliberation act
-> trace write
-> optional LLM narration
```

This should be added after the existing `AnalysisPipeline` write/reconcile stage rather than replacing it.

## Proposed File Plan

Phase 1 should add the following files:

- `freeman/agent/consciousness.py`
- `freeman/memory/selfmodel.py`
- `freeman/llm/identity_narrator.py`
- `freeman/llm/explanation_renderer.py`
- `docs/CONSCIOUSNESS_ARCHITECTURE.md`

Likely integration points:

- `freeman/agent/analysispipeline.py`
- `freeman/memory/reconciler.py`
- `freeman/interface/cli.py`
- `freeman/docs/ARCHITECTURE.md`

## MVP Scope

### Stage 1: structure only

Goal:
Introduce the state container and self-model schema without changing behavior.

Deliverables:

- `ConsciousState` dataclass
- `SelfModelNode` / `SelfModelEdge` dataclasses
- namespace convention for self-model nodes in KG
- trace event schema

### Stage 2: deterministic self-model updates

Goal:
Generate self-model diffs after every pipeline run.

Deliverables:

- `capability_review`
- `attention_rebalance`
- `trait_consolidation`
- self-model snapshot persistence
- integration into analysis pipeline

### Stage 3: idle deliberation

Goal:
Allow internal non-signal-triggered state evolution.

Deliverables:

- `IdleScheduler`
- `hypothesis_aging`
- `consistency_check`
- single-step internal act after pipeline completion when threshold is exceeded

No background thread is needed for MVP.

### Stage 4: read-only narration

Goal:
Render the structured state for human inspection without moving cognition into the LLM.

Deliverables:

- `IdentityNarrator`
- `ExplanationRenderer`
- CLI/API endpoint for identity snapshots

## Acceptance Tests

### Structural tests

- `ConsciousState` serializes and deserializes cleanly
- self-model nodes persist through KG save/load
- trace events preserve deterministic diffs
- non-engine writes to `SelfModelGraph` raise `SelfModelAccessError`

### Reproducibility tests

- replaying the same trace reconstructs the same final state
- swapping LLM backend does not change `ConsciousState`
- narrator output may differ stylistically, but underlying state references do not
- duplicated source delivery does not duplicate state mutation when `signal_id` has already been committed

### Behavioral tests

- rising forecast MAE lowers `self_capability`
- rising uncertainty or neglected goals increase attention weight
- stale unsupported hypotheses decay
- contradictory nodes trigger `consistency_check` output
- `trait_consolidation` weakens traits associated with persistent MAE deterioration

### Interface tests

- `freeman identity` returns structured self-model snapshot
- explanation renderer can explain a concrete trace slice

## Non-Goals for MVP

Do not add these yet:

- continuous background threads
- autonomous tool execution from self-model alone
- free-form LLM reflection that mutates identity
- unrestricted narrative memory as a source of truth

## Implementation Order

Recommended execution order:

1. `selfmodel.py`
2. `ConsciousState` in `consciousness.py`
3. trace schema and replay helpers
4. pipeline integration hooks
5. `IdleScheduler`
6. read-only LLM narrators

This order minimizes risk because it introduces structure first, operators second, and language-facing components last.
