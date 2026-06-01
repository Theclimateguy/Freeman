# Verifier

## Role

Confirm or reject proposed hypotheses against the canonical world state, realized outcomes, and stored causal paths before planner state is allowed to persist as trustworthy.

## Reads from

- Graph layer: nodes with `trail_type = "llm_propose"`.
- Trail filter: `freeman.agent.domainregistry.trail_scope_for_role("verifier")`.

## Writes to

- Graph layer: hypothesis verification status, refreshed forecast outcomes, and verifier-approved state transitions.
- Trail left behind: `verified`.

## Triggers

- Due forecast checks in `freeman.runtime.stream_runtime._verify_due_forecasts(...)`.
- Causal-path confirmation in `freeman.memory.reconciler.Reconciler.verify_causal_path(...)`.

## Idle Behaviour

For this role, crossing `IdleScheduler.threshold` means stale proposed hypotheses should be resolved or downgraded so the planner does not continue to reason over unverified branches.

## Handoff

- Next role: `planner`.
- Expected chain: `llm_propose` node -> verification pass -> `verified` trail -> planner refresh.

## Key Methods

- `freeman.runtime.stream_runtime._verify_due_forecasts(...)`
- `freeman.agent.forecastregistry.ForecastRegistry.due(...)`
- `freeman.memory.reconciler.Reconciler.verify_causal_path(...)`
- `freeman.agent.consciousness.ConsciousnessEngine.refresh_after_epistemic_update(...)`
