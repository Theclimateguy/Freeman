# Planner

## Role

Read repaired or verified state and maintain the shadow planning graph that expresses goals, active hypotheses, and attention allocation.

## Reads from

- Graph layer: nodes with `trail_type in {"verified", "repair"}`.
- Trail filter: `freeman.agent.domainregistry.trail_scope_for_role("planner")`.

## Writes to

- Graph layer: self-model / shadow-graph nodes such as `goal_state`, `active_hypothesis`, `identity_trait`, and `attention_focus`.
- Trail left behind: `read_plan`.

## Triggers

- Post-pipeline updates in `freeman.agent.analysispipeline.AnalysisPipeline`.
- Epistemic refresh after forecast verification through `ConsciousnessEngine.refresh_after_epistemic_update(...)`.

## Idle Behaviour

For this role, `IdleScheduler.threshold` controls when the planner should rebalance active hypotheses, age stale beliefs, and refresh attention weights instead of waiting for new external signals.

## Handoff

- Next role: `narrator`.
- Expected chain: repaired/verified node -> planner shadow state -> `read_plan` trail -> narrator proposal pass.

## Key Methods

- `freeman.agent.consciousness.ConsciousnessEngine._project_goal_state(...)`
- `freeman.agent.consciousness.ConsciousnessEngine._project_active_hypotheses(...)`
- `freeman.agent.consciousness.ConsciousnessEngine._attention_rebalance(...)`
- `freeman.agent.consciousness.ConsciousnessEngine._apply_diff(...)`
