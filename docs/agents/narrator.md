# Narrator

## Role

Translate planner state into explicit proposal text or structured hypothesis payloads that can be checked by the verifier instead of written directly into the canonical graph.

## Reads from

- Graph layer: nodes with `trail_type = "read_plan"`.
- Trail filter: `freeman.agent.domainregistry.trail_scope_for_role("narrator")`.

## Writes to

- Graph layer: proposal-layer or shadow-graph hypothesis payloads intended for verification.
- Trail left behind: `llm_propose`.

## Triggers

- Planner output becoming stable enough to externalize as a proposal.
- LLM-facing rendering paths such as `freeman.llm.identity_narrator` and `freeman.llm.explanation_renderer`.

## Idle Behaviour

For this role, `IdleScheduler.threshold` should act as a cooldown guard: if the planner frontier was just read, the narrator should avoid repeatedly rephrasing the same proposal until new repaired or verified evidence arrives.

## Handoff

- Next role: `verifier`.
- Expected chain: `read_plan` node -> narrative/LLM proposal -> `llm_propose` trail -> verifier queue.

## Key Methods

- `freeman.agent.consciousness.ConsciousState.agent_role = "narrator"`
- `freeman.agent.domainregistry.trail_scope_for_role("narrator")`
- `freeman.llm.identity_narrator.IdentityNarrator.structured_snapshot(...)`
- `freeman.llm.identity_narrator.IdentityNarrator.render(...)`
- `freeman.llm.explanation_renderer.ExplanationRenderer.explain_trace(...)`
