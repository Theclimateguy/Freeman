# Agent Roles

Freeman `3.3.1` splits the agent layer into role-scoped workers that read only
the trail states they are supposed to handle.

## Trail Chain

```text
ingestor -> [ingest trail] -> repairer
repairer -> [repair trail] -> planner
planner  -> [read_plan trail] -> narrator
narrator -> [llm_propose trail] -> verifier
verifier -> [verified trail] -> planner
```

## Code Anchors

- Role state lives in `freeman.agent.consciousness.ConsciousState.agent_role`.
- Role permissions are enforced by `freeman.agent.consciousness.ConsciousnessEngine._require_permission(...)`.
- Role trail scopes live in `freeman.agent.domainregistry.ROLE_TRAIL_SCOPE` and `trail_scope_for_role(...)`.
- Scheduler routing is enforced by `freeman.agent.attentionscheduler.AttentionScheduler.eligible_tasks(trail_scope=...)`.
- Trail labels are stored on node metadata as `trail_type` / `trail_intensity` in `freeman.memory.knowledgegraph.KGNode.metadata`.

## Per-Role Specs

- [ingestor](ingestor.md)
- [repairer](repairer.md)
- [planner](planner.md)
- [narrator](narrator.md)
- [verifier](verifier.md)
