# Repairer

## Role

Normalize fresh or LLM-proposed graph state so the canonical model can be updated without schema drift or unresolved ontology gaps.

## Reads from

- Graph layer: nodes with `trail_type in {"ingest", "llm_propose"}`.
- Trail filter: `freeman.agent.domainregistry.trail_scope_for_role("repairer")`.

## Writes to

- Graph layer: repaired canonical nodes, ontology-gap review state, and repair-trigger metadata.
- Trail left behind: `repair`.

## Triggers

- Accumulated ontology-gap traits inside `freeman.agent.consciousness.ConsciousnessEngine._pending_ontology_gap_traits()`.
- Reconciliation / canonical update pressure in `freeman.memory.reconciler.Reconciler`.

## Idle Behaviour

For this role, crossing `IdleScheduler.threshold` means the repairer should spend slack budget on anomaly review and ontology-gap consolidation before planner work continues.

## Handoff

- Next role: `planner`.
- Expected chain: `ingest` or `llm_propose` node -> repair pass -> `repair` trail -> planner frontier.

## Key Methods

- `freeman.agent.consciousness.ConsciousnessEngine._emit_ontology_repair_request(...)`
- `freeman.agent.consciousness.ConsciousnessEngine._mark_anomaly_candidate_reviewed(...)`
- `freeman.memory.reconciler.Reconciler.reconcile(...)`
- `freeman.memory.knowledgegraph.KnowledgeGraph.update_node(...)`
