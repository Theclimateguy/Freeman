# Ingestor

## Role

Absorb raw external signals and create the first machine-readable candidate node before any repair, planning, or verification occurs.

## Reads from

- Graph layer: untrailed nodes only (`trail_type = None`).
- Trail filter: `freeman.agent.domainregistry.trail_scope_for_role("ingestor") -> (None,)`.

## Writes to

- Graph layer: candidate/raw KG nodes, typically signal- or anomaly-adjacent payloads.
- Trigger metadata: contradictory same-topic/entity signals are marked with `conflict_score`, `conflict_reason`, and `conflicts_with` before downstream budgeting.
- Trail left behind: `ingest`.

## Triggers

- New upstream signal arrival through `freeman.agent.signalingestion.SignalIngestionEngine`.
- Frontier selection through `freeman.agent.attentionscheduler.AttentionScheduler.eligible_tasks(...)` with the ingestor trail scope.

## Idle Behaviour

For this role, `IdleScheduler.threshold` should be interpreted conservatively: if idle score is high, the ingestor should prefer clearing stale raw nodes rather than inventing new hypotheses.

## Handoff

- Next role: `repairer`.
- Expected chain: raw node -> `ingest` trail -> repair queue.

## Key Methods

- `freeman.agent.signalingestion.SignalIngestionEngine.ingest(...)`
- `freeman.agent.attentionscheduler.AttentionScheduler.eligible_tasks(...)`
- `freeman.memory.knowledgegraph.KnowledgeGraph.add_node(...)`
- `freeman.agent.consciousness.ConsciousnessEngine._require_permission("candidate_node", ...)`
