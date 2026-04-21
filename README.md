# Freeman Hive Mind

This branch (`hive_mind`) contains only the multi-agent coordination extension for Freeman.

For the full project documentation (installation, runtime, CLI/API, architecture, and baseline behavior), use the `main` branch README:

- [Freeman README (main)](https://github.com/Theclimateguy/Freeman/blob/main/README.md)

## Hive Mind Scope

Implemented in this branch:

- cooperative KG node locking: `KnowledgeGraph.try_lock(...)`, `KnowledgeGraph.unlock(...)`
- causal pheromone trail on edges: `KGEdge.trail_weight`
- trail deposit after successful updates: `KnowledgeGraph.deposit_trail(...)`
- trail-aware attention: `AttentionTask.trail_weight` in scheduler interest components
- trail evaporation in reconciler: `trail <- trail * exp(-gamma)` for `causes`/`propagates_to`
- tests covering lock/trail/scheduler/evaporation behavior

## Changed Files (vs `main`)

- `freeman/memory/knowledgegraph.py`
- `freeman/agent/attentionscheduler.py`
- `freeman/memory/reconciler.py`
- `freeman/agent/analysispipeline.py`
- `tests/test_hive_mind.py`
- `docs/ARCHITECTURE.md`
