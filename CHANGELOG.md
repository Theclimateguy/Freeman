# Changelog

## v1.1.0

This release brings the repository to a working USIM-AGENT baseline covering the v0.1 and v0.2 technical specifications.

### Added

- `WorldGraph` / `OutcomeRegistry` compatibility layer on top of the existing `WorldState`-based simulator.
- `EvolutionRegistry` and explicit additive operator view.
- Aggregate verifier API, fixed-point result model, and spectral-radius guard.
- Knowledge graph, session log, reconciler, confidence thresholds, persistence, and graph export.
- Agent layer: analysis pipeline, signal ingestion, UCB scheduler, cost governance.
- Interface layer: CLI, minimal REST API, KG export, model override API, simulation diff.
- v0.2 modules: compile validation, ensemble sign consensus, historical fit scoring, uncertainty propagation.
- Full end-to-end integration coverage.
- Repository documentation:
  - architecture
  - API map
  - progress log

### Changed

- `pytest tests/` now works directly in the repo without manual `PYTHONPATH` manipulation.
- Default memory backend is explicitly configured as NetworkX + JSON.
- Packaging metadata now points to `README.md`.

### Validation

- `pytest tests/` -> `44 passed`
