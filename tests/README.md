# Integration and Behavior Benchmarks

## Japan Debt Scenario

`tests/test_debt_scenario.py` adds a Freeman-side integration benchmark that mirrors the qualitative GIM debt-crisis chain:

- `japan_debt_ratio -> japan_gdp_growth` is negative
- `japan_gdp_growth -> japan_political_stability` is positive
- `japan_political_stability -> global_risk_appetite` is positive
- `japan_debt_ratio -> global_risk_appetite` is negative

The assertions are chosen to match the expected GIM-style qualitative outcome:

- debt ratio rises over the run
- `debt_crisis` probability increases over time
- the simulator completes all 30 steps
- no hard verifier violations occur

This repository does not vendor the original GIM implementation, so the benchmark is documented as a directional compatibility contract rather than a byte-for-byte trajectory match.

## Replay-Driven Agent Harness

`tests/harness.py` provides a deterministic `AgentHarness` that replays signal streams through:

- `SignalIngestionEngine`
- `SignalMemory`
- `AttentionScheduler` with `ObligationQueue`
- `AnalysisPipeline`
- optional `ProactiveEmitter`

The replay fixtures live in `tests/fixtures/signals/`:

- `water_shock.jsonl`
- `japan_debt_shock.jsonl`
- `null_stream.jsonl`

`tests/test_agent_behavior.py` asserts the current behavioral contracts:

- severe shock streams trigger at least one analysis decision
- KG state grows after a real shock cycle
- null streams stay in `WATCH`
- unresolved obligations can dominate a fresher task
- hard verifier violations produce proactive `alert` events
