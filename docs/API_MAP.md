# Freeman API Map

> Actual note: this map describes the current code in `main`. Legacy research harnesses are listed explicitly as legacy where relevant.

This is the implementation-facing map of modules, classes, and primary functions exposed by the current codebase.

## Top-Level Packages

### `freeman.core`

- State and world model:
  - `WorldGraph`
  - `WorldState`
  - `OutcomeRegistry`
  - `WorldGraph.apply_shocks()`
- Types:
  - `Actor`
  - `Resource`
  - `Relation`
  - `Outcome`
  - `ParameterVector`
  - `CausalEdge`
  - `Policy`
  - `Violation`
- Evolution:
  - `EvolutionRegistry`
  - `get_operator()`
  - `LinearTransition`
  - `StockFlowTransition`
  - `LogisticGrowthTransition`
  - `ThresholdTransition`
  - `CoupledTransition`
- Transition/scoring:
  - `step_world()`
  - `raw_outcome_scores()`
  - `regime_shift_matches()`
  - `effective_edge_weight()`
  - `softmax_distribution()`
  - `score_outcomes()`
  - `compute_confidence()`
- Multi-domain:
  - `SharedResourceBus`
  - `MultiDomainWorld`
  - `MultiDomainSimResult`
- v0.2:
  - `CompileCandidate`
  - `HistoricalFitScore`
  - `OperatorFitReport`
  - `CompileValidationReport`
  - `CompileValidator`
  - `CompileValidator.validate()`
  - `CompileValidator.compare_operators()`
  - `CompileValidator.fit_outcome_weights()`
  - `ParameterDistribution`
  - `ScenarioSample`
  - `OutcomeDistribution`
  - `ConfidenceReport`
  - `UncertaintyEngine`

### `freeman.verifier`

- Low-level checks:
  - `level0_check()`
  - `level1_check()`
  - `level2_check()`
- Jacobian utilities:
  - `compute_jacobian()`
  - `spectral_radius()`
- Fixed point:
  - `compute_corrections()`
  - `apply_corrections()`
  - `iterate_fixed_point()`
  - `find_fixed_point()`
  - `FixedPointResult`
- Aggregate API:
  - `VerifierConfig`
  - `Verifier`
  - `VerificationReport`

### `freeman.memory`

- Knowledge graph:
  - `KGNode`
  - `KGEdge`
  - `KnowledgeGraph`
  - `KnowledgeGraph.explain_causal_path()`
  - `semantic_query()`
- Semantic index:
  - `KGVectorStore`
- Session log:
  - `KGDelta`
  - `AttentionStep`
  - `TaskRecord`
  - `SessionLog`
- Reconciliation:
  - `ConfidenceThresholds`
  - `ReconciliationResult`
  - `Reconciler`
  - `Reconciler.verify_causal_path()`
  - `update_self_model()`
- Self-model:
  - `SelfModelAccessError`
  - `SelfModelNode`
  - `SelfModelEdge`
  - `SelfModelGraph`
- Epistemic memory:
  - `EpistemicLog`
  - `EpistemicLog.domain_mae()`
  - `EpistemicLog.domain_weight()`

### `freeman.agent`

- Domain orchestration:
  - `DomainTemplate`
  - `DomainTemplateRegistry`
- Pipeline:
  - `AnalysisPipeline`
  - `AnalysisPipelineConfig`
  - `AnalysisPipelineResult`
  - `ForecastSummary`
  - `CausalStep`
  - `ForecastExplanation`
  - `AnalysisPipeline.list_forecasts()`
  - `AnalysisPipeline.explain_forecast()`
  - `AnalysisPipeline.update()`
  - `AnalysisPipeline._export_causal_edges()`
- Counterfactual planning:
  - `PolicyEvaluator`
  - `PolicyEvalResult`
- Dynamic calibration:
  - `ParameterEstimator`
- Forecasts:
  - `Forecast`
  - `Forecast.causal_path`
  - `ForecastRegistry`
- Signals:
  - `Signal`
  - `SignalRecord`
  - `SignalMemory`
  - `ShockClassification`
  - `SignalTrigger`
  - `ManualSignalSource`
  - `RSSSignalSource`
  - `TavilySignalSource`
  - `SignalIngestionEngine`
- Attention:
  - `ForecastDebt`
  - `ConflictDebt`
  - `AnomalyDebt`
  - `InterestNormalizer`
  - `ObligationQueue`
  - `AttentionTask`
  - `AttentionDecision`
  - `AttentionScheduler`
- Proactive interface:
  - `ProactiveEvent`
  - `ProactiveEmitter`
- Cost governance:
  - `CostEstimate`
  - `BudgetPolicy`
  - `BudgetDecision`
  - `CostModel`
- Consciousness:
  - `TraceEvent`
  - `ConsciousState`
  - `ConsciousnessEngine`
  - `IdleScheduler`

### `freeman.interface`

- CLI:
  - `build_parser()`
  - `main()`
  - `identity`
  - `explain --trace-id <id>`
- REST:
  - `InterfaceAPI`
  - `run_server()`
- Export:
  - `KnowledgeGraphExporter`
- Override and diff:
  - `OverrideAuditEntry`
  - `DomainOverrideRecord`
  - `ModelOverrideAPI`
  - `DiffEntry`
  - `SimulationDiffReport`
  - `build_simulation_diff()`
  - `export_simulation_diff()`

### `freeman.domain`

- `DomainCompiler`
- `DomainRegistry`
- schema validation helpers in `schema.py`

### `freeman.game`

- `SimConfig`
- `GameRunner`
- `GameRunner.prepare()`
- `GameRunner.run_prepared()`
- `SimResult`

### `freeman.llm`

- `EmbeddingAdapter`
- `DeterministicEmbeddingAdapter`
- `HashingEmbeddingAdapter`
- `OpenAIEmbeddingClient`
- `OllamaEmbeddingClient`
- `OllamaChatClient`
- `DeepSeekChatClient`
- `DeepSeekFreemanOrchestrator`
- `IdentityNarrator`
- `ExplanationRenderer`
- `LLMDrivenSimulationRun`

### `freeman.runtime`

- `AgentRuntime`
- `CheckpointManager`
- `EventLog`
- `StreamCursorStore`
- `RuntimePaths`
- `RuntimeStorage`
- `BootstrapResult`
- `RuntimeContext`
- `SignalResult`
- `LoopSummary`
- `stream_runtime.main()` (generic daemon-like long-run loop)

### `scripts`

- Live evaluation:
  - `scripts/run_real_llm_e2e.py`
  - legacy pre-daemon evaluation harness
  - real-LLM signal classification
  - template shock inference
  - simulation interpretation
  - memory-only follow-up answers over KG retrieval
- FAAB benchmark:
  - `scripts/benchmark_faab/generate_dummy_dataset.py`
  - `scripts/benchmark_faab/run_benchmark.py`
  - `scripts/benchmark_faab/runner.py`
  - `BenchmarkRunner`
  - `RunnerConfig`
  - `MODE_A_FULL`
  - `MODE_B_AMNESIC`
  - `MODE_C_HASH`
  - `MODE_D_LLMONLY`

### `tests`

- Behavioral harness:
  - `AgentHarness`
  - `CycleResult`
- Connector package coverage:
  - `tests/test_connectors.py`
  - factory + `rss` + `http_json` + `web` source adapters
- Replay fixtures:
  - `tests/fixtures/signals/water_shock.jsonl`
  - `tests/fixtures/signals/japan_debt_shock.jsonl`
  - `tests/fixtures/signals/null_stream.jsonl`

## CLI Map

Command:
- `python -m freeman.interface.cli run <schema_path> [--policies-path <path>]`
  - compile schema, run the analysis pipeline, print simulation JSON
- `python -m freeman.interface.cli identity [--narrative]`
  - print structured consciousness snapshot; optional LLM narrative projection
- `python -m freeman.interface.cli explain --trace-id <id>`
  - render causal explanation for one trace event from runtime event log/checkpoint
- `python -m freeman.interface.cli query [--text ...] [--status ...] [--node-type ...] [--min-confidence ...]`
  - query KG nodes
- `python -m freeman.interface.cli export-kg <html|json-ld|dot> <output_path>`
  - export KG in one of the supported formats
- `python -m freeman.interface.cli status`
  - show KG counts and storage path
- `python -m freeman.interface.cli reconcile <session_log_path>`
  - reconcile one saved session log into the KG
- `python -m freeman.interface.cli kg-archive [--node-id <id>] [--reason <reason>]`
  - archive one node or auto-archive all low-confidence nodes
- `freeman --config-path <config> kg-reindex [--batch-size <n>] [--use-stub-embeddings]`
  - re-embed legacy nodes without vectors and sync them into ChromaDB using the configured embedding backend
- `python -m freeman.interface.cli override-param <world_path> <param_path> <value> [--output-path <path>]`
  - apply one parameter override to a world snapshot
- `python -m freeman.interface.cli override-sign <world_path> <edge_id> <expected_sign> [--output-path <path>]`
  - override one causal-edge sign in a world snapshot
- `python -m freeman.interface.cli rerun-domain <world_path> [--max-steps <n>] [--output-path <path>]`
  - rerun a world snapshot after overrides
- `python -m freeman.interface.cli diff-domain <baseline_path> <current_path> [--output-path <path>]`
  - export structured differences between two snapshots or simulation payloads

Runtime command:

- `python -m freeman.runtime.stream_runtime --config-path config.yaml --schema-path freeman/domain/profiles/gim15.json --hours 8 --poll-seconds 600 --analysis-interval-seconds 1.0 --resume --model auto`
  - generic long local signal ingestion with deterministic checkpoint/resume, persistent pending queue, monotonic `runtime_step`, due-forecast verification, causal-path verification, synchronous consciousness refresh, and configurable source adapters
- `python -m freeman.runtime.stream_runtime --config-path config.climate.yaml --hours 8 --poll-seconds 600 --analysis-interval-seconds 1.0 --resume --model auto`
  - the same daemon runtime with a climate-oriented example config only; this is not a separate runtime implementation
- `python -m freeman.runtime.stream_runtime --config-path config.yaml --bootstrap-mode llm_synthesize --domain-brief-path <brief.md> --hours 8 --poll-seconds 600 --analysis-interval-seconds 1.0 --resume --model auto`
  - synthesize a verifier-repaired Freeman schema from a natural-language brief, persist the bootstrap package and `bootstrap_attempts`, then run the same daemon loop
- `python -m freeman.runtime.stream_runtime --config-path config.yaml --query forecasts [--status <pending|verified|expired>]`
  - load saved runtime artifacts and print compact forecast summaries without starting the daemon loop
- `python -m freeman.runtime.stream_runtime --config-path config.yaml --query explain --forecast-id <id>`
  - render one forecast as a human-readable causal chain with scalar error and path confirmation / refutation
- `python -m freeman.runtime.stream_runtime --config-path config.yaml --query anomalies`
  - print saved `anomaly_candidate` nodes together with escalated `ontology_gap` traits
- `python -m freeman.runtime.stream_runtime --config-path config.yaml --query causal --limit 20`
  - print recent exported causal edges (`causes`, `propagates_to`, `threshold_exceeded`)

## Benchmark Map

Command:
- `python scripts/benchmark_faab/run_benchmark.py --dataset <cases.jsonl> --output-dir <dir> [--state-time-decay 0.5]`
  - run longitudinal FAAB evaluation across all benchmark modes
- `python scripts/benchmark_faab/generate_dummy_dataset.py`
  - emit the default four-case dummy benchmark dataset
- `scripts/benchmark_faab/metrics.py`
  - `brier_score()`

Outputs:
- `metrics.csv`
  - one row per `(case_id, mode)`
  - includes `t0_brier_score` and `t1_brier_score`
- `summary.json`
  - full serialized predictions and metadata
- `traces/*.json`
  - per-case / per-mode prompt and prediction traces
- `kg_snapshots/*.json`
  - persisted KG state after the `T1` step

## Universal Update Map

Core formula:
- `ParameterVector`
  - `outcome_modifiers`
  - `shock_decay`
  - `edge_weight_deltas`
  - `rationale`

Agent update path:
1. `ParameterEstimator.estimate(previous_world, new_signal_text)`
2. `AnalysisPipeline.update(previous_world, parameter_vector, ...)`
3. KG analysis node stores `metadata["parameter_vector"]`

Affected simulator surfaces:
- `WorldGraph.apply_shocks()`
- `raw_outcome_scores()`
- resource coupling inside `freeman.core.evolution`
- actor-state coupling inside `step_world()`

## REST Map

### `GET /status`

Returns:

- `knowledge_graph_path`
- `node_count`
- `edge_count`
- `status_counts`

### `POST /query`

Accepted fields:

- `text`
- `status`
- `node_type`
- `min_confidence`

Returns:

- `matches`
- `count`

### `PATCH /domain/{id}/params`

Body:

- `overrides`
- optional `actor`

### `PATCH /domain/{id}/edges/{edge_id}`

Body:

- `expected_sign`
- optional `actor`

### `POST /domain/{id}/rerun`

Returns the rerun simulation payload for the current human-adjusted version.

### `GET /domain/{id}/diff`

Returns machine-vs-current diff plus override audit history.

## Core Execution APIs

### Deterministic simulation

```python
from freeman.domain.compiler import DomainCompiler
from freeman.game.runner import GameRunner, SimConfig

world = DomainCompiler().compile(schema)
result = GameRunner(SimConfig(max_steps=30)).run(world, policies=[])
```

### Aggregate verification

```python
from freeman.verifier.verifier import Verifier

report = Verifier().run(world, levels=(1, 2))
```

### Knowledge graph update

```python
from freeman.memory.knowledgegraph import KnowledgeGraph
from freeman.memory.reconciler import Reconciler
from freeman.memory.sessionlog import SessionLog

kg = KnowledgeGraph()
session = SessionLog(session_id="demo")
result = Reconciler().reconcile(kg, session)
```

### Forecast tracking

```python
from datetime import datetime, timezone

from freeman.agent.forecastregistry import Forecast, ForecastRegistry

registry = ForecastRegistry(auto_load=False, auto_save=False)
registry.record(
    Forecast(
        forecast_id="water:5:scarcity",
        domain_id="water",
        outcome_id="scarcity",
        predicted_prob=0.62,
        session_id="session-1",
        horizon_steps=5,
        created_at=datetime.now(timezone.utc),
        created_step=5,
    )
)
due = registry.due(current_step=10)
verified = registry.verify("water:5:scarcity", actual_prob=0.55, verified_at=datetime.now(timezone.utc))
```

### Replay-driven agent cycle

```python
from harness import AgentHarness

result = AgentHarness(schema, replay_signals, enable_emitter=True).run_cycle()
print(result.decisions, result.proactive_events)
```

### End-to-end pipeline

```python
from freeman.agent import AnalysisPipeline, PolicyEvaluator
from freeman.game.runner import SimConfig

pipeline = AnalysisPipeline()
result = pipeline.run(schema, policies=[])

planner = PolicyEvaluator(sim_config=SimConfig(max_steps=8))
ranked = pipeline.run(
    schema,
    policy_evaluator=planner,
    candidate_policies=[policy_a, policy_b, policy_c],
).policy_ranking
```

### Compile validation

```python
from freeman.core.compilevalidator import CompileCandidate, CompileValidator

validator = CompileValidator()
report = validator.validate_candidates(candidates, historical_data=history)
```

### Uncertainty propagation

```python
from freeman.core.uncertainty import ParameterDistribution, UncertaintyEngine

engine = UncertaintyEngine()
distribution = engine.monte_carlo(world, distributions, monte_carlo_samples=100)
confidence = engine.confidence_from_variance(distribution)
```

### Human override workflow

```python
from freeman.interface.modeloverride import ModelOverrideAPI

api = ModelOverrideAPI()
api.register_domain(world.domain_id, world)
api.patch_params(world.domain_id, {"resources.x.value": 12.0})
api.patch_edge(world.domain_id, "x->y", "+")
rerun = api.rerun_domain(world.domain_id)
diff = api.get_diff(world.domain_id)
```

## Config Map

`config.yaml` currently exposes:

- `freeman.default_evolution`
- `freeman.level0_hard_stop`
- `freeman.epsilon`
- `freeman.sign_epsilon`
- `sim.max_steps`
- `sim.dt`
- `sim.level2_check_every`
- `sim.level2_shock_delta`
- `sim.stop_on_hard_level2`
- `sim.convergence_check_steps`
- `sim.convergence_epsilon`
- `sim.fixed_point_max_iter`
- `sim.fixed_point_alpha`
- `sim.seed`
- `multiworld.sync_mode`
- `agent.stream_keywords`
- `agent.stream_filter.min_relevance_score`
- `agent.stream_filter.min_keyword_matches`
- `agent.stream_filter.agent_min_relevance_score`
- `agent.bootstrap.max_retries`
- `agent.bootstrap.trial_steps`
- `memory.backend`
- `memory.json_path`
- `memory.reconciler.merge_threshold`
- `memory.reconciler.compaction_interval`

## Test Map

- Core foundation:
  - [tests/test_core_phase1.py](../tests/test_core_phase1.py)
- Verifier:
  - [tests/test_verifier.py](../tests/test_verifier.py)
  - [tests/test_fixedpoint.py](../tests/test_fixedpoint.py)
- Memory:
  - [tests/test_reconciler.py](../tests/test_reconciler.py)
- Agent:
  - [tests/test_attention.py](../tests/test_attention.py)
- v0.2:
  - [tests/test_compilevalidator.py](../tests/test_compilevalidator.py)
  - [tests/test_uncertainty.py](../tests/test_uncertainty.py)
  - [tests/test_costmodel.py](../tests/test_costmodel.py)
  - [tests/test_humanoverride.py](../tests/test_humanoverride.py)
- Integration:
  - [tests/test_integration.py](../tests/test_integration.py)
