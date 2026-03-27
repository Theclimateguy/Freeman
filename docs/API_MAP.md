# Freeman API Map

This is the implementation-facing map of modules, classes, and primary functions exposed by the current codebase.

## Top-Level Packages

### `freeman.core`

- State and world model:
  - `WorldGraph`
  - `WorldState`
  - `OutcomeRegistry`
- Types:
  - `Actor`
  - `Resource`
  - `Relation`
  - `Outcome`
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
  - `CompileValidationReport`
  - `CompileValidator`
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

### `freeman.agent`

- Domain orchestration:
  - `DomainTemplate`
  - `DomainTemplateRegistry`
- Pipeline:
  - `AnalysisPipeline`
  - `AnalysisPipelineConfig`
  - `AnalysisPipelineResult`
- Signals:
  - `Signal`
  - `ShockClassification`
  - `SignalTrigger`
  - `ManualSignalSource`
  - `RSSSignalSource`
  - `TavilySignalSource`
  - `SignalIngestionEngine`
- Attention:
  - `AttentionTask`
  - `AttentionDecision`
  - `AttentionScheduler`
- Cost governance:
  - `CostEstimate`
  - `BudgetPolicy`
  - `BudgetDecision`
  - `CostModel`

### `freeman.interface`

- CLI:
  - `build_parser()`
  - `main()`
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
- `SimResult`

### `freeman.llm`

- `EmbeddingAdapter`
- `DeterministicEmbeddingAdapter`
- `OpenAIEmbeddingClient`
- `DeepSeekChatClient`
- `DeepSeekFreemanOrchestrator`
- `LLMDrivenSimulationRun`

## CLI Map

Command:
- `python -m freeman.interface.cli run <schema_path> [--policies-path <path>]`
  - compile schema, run the analysis pipeline, print simulation JSON
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
  - re-embed legacy nodes without vectors and sync them into ChromaDB
- `python -m freeman.interface.cli override-param <world_path> <param_path> <value> [--output-path <path>]`
  - apply one parameter override to a world snapshot
- `python -m freeman.interface.cli override-sign <world_path> <edge_id> <expected_sign> [--output-path <path>]`
  - override one causal-edge sign in a world snapshot
- `python -m freeman.interface.cli rerun-domain <world_path> [--max-steps <n>] [--output-path <path>]`
  - rerun a world snapshot after overrides
- `python -m freeman.interface.cli diff-domain <baseline_path> <current_path> [--output-path <path>]`
  - export structured differences between two snapshots or simulation payloads

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

### End-to-end pipeline

```python
from freeman.agent.analysispipeline import AnalysisPipeline

pipeline = AnalysisPipeline()
result = pipeline.run(schema, policies=[])
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
- `memory.backend`
- `memory.json_path`

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
