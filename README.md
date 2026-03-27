# Freeman

Freeman is a domain-agnostic world simulator for LLM agents.

It compiles a JSON domain schema into a deterministic simulation world, evolves the world through pluggable transition operators, verifies structural and stepwise consistency, and scores outcome probabilities.

Freeman can now also sit inside an LLM-driven loop: a model proposes a compact domain package from a natural-language brief, Freeman compiles and simulates it, and the model interprets the resulting trajectory.

## Scope

- Universal simulation core for actors, resources, relations, outcomes, and causal graphs
- Pluggable evolution operators: linear, stock-flow, logistic, threshold, coupled
- Three-level verifier with hard stops on level 0 invariants
- Multi-domain composition through shared resources
- Tool API for compile, run, inspect, and verify workflows
- DeepSeek-backed orchestration layer for brief -> domain package -> simulation -> interpretation

## Install

```bash
pip install -e .
```

## Core Workflow

1. Describe a domain in natural language or provide a Freeman JSON schema.
2. Compile it into a `WorldState`.
3. Run the simulation with policies.
4. Inspect outcome probabilities, trajectory snapshots, and verifier output.

## LLM-Driven Workflow

Freeman includes a minimal DeepSeek orchestration layer in:

- `freeman/llm/deepseek.py`
- `freeman/llm/orchestrator.py`
- `scripts/run_deepseek_simulation.py`

The intended loop is:

1. DeepSeek converts a natural-language domain brief into a compact structured package.
2. Freeman validates and compiles the package into an executable world.
3. Freeman runs the simulation and returns trajectory and verifier outputs.
4. DeepSeek interprets the result and proposes the next policy or scenario revision.

Run a local DeepSeek-driven simulation with:

```bash
python3 scripts/run_deepseek_simulation.py \
  --domain-brief "Describe the system you want to simulate" \
  --key-file DS.txt \
  --max-steps 20 \
  --output runs/demo.json
```

Notes:

- Set `DEEPSEEK_API_KEY`.
- `runs/` is a local artifact directory and is ignored by git.
- The current LLM workflow is pragmatic rather than fully autonomous: the most robust path is a compact concept, validation, repair, and then execution.

## Test

```bash
python3 -m pytest -q
```
