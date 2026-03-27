# Freeman

Freeman is a domain-agnostic world simulator for LLM agents.

It compiles a JSON domain schema into a deterministic simulation world, evolves the world through pluggable transition operators, verifies structural and stepwise consistency, and scores outcome probabilities.

## Scope

- Universal simulation core for actors, resources, relations, outcomes, and causal graphs
- Pluggable evolution operators: linear, stock-flow, logistic, threshold, coupled
- Three-level verifier with hard stops on level 0 invariants
- Multi-domain composition through shared resources
- Tool API for compile, run, inspect, and verify workflows

## Install

```bash
pip install -e .
```

## Test

```bash
python3 -m pytest -q
```
