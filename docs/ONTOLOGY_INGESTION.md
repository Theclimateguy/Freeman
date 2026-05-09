# Ontology Ingestion Modes

Freeman should be treated as an ontology-construction runtime, not only as a "random news to graph" system. The right question is:

\[
\text{graph quality} = f(\text{seed quality}, \text{brief quality}, \text{model quality}, \text{verification}, \text{ingestion constraints})
\]

The repository now makes those ingestion constraints explicit through `bootstrap_contract` persisted in `bootstrap_package.json`.

## Supported strategies

### 1. `seed_schema`

Use an existing Freeman schema or seed graph as the bootstrap ontology.

Best for:
- production baselines
- deterministic experiments
- ontology maintenance on top of a trusted seed graph

Limitations:
- initial ontology coverage is bounded by the seed graph
- novelty must arrive later through stream updates and ontology repair

### 2. `brief_local_etl`

Use a natural-language brief and build the ontology locally through ETL (`skeleton -> edges -> verifier`).

Best for:
- privacy-sensitive work
- local experimentation
- cheap iterative ontology drafts

Limitations:
- highly sensitive to brief quality
- local 7b/14b-class models may timeout, omit entities, or underfit causal structure

### 3. `brief_local_etl_with_fallback_seed`

Try local ETL first, but require a deterministic fallback seed graph.

Best for:
- local-first production runs
- workflows where a graph must always be returned

Limitations:
- if ETL fails, the returned graph may be dominated by the fallback ontology rather than the brief
- benchmarking must record whether the final graph came from `etl_from_brief` or `fallback_schema_seed`

### 4. `brief_remote_etl`

Use a natural-language brief and a stronger remote API model.

Best for:
- first-pass ontology induction in complex domains
- broad ontology construction before local hardening

Limitations:
- network dependency, API cost, and governance constraints
- weaker reproducibility unless prompt, model version, and brief are pinned

### 5. `brief_remote_etl_with_fallback_seed`

Use a strong remote model, but keep a deterministic fallback ontology.

Best for:
- expensive or mission-critical bootstraps
- complex domains where remote ETL is desirable but failure is unacceptable

Limitations:
- same fallback contamination risk as the local fallback strategy
- mixed operational risk: remote outage plus fallback drift

## Recommended operating policy

Use a three-tier policy instead of forcing one mode everywhere:

1. `seed_schema` for stable production baselines.
2. `brief_remote_etl` or `brief_local_etl` for ontology induction and expansion.
3. `*_with_fallback_seed` only when continuity matters more than attribution purity.

In other words, Freeman should explicitly separate:

\[
\text{ontology induction} \neq \text{runtime continuity}
\]

If the goal is ontology quality, prefer the strongest available brief plus model. If the goal is guaranteed operation, keep a seed graph and record when the system fell back to it.
