# Freeman librarian

`Freeman librarian` is a document-centric fork of Freeman. It is designed to build a persistent world model from a corpus of documents rather than from a one-shot domain brief or a hand-written schema.

Target pipeline:

```text
documents -> structured extraction -> ontology assembly -> world compilation
          -> verification -> KG reconciliation -> incremental updates
```

## Why It Exists

The original Freeman is strong at:

- deterministic world simulation
- verification of world and causal models
- persistent memory in a knowledge graph
- reconciliation of new evidence against past claims

`Freeman librarian` keeps that core but changes the bootstrap path.

Instead of:

```text
brief/schema -> compile -> simulate
```

it works as:

```text
document corpus -> extract entities/relations -> build ontology -> compile world
```

This is useful when the source of truth is a real document collection:

- regulations
- procedures
- internal policies
- contracts
- organizational documents
- legal corpora

## What Is Implemented

- isolated fork namespace in [`freeman_librarian/`](freeman_librarian)
- `DocumentExtractor` for `txt`, `md`, `json`, `pdf`, `docx`
- `DocumentBootstrapper` for document-driven world construction
- organization-specific ontology:
  - actors: employees, roles, units
  - resources: processes, documents, artifacts, systems
  - relations: `owns`, `participates_in`, `requires`, `delegates_to`, `reports_to`
- Level 3 verifier invariants on top of inherited Freeman verification
- CLI entrypoint `freeman-librarian`
- legal benchmark smoke-test adapter for Russian legal documents

## Repository Layout

```text
freeman_librarian/
  extractor/      document parsing and normalization
  bootstrap/      document bootstrap orchestration
  ontology/       org/document ontology and compiler
  verifier/       inherited verifier + level3 invariants
  demo/           reproducible demos on public datasets
  connectors/     document-facing exports
```

The original `freeman/` package is still present in this branch as inherited core code. The active fork logic lives in `freeman_librarian/`.

## Install

Core install:

```bash
pip install .
```

Optional extras:

```bash
pip install ".[dev]"
pip install ".[documents]"
pip install ".[demo]"
```

## Quick Start

Run document bootstrap on local files:

```bash
python -m freeman_librarian.interface.cli bootstrap-docs \
  ./docs/reglament.md ./docs/policy.docx \
  --domain-id finance_ops \
  --world-output-path ./runs/finance_ops/world.json \
  --session-log-output-path ./runs/finance_ops/session.json \
  --report-output-path ./runs/finance_ops/report.json
```

Or use the installed script:

```bash
freeman-librarian bootstrap-docs \
  ./docs/reglament.md ./docs/policy.docx \
  --domain-id finance_ops
```

## Legal Demo

There is a reproducible smoke-test path on a small Russian legal corpus from Hugging Face:

- dataset: [parlorsky/legal-rag-benchmark-ru](https://huggingface.co/datasets/parlorsky/legal-rag-benchmark-ru)
- demo script: [`scripts/freeman_librarian/run_legal_demo.py`](scripts/freeman_librarian/run_legal_demo.py)
- example config: [`examples/freeman_librarian_legal_demo.yaml`](examples/freeman_librarian_legal_demo.yaml)

Run:

```bash
python scripts/freeman_librarian/run_legal_demo.py --max-docs 4
```

This writes artifacts under:

```text
./runs/freeman_librarian_legal_demo/
```

including:

- `world.json`
- `report.json`
- `session.json`
- `summary.json`
- `kg_state.json`
- `raw_docs/`
- `prepared_docs/`

## Current Limits

- the extractor is still heuristic and conservative
- raw legal or regulatory text often needs an adapter layer before ontology compilation
- the current demo validates the pipeline, not domain completeness
- calibration should be done on the actual target corpus

## Current Smoke-Test Status

The current legal demo run on 4 documents succeeds end-to-end:

- documents downloaded and converted into structured notes
- ontology assembled
- world compiled
- verification passed
- KG written and exportable

Remaining verifier warnings are soft `shock_decay` alerts on document resources, not hard structural failures.
