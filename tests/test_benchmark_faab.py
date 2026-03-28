"""Dry-run coverage for the FAAB benchmark package."""

from __future__ import annotations

import json
from pathlib import Path

from freeman.llm import HashingEmbeddingAdapter
from scripts.benchmark_faab.generate_dummy_dataset import generate_dummy_dataset
from scripts.benchmark_faab.runner import (
    BenchmarkRunner,
    HeuristicBenchmarkClient,
    MODE_A_FULL,
    MODE_B_AMNESIC,
    load_cases,
)


def test_generate_dummy_dataset_writes_jsonl(tmp_path: Path) -> None:
    output = generate_dummy_dataset(tmp_path / "cases.jsonl")

    assert output.exists()
    cases = load_cases(output)
    assert len(cases) == 4
    assert cases[0].domain == "climate_risk"
    assert cases[1].ground_truth_t2["dominant_outcome"] == "recession_spiral"


def test_benchmark_runner_mode_a_writes_trace_and_snapshot(tmp_path: Path) -> None:
    dataset_path = generate_dummy_dataset(tmp_path / "cases.jsonl")
    case = load_cases(dataset_path)[0]
    run_dir = tmp_path / "faab_run"
    runner = BenchmarkRunner(
        mode=MODE_A_FULL,
        output_dir=run_dir,
        llm_client=HeuristicBenchmarkClient(),
        semantic_embedding_adapter=HashingEmbeddingAdapter(),
    )

    result = runner.evaluate_case(case)

    assert result.status == "ok"
    assert result.t0_prediction is not None
    assert result.t1_prediction is not None
    trace_path = run_dir / "traces" / "climate_drought_response__mode_a_full.json"
    snapshot_path = run_dir / "kg_snapshots" / "climate_drought_response__mode_a_full__t1.json"
    assert trace_path.exists()
    assert snapshot_path.exists()
    trace = json.loads(trace_path.read_text(encoding="utf-8"))
    assert len(trace["llm_calls"]) == 4
    assert trace["steps"]["t1"]["retrieved_node_count"] >= 1


def test_benchmark_runner_mode_b_clears_memory_before_t1(tmp_path: Path) -> None:
    dataset_path = generate_dummy_dataset(tmp_path / "cases.jsonl")
    case = load_cases(dataset_path)[0]
    run_dir = tmp_path / "faab_run"
    runner = BenchmarkRunner(
        mode=MODE_B_AMNESIC,
        output_dir=run_dir,
        llm_client=HeuristicBenchmarkClient(),
        semantic_embedding_adapter=HashingEmbeddingAdapter(),
    )

    result = runner.evaluate_case(case)

    assert result.status == "ok"
    assert result.t1_prediction is not None
    assert result.t1_prediction.retrieved_node_count == 0
    assert result.retrieval_precision == 0.0
    assert result.autonomy_flag is False
