"""Dry-run coverage for the FAAB benchmark package."""

from __future__ import annotations

import json
from pathlib import Path

from freeman.llm import HashingEmbeddingAdapter
from scripts.benchmark_faab.generate_dummy_dataset import generate_dummy_dataset
from scripts.benchmark_faab.metrics import brier_score, outcome_tar_at_n, probability_tar_at_n
from scripts.benchmark_faab.runner import (
    BenchmarkRunner,
    HeuristicBenchmarkClient,
    MODE_A_FULL,
    MODE_B_AMNESIC,
    RunnerConfig,
    load_cases,
)


def test_brier_score_perfect() -> None:
    assert brier_score({"A": 1.0, "B": 0.0}, "A") == 0.0


def test_brier_score_worst() -> None:
    assert brier_score({"A": 0.0, "B": 1.0}, "A") == 2.0


def test_brier_score_uniform_two() -> None:
    assert brier_score({"A": 0.5, "B": 0.5}, "A") == 0.5


def test_probability_tar_at_n_exact_match() -> None:
    score = probability_tar_at_n(
        [
            {"A": 0.8, "B": 0.2},
            {"A": 0.8, "B": 0.2},
            {"A": 0.8, "B": 0.2},
        ]
    )

    assert score == 1.0


def test_probability_tar_at_n_detects_drift() -> None:
    score = probability_tar_at_n(
        [
            {"A": 0.8, "B": 0.2},
            {"A": 0.75, "B": 0.25},
        ],
        epsilon=1.0e-3,
    )

    assert score == 0.0


def test_outcome_tar_at_n_requires_identical_labels() -> None:
    assert outcome_tar_at_n(["A", "A", "A"]) == 1.0
    assert outcome_tar_at_n(["A", "B"]) == 0.0


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
    assert len(trace["llm_calls"]) == 5
    assert trace["steps"]["t1"]["retrieved_node_count"] >= 1
    assert "parameter_vectors" in trace
    assert trace["parameter_vectors"]["t1"]["shock_decay"] < 1.0
    assert result.t0_brier_score is not None
    assert result.t1_brier_score is not None


def test_benchmark_runner_repeatability_metrics_are_recorded(tmp_path: Path) -> None:
    dataset_path = generate_dummy_dataset(tmp_path / "cases.jsonl")
    case = load_cases(dataset_path)[0]
    run_dir = tmp_path / "faab_repeat_run"
    runner = BenchmarkRunner(
        mode=MODE_A_FULL,
        output_dir=run_dir,
        llm_client=HeuristicBenchmarkClient(),
        semantic_embedding_adapter=HashingEmbeddingAdapter(),
        config=RunnerConfig(output_dir=run_dir, repeat_runs=3),
    )

    result = runner.evaluate_case(case)

    assert result.status == "ok"
    assert result.repeat_runs == 3
    assert result.successful_runs == 3
    assert result.t0_max_l1_repeat_distance == 0.0
    assert result.t1_max_l1_repeat_distance == 0.0
    assert result.t0_primary_tar == 1.0
    assert result.t1_primary_tar == 1.0
    assert result.t0_secondary_tar == 1.0
    assert result.t1_secondary_tar == 1.0
    assert len(result.repeats) == 3
    assert (run_dir / "traces" / "climate_drought_response__mode_a_full__repeat_01.json").exists()
    assert (run_dir / "traces" / "climate_drought_response__mode_a_full__repeat_03.json").exists()


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
