"""CLI entrypoint for the Freeman Autonomous Analyst Benchmark."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import logging
from pathlib import Path
import sys
from typing import List

import pandas as pd

PACKAGE_ROOT = Path(__file__).resolve().parent
REPO_ROOT = PACKAGE_ROOT.parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from scripts.benchmark_faab.generate_dummy_dataset import generate_dummy_dataset  # noqa: E402
from scripts.benchmark_faab.runner import (  # noqa: E402
    ALL_MODES,
    BenchmarkRunner,
    HeuristicBenchmarkClient,
    RunnerConfig,
    load_cases,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="run_benchmark")
    parser.add_argument(
        "--dataset",
        default=str(PACKAGE_ROOT / "dataset" / "cases.jsonl"),
        help="Path to the benchmark JSONL dataset.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Run directory. Defaults to runs/faab_eval_<timestamp>/",
    )
    parser.add_argument(
        "--modes",
        nargs="+",
        default=ALL_MODES,
        choices=ALL_MODES,
        help="Benchmark modes to execute.",
    )
    parser.add_argument("--retrieval-top-k", type=int, default=8)
    parser.add_argument("--max-context-nodes", type=int, default=12)
    parser.add_argument("--max-steps", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--embedding-model", default="nomic-embed-text")
    parser.add_argument("--ollama-base-url", default="http://127.0.0.1:11434")
    parser.add_argument("--deepseek-model", default="deepseek-chat")
    parser.add_argument("--deepseek-base-url", default="https://api.deepseek.com")
    parser.add_argument("--deepseek-timeout-seconds", type=float, default=90.0)
    parser.add_argument("--state-time-decay", type=float, default=0.5)
    parser.add_argument("--shared-memory-across-cases", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="Use the deterministic heuristic client instead of real LLM calls.")
    parser.add_argument("--generate-dummy-if-missing", action="store_true")
    return parser


def main(argv: List[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    run_root = Path(args.output_dir).resolve() if args.output_dir else (REPO_ROOT / "runs" / f"faab_eval_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}").resolve()
    run_root.mkdir(parents=True, exist_ok=True)
    dataset_path = Path(args.dataset).resolve()
    if not dataset_path.exists():
        if not args.generate_dummy_if_missing:
            raise FileNotFoundError(f"Dataset not found: {dataset_path}")
        generate_dummy_dataset(dataset_path)
    cases = load_cases(dataset_path)
    config = RunnerConfig(
        output_dir=run_root,
        retrieval_top_k=args.retrieval_top_k,
        max_context_nodes=args.max_context_nodes,
        max_steps=args.max_steps,
        seed=args.seed,
        embedding_model=args.embedding_model,
        ollama_base_url=args.ollama_base_url,
        deepseek_model=args.deepseek_model,
        deepseek_base_url=args.deepseek_base_url,
        deepseek_timeout_seconds=args.deepseek_timeout_seconds,
        shared_memory_across_cases=bool(args.shared_memory_across_cases),
        state_time_decay=args.state_time_decay,
    )
    llm_client = HeuristicBenchmarkClient() if args.dry_run else None

    all_results = []
    for mode in args.modes:
        runner = BenchmarkRunner(
            mode=mode,
            output_dir=run_root,
            llm_client=llm_client,
            config=config,
        )
        results = runner.run_cases(cases)
        all_results.extend(result.snapshot() for result in results)

    metrics = pd.DataFrame(
        [
            {
                "case_id": item["case_id"],
                "mode": item["mode"],
                "t0_accuracy": item["t0_accuracy"],
                "t1_accuracy": item["t1_accuracy"],
                "retrieval_precision": item["retrieval_precision"],
                "autonomy_flag": item["autonomy_flag"],
            }
            for item in all_results
        ]
    )
    metrics.to_csv(run_root / "metrics.csv", index=False)
    summary_path = run_root / "summary.json"
    summary_path.write_text(json.dumps(all_results, indent=2, sort_keys=True), encoding="utf-8")
    print(summary_path)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
