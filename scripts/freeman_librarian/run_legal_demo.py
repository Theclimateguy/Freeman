"""Run Freeman librarian on a small Russian legal benchmark subset."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from freeman_librarian.demo import DEFAULT_LEGAL_BENCHMARK_REPO, run_legal_benchmark_demo


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-id", default=DEFAULT_LEGAL_BENCHMARK_REPO)
    parser.add_argument("--max-docs", type=int, default=4)
    parser.add_argument("--domain-id", default="legal_benchmark_ru_demo")
    parser.add_argument("--output-root", default="./runs/freeman_librarian_legal_demo")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    result = run_legal_benchmark_demo(
        args.output_root,
        repo_id=args.repo_id,
        max_docs=args.max_docs,
        domain_id=args.domain_id,
    )
    print(json.dumps(result.summary, indent=2, sort_keys=True, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
