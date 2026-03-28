"""Generate a dummy JSONL dataset for the Freeman Autonomous Analyst Benchmark."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Dict, List

PACKAGE_ROOT = Path(__file__).resolve().parent
REPO_ROOT = PACKAGE_ROOT.parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def default_dummy_cases() -> List[Dict[str, Any]]:
    """Return a compact cross-domain dummy benchmark set."""

    return [
        {
            "case_id": "climate_drought_response",
            "domain": "climate_risk",
            "t0_signal": (
                "Snowpack finished far below normal and soil moisture is already depleted across the basin. "
                "Reservoir operators say carryover storage is weaker than last year."
            ),
            "t1_signal": (
                "A severe early heatwave arrived, emergency water restrictions were delayed, and orchard stress is rising "
                "as reservoir levels fall toward critical thresholds."
            ),
            "ground_truth_t2": {
                "dominant_outcome": "water_shortage_spiral",
                "key_metric_name": "reservoir_storage",
                "key_metric": 28.0,
            },
        },
        {
            "case_id": "macro_trade_to_recession",
            "domain": "macroeconomy",
            "t0_signal": (
                "Freight costs and new tariffs pushed import prices sharply higher, while households report renewed inflation anxiety."
            ),
            "t1_signal": (
                "Three months later, manufacturers announced layoffs, bank credit standards tightened, and new orders fell again "
                "despite the central bank staying hawkish."
            ),
            "ground_truth_t2": {
                "dominant_outcome": "recession_spiral",
                "key_metric_name": "business_demand",
                "key_metric": 24.0,
            },
        },
        {
            "case_id": "relationship_repair_after_stress",
            "domain": "social_relationships",
            "t0_signal": (
                "Work overload, delayed replies, and jealousy around an ex-partner triggered repeated arguments and visible trust erosion."
            ),
            "t1_signal": (
                "The couple started weekly check-ins, apologized openly, and agreed to counseling after one long honest conversation."
            ),
            "ground_truth_t2": {
                "dominant_outcome": "repair_path",
                "key_metric_name": "trust_level",
                "key_metric": 70.0,
            },
        },
        {
            "case_id": "film_buzz_frontload",
            "domain": "film_release",
            "t0_signal": (
                "Trailer buzz is extreme, pre-sales are strong, and marketing spend keeps climbing despite mixed critic reactions."
            ),
            "t1_signal": (
                "Opening weekend was large, but weekday grosses collapsed quickly, exit scores disappointed, and audience chatter suggests "
                "a front-loaded release."
            ),
            "ground_truth_t2": {
                "dominant_outcome": "front_loaded_opening",
                "key_metric_name": "box_office_legs",
                "key_metric": 22.0,
            },
        },
    ]


def generate_dummy_dataset(output_path: str | Path) -> Path:
    """Write the default dummy dataset to JSONL."""

    target = Path(output_path).resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    lines = [json.dumps(case, ensure_ascii=False) for case in default_dummy_cases()]
    target.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return target


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="generate_dummy_dataset")
    parser.add_argument(
        "--output",
        default=str(PACKAGE_ROOT / "dataset" / "cases.jsonl"),
        help="Path to the output JSONL dataset.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    path = generate_dummy_dataset(args.output)
    print(path)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
