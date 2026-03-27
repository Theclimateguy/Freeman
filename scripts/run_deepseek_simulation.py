"""Run an LLM-driven Freeman simulation from a natural-language domain brief."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from freeman.llm.deepseek import DeepSeekChatClient
from freeman.llm.orchestrator import DeepSeekFreemanOrchestrator


def _load_api_key(path: str | None) -> str:
    """Load the DeepSeek API key from env or a local file."""

    env_key = os.getenv("DEEPSEEK_API_KEY")
    if env_key:
        return env_key.strip()
    if path:
        return Path(path).read_text(encoding="utf-8").strip()
    raise RuntimeError("DeepSeek API key not found. Set DEEPSEEK_API_KEY or pass --key-file.")


def main() -> None:
    """Parse CLI arguments, run the orchestrator, and print a compact report."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--domain-brief", required=True, help="Natural-language description of the target domain.")
    parser.add_argument("--key-file", default="DS.txt", help="Path to the DeepSeek API key file.")
    parser.add_argument("--max-steps", type=int, default=20, help="Number of Freeman simulation steps.")
    parser.add_argument("--seed", type=int, default=42, help="Deterministic Freeman seed.")
    parser.add_argument("--output", default="", help="Optional JSON output path for the full run artifact.")
    args = parser.parse_args()

    client = DeepSeekChatClient(api_key=_load_api_key(args.key_file))
    orchestrator = DeepSeekFreemanOrchestrator(client)
    run = orchestrator.run(args.domain_brief, max_steps=args.max_steps, seed=args.seed)

    if args.output:
        orchestrator.save_run(run, args.output)

    summary = {
        "world_id": run.world_id,
        "synthesis_attempts": run.synthesis_attempts,
        "final_outcomes": run.simulation["final_outcome_probs"],
        "confidence": run.simulation["confidence"],
        "steps_run": run.simulation["steps_run"],
        "interpretation": run.interpretation,
        "output_path": args.output or None,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
