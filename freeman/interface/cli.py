"""Minimal CLI for the Freeman lite runtime."""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any

from freeman.lite_api import compile as compile_model
from freeman.lite_api import export_kg, query as query_kg, update as update_model


def _print_json(payload: dict[str, Any]) -> None:
    print(json.dumps(payload, indent=2, sort_keys=True))


def _add_config_option(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", dest="config_path", default=None)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="freeman-lite", description="Freeman lite CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="Compile a model or update it with one signal.")
    _add_config_option(run_parser)
    run_group = run_parser.add_mutually_exclusive_group(required=True)
    run_group.add_argument("--schema", help="Path to a JSON/YAML schema file.")
    run_group.add_argument("--brief", help="Domain brief text or path to a text file.")
    run_group.add_argument("--signal", help="Signal text or path to a text file.")
    run_parser.add_argument("--verify-l2", action="store_true", help="Run level-2 verification in addition to L0/L1.")

    query_parser = subparsers.add_parser("query", help="Lexically query the persisted knowledge graph.")
    _add_config_option(query_parser)
    query_parser.add_argument("text", help="Query text.")
    query_parser.add_argument("--top-k", type=int, default=None)

    export_parser = subparsers.add_parser("export-kg", help="Export the knowledge graph.")
    _add_config_option(export_parser)
    export_parser.add_argument("--output", required=True, help="Output path.")
    export_parser.add_argument("--format", default="json", choices=["json", "jsonld", "dot", "html"])

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "run":
            if args.schema is not None:
                _print_json(compile_model(args.schema, config_path=args.config_path, verify_level2=args.verify_l2))
                return 0
            if args.brief is not None:
                _print_json(compile_model(args.brief, config_path=args.config_path, verify_level2=args.verify_l2))
                return 0
            _print_json(update_model(args.signal, config_path=args.config_path, verify_level2=args.verify_l2))
            return 0
        if args.command == "query":
            _print_json(query_kg(args.text, config_path=args.config_path, top_k=args.top_k))
            return 0
        if args.command == "export-kg":
            output_path = export_kg(args.output, config_path=args.config_path, fmt=args.format)
            _print_json({"status": "exported", "format": args.format, "path": str(output_path)})
            return 0
    except Exception as exc:  # noqa: BLE001
        error_payload = {"status": "error", "error": str(exc)}
        print(json.dumps(error_payload, indent=2, sort_keys=True), file=sys.stderr)
        return 1
    parser.error(f"Unsupported command: {args.command}")
    return 2


__all__ = ["build_parser", "main"]
