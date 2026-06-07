"""Query-mode handlers for the Freeman stream runtime CLI."""

from __future__ import annotations

import argparse
import json
from typing import Any

from freeman.agent.analysispipeline import AnalysisPipeline
from freeman.runtime.lifecycle import RuntimePaths
from freeman.runtime.queryengine import (
    RuntimeAnswerEngine,
    RuntimeQueryEngine,
    load_runtime_artifacts as _load_runtime_query_artifacts,
)

def _load_query_pipeline(config: dict[str, Any], paths: RuntimePaths) -> AnalysisPipeline:
    del config
    return _load_runtime_query_artifacts(paths.config_path).pipeline


def _query_anomalies(pipeline: AnalysisPipeline) -> dict[str, Any]:
    anomaly_nodes = [
        node.snapshot()
        for node in pipeline.knowledge_graph.query(node_type="anomaly_candidate")
    ]
    ontology_gap_traits = [
        node.snapshot()
        for node in pipeline.knowledge_graph.query(
            node_type="identity_trait",
            metadata_filters={"payload.trait_key": "ontology_gap"},
        )
    ]
    return {
        "anomaly_candidates": anomaly_nodes,
        "ontology_gap_traits": ontology_gap_traits,
    }


def _query_causal_edges(pipeline: AnalysisPipeline, *, limit: int) -> list[dict[str, Any]]:
    causal_edges = [
        edge.snapshot()
        for edge in pipeline.knowledge_graph.edges()
        if edge.relation_type in {"causes", "propagates_to", "threshold_exceeded"}
    ]
    causal_edges.sort(
        key=lambda item: (
            int((item.get("metadata") or {}).get("runtime_step", -1)),
            str(item.get("updated_at", "")),
            str(item.get("id", "")),
        ),
        reverse=True,
    )
    return causal_edges[: max(int(limit), 1)]


def _handle_query_mode(args: argparse.Namespace, config: dict[str, Any], paths: RuntimePaths) -> int:
    if args.query in {"semantic", "answer"}:
        if not str(args.text or "").strip():
            raise RuntimeError(f"--query {args.query} requires --text.")
        artifacts = _load_runtime_query_artifacts(paths.config_path)
        if args.query == "semantic":
            print(
                json.dumps(
                    RuntimeQueryEngine(artifacts).semantic_query(str(args.text), limit=args.limit).to_dict(),
                    indent=2,
                    sort_keys=True,
                )
            )
            return 0
        print(
            json.dumps(
                RuntimeAnswerEngine(artifacts).answer(str(args.text), limit=args.limit),
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    pipeline = _load_query_pipeline(config, paths)
    if args.query == "forecasts":
        payload = [summary.to_dict() for summary in pipeline.list_forecasts(status=args.status)]
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0
    if args.query == "explain":
        if not args.forecast_id:
            raise RuntimeError("--query explain requires --forecast-id.")
        explanation = pipeline.explain_forecast(str(args.forecast_id))
        print(explanation.to_text())
        return 0
    if args.query == "anomalies":
        print(json.dumps(_query_anomalies(pipeline), indent=2, sort_keys=True))
        return 0
    if args.query == "causal":
        print(json.dumps(_query_causal_edges(pipeline, limit=args.limit), indent=2, sort_keys=True))
        return 0
    raise ValueError(f"Unsupported query mode: {args.query}")


__all__ = [
    "_handle_query_mode",
]
