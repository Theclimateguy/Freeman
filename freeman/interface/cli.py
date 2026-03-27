"""Command-line interface for Freeman."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import yaml

from freeman.agent.analysispipeline import AnalysisPipeline
from freeman.game.runner import SimConfig
from freeman.interface.api import InterfaceAPI
from freeman.interface.kgexport import KnowledgeGraphExporter
from freeman.interface.modeloverride import ModelOverrideAPI
from freeman.interface.simulationdiff import build_simulation_diff, export_simulation_diff
from freeman.llm import DeterministicEmbeddingAdapter, OpenAIEmbeddingClient
from freeman.core.world import WorldState
from freeman.memory.knowledgegraph import KnowledgeGraph
from freeman.memory.reconciler import Reconciler
from freeman.memory.sessionlog import SessionLog
from freeman.memory.vectorstore import KGVectorStore


def _load_payload(path: str | Path) -> Any:
    target = Path(path).resolve()
    text = target.read_text(encoding="utf-8")
    if target.suffix.lower() in {".yaml", ".yml"}:
        return yaml.safe_load(text)
    return json.loads(text)


def _coerce_scalar(value: str) -> Any:
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return value


def _load_config(path: str | Path) -> dict[str, Any]:
    target = Path(path).resolve()
    if not target.exists():
        return {}
    return yaml.safe_load(target.read_text(encoding="utf-8")) or {}


def _resolve_path(base: Path, candidate: str | None, default: str) -> Path:
    target = Path(candidate or default)
    return target if target.is_absolute() else (base / target).resolve()


def _build_vectorstore(config: dict[str, Any], *, config_path: Path) -> KGVectorStore | None:
    memory_cfg = config.get("memory", {})
    vector_cfg = memory_cfg.get("vector_store", {})
    if not vector_cfg.get("enabled", False):
        return None
    backend = str(vector_cfg.get("backend", "chroma")).lower()
    if backend != "chroma":
        raise ValueError(f"Unsupported vector store backend: {backend}")
    path = _resolve_path(config_path.parent, vector_cfg.get("path"), "./data/chroma_db")
    collection = str(vector_cfg.get("collection", "kg_nodes"))
    return KGVectorStore(path=path, collection_name=collection)


def _build_embedding_adapter(config: dict[str, Any], *, use_stub: bool = False) -> tuple[Any, str]:
    memory_cfg = config.get("memory", {})
    model = str(memory_cfg.get("embedding_model", "text-embedding-3-small"))
    if use_stub or not os.getenv("OPENAI_API_KEY"):
        return DeterministicEmbeddingAdapter(), "deterministic_stub"
    return OpenAIEmbeddingClient(api_key=os.environ["OPENAI_API_KEY"], model=model), model


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="freeman")
    parser.add_argument("--config-path", default="config.yaml")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run")
    run_parser.add_argument("schema_path")
    run_parser.add_argument("--policies-path")

    query_parser = subparsers.add_parser("query")
    query_parser.add_argument("--text")
    query_parser.add_argument("--status")
    query_parser.add_argument("--node-type")
    query_parser.add_argument("--min-confidence", type=float)

    export_parser = subparsers.add_parser("export-kg")
    export_parser.add_argument("format", choices=["html", "json-ld", "dot"])
    export_parser.add_argument("output_path")

    subparsers.add_parser("status")

    reconcile_parser = subparsers.add_parser("reconcile")
    reconcile_parser.add_argument("session_log_path")

    archive_parser = subparsers.add_parser("kg-archive")
    archive_parser.add_argument("--node-id")
    archive_parser.add_argument("--reason", default="manual_archive")

    reindex_parser = subparsers.add_parser("kg-reindex")
    reindex_parser.add_argument("--batch-size", type=int, default=100)
    reindex_parser.add_argument("--use-stub-embeddings", action="store_true")

    override_param_parser = subparsers.add_parser("override-param")
    override_param_parser.add_argument("world_path")
    override_param_parser.add_argument("param_path")
    override_param_parser.add_argument("value")
    override_param_parser.add_argument("--output-path")

    override_sign_parser = subparsers.add_parser("override-sign")
    override_sign_parser.add_argument("world_path")
    override_sign_parser.add_argument("edge_id")
    override_sign_parser.add_argument("expected_sign")
    override_sign_parser.add_argument("--output-path")

    rerun_parser = subparsers.add_parser("rerun-domain")
    rerun_parser.add_argument("world_path")
    rerun_parser.add_argument("--max-steps", type=int, default=5)
    rerun_parser.add_argument("--output-path")

    diff_parser = subparsers.add_parser("diff-domain")
    diff_parser.add_argument("baseline_path")
    diff_parser.add_argument("current_path")
    diff_parser.add_argument("--output-path")

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    config_path = Path(args.config_path).resolve()
    config = _load_config(config_path)
    vectorstore = _build_vectorstore(config, config_path=config_path)
    needs_embeddings = vectorstore is not None or args.command == "kg-reindex"
    embedding_adapter = None
    embedding_backend = None
    if needs_embeddings:
        embedding_adapter, embedding_backend = _build_embedding_adapter(
            config,
            use_stub=getattr(args, "use_stub_embeddings", False),
        )
    knowledge_graph = KnowledgeGraph(
        config_path=config_path,
        llm_adapter=embedding_adapter,
        vectorstore=vectorstore,
    )

    if args.command == "run":
        schema = _load_payload(args.schema_path)
        policies = _load_payload(args.policies_path) if args.policies_path else []
        result = AnalysisPipeline(knowledge_graph=knowledge_graph).run(schema, policies=policies)
        print(json.dumps(result.simulation, indent=2, sort_keys=True))
        return 0

    if args.command == "query":
        payload = InterfaceAPI(knowledge_graph).post_query(
            text=args.text,
            status=args.status,
            node_type=args.node_type,
            min_confidence=args.min_confidence,
        )
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0

    if args.command == "export-kg":
        exporter = KnowledgeGraphExporter()
        if args.format == "html":
            output = exporter.export_html(knowledge_graph, args.output_path)
        elif args.format == "json-ld":
            output = exporter.export_json_ld(knowledge_graph, args.output_path)
        else:
            output = exporter.export_dot(knowledge_graph, args.output_path)
        print(str(output))
        return 0

    if args.command == "status":
        payload = InterfaceAPI(knowledge_graph).get_status()
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0

    if args.command == "reconcile":
        session_log = SessionLog.load(args.session_log_path)
        result = Reconciler().reconcile(knowledge_graph, session_log)
        print(json.dumps(result.__dict__, indent=2, sort_keys=True))
        return 0

    if args.command == "kg-archive":
        if args.node_id:
            archived = knowledge_graph.archive(args.node_id, reason=args.reason)
            print(json.dumps(archived.snapshot(), indent=2, sort_keys=True))
            return 0
        archived_ids = []
        for node in knowledge_graph.nodes():
            if node.confidence < 0.15 and node.status != "archived":
                knowledge_graph.archive(node.id, reason=args.reason)
                archived_ids.append(node.id)
        print(json.dumps({"archived_ids": archived_ids}, indent=2, sort_keys=True))
        return 0

    if args.command == "kg-reindex":
        if vectorstore is None:
            print(json.dumps({"reembedded": 0, "synced": 0, "status": "vector_store_disabled"}, indent=2, sort_keys=True))
            return 0
        missing = [
            node
            for node in knowledge_graph.nodes(lazy_embed=False)
            if node.status != "archived" and node.content and not node.embedding
        ]
        reembedded = 0
        for batch_start in range(0, len(missing), args.batch_size):
            batch = missing[batch_start : batch_start + args.batch_size]
            for node in batch:
                node.embedding = []
                knowledge_graph.update_node(node)
                reembedded += 1
        synced = vectorstore.sync_from_kg(knowledge_graph)
        knowledge_graph.save()
        print(
            json.dumps(
                {
                    "embedding_backend": embedding_backend,
                    "reembedded": reembedded,
                    "synced": synced,
                    "vectorstore_path": str(vectorstore.path),
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0

    if args.command == "override-param":
        world = WorldState.from_snapshot(_load_payload(args.world_path))
        override_api = ModelOverrideAPI()
        override_api.register_domain(world.domain_id, world)
        payload = override_api.patch_params(world.domain_id, {args.param_path: _coerce_scalar(args.value)})
        if args.output_path:
            output = Path(args.output_path).resolve()
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(
                json.dumps(override_api.records[world.domain_id].current_world.snapshot(), indent=2, sort_keys=True),
                encoding="utf-8",
            )
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0

    if args.command == "override-sign":
        world = WorldState.from_snapshot(_load_payload(args.world_path))
        override_api = ModelOverrideAPI()
        override_api.register_domain(world.domain_id, world)
        payload = override_api.patch_edge(world.domain_id, args.edge_id, args.expected_sign)
        if args.output_path:
            output = Path(args.output_path).resolve()
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(
                json.dumps(override_api.records[world.domain_id].current_world.snapshot(), indent=2, sort_keys=True),
                encoding="utf-8",
            )
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0

    if args.command == "rerun-domain":
        world = WorldState.from_snapshot(_load_payload(args.world_path))
        override_api = ModelOverrideAPI(sim_config=SimConfig(max_steps=args.max_steps))
        override_api.register_domain(world.domain_id, world)
        payload = override_api.rerun_domain(world.domain_id)
        if args.output_path:
            output = Path(args.output_path).resolve()
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0

    if args.command == "diff-domain":
        before = _load_payload(args.baseline_path)
        after = _load_payload(args.current_path)
        domain_id = before.get("domain_id", after.get("domain_id", "unknown"))
        report = build_simulation_diff(domain_id=domain_id, before=before, after=after)
        if args.output_path:
            export_simulation_diff(report, args.output_path)
        print(json.dumps(report.snapshot(), indent=2, sort_keys=True))
        return 0

    return 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
