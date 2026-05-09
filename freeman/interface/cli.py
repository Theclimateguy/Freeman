"""Command-line interface for Freeman."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import yaml

from freeman.agent.analysispipeline import AnalysisPipeline
from freeman.agent.costmodel import BudgetLedger, build_budget_policy, budget_tracking_enabled
from freeman.game.runner import SimConfig
from freeman.interface.api import InterfaceAPI
from freeman.interface.factory import (
    build_chat_client as _shared_build_chat_client,
    build_embedding_adapter as _shared_build_embedding_adapter,
    build_knowledge_graph as _shared_build_knowledge_graph,
    build_vectorstore as _shared_build_vectorstore,
    resolve_event_log_path as _shared_event_log_path,
    resolve_memory_json_path as _shared_memory_json_path,
    resolve_path as _shared_resolve_path,
    resolve_runtime_path as _shared_runtime_path,
)
from freeman.interface.identity import build_identity_state
from freeman.interface.kgevolution import KnowledgeGraphEvolutionExporter
from freeman.interface.kgexport import KnowledgeGraphExporter
from freeman.interface.modeloverride import ModelOverrideAPI
from freeman.interface.simulationdiff import build_simulation_diff, export_simulation_diff
from freeman.llm import ExplanationRenderer, IdentityNarrator
from freeman.core.world import WorldState
from freeman.memory.knowledgegraph import KGNode, KnowledgeGraph
from freeman.memory.reconciler import Reconciler
from freeman.memory.sessionlog import SessionLog
from freeman.runtime.checkpoint import CheckpointManager
from freeman.runtime.event_log import EventLog
from freeman.runtime.queryengine import RuntimeAnswerEngine, RuntimeQueryEngine, load_runtime_artifacts

DEFAULT_CONFIG: dict[str, Any] = {
    "agent": {
        "interest_function": "default",
        "budget_usd_per_day": 0.50,
        "source_refresh_seconds": 300,
        "sources": [],
        "stream_keywords": [],
        "bootstrap": {
            "mode": "llm_synthesize",
            "schema_path": None,
            "policies_path": None,
            "fallback_schema_path": "./freeman/domain/profiles/gim15.json",
            "domain_brief_path": "./examples/domain_brief_climate_news.md",
            "domain_brief": "",
            "package_normalization": "auto",
        },
    },
    "llm": {
        "provider": "ollama",
        "model": "qwen2.5-coder:14b",
        "base_url": "http://127.0.0.1:11434",
        "timeout_seconds": 90.0,
    },
    "freeman": {
        "default_evolution": "stock_flow",
        "level0_hard_stop": True,
        "epsilon": 1.0e-8,
        "sign_epsilon": 1.0e-4,
    },
    "sim": {
        "max_steps": 50,
        "dt": 1.0,
        "level2_check_every": 5,
        "level2_shock_delta": 0.01,
        "stop_on_hard_level2": True,
        "convergence_check_steps": 20,
        "convergence_epsilon": 1.0e-4,
        "fixed_point_max_iter": 20,
        "fixed_point_alpha": 0.1,
        "seed": 42,
    },
    "memory": {
        "backend": "networkx-json",
        "json_path": "./data/kg_state.json",
        "vector_store": {
            "enabled": False,
            "backend": "chroma",
            "path": "./data/chroma_db",
            "collection": "kg_nodes",
        },
        "embedding_provider": "hashing",
        "hashing_embedding_dimension": 384,
        "embedding_model": "text-embedding-3-small",
        "embedding_base_url": "http://127.0.0.1:11434",
        "embedding_timeout_seconds": 120.0,
        "embedding_prompt_prefix": "",
        "retrieval_top_k": 15,
        "max_context_nodes": 30,
        "session_log_path": "./data/sessions/",
    },
    "runtime": {
        "mode": "follow",
        "poll_interval_seconds": 30,
        "checkpoint_every_n_events": 25,
        "checkpoint_every_n_seconds": 300,
        "runtime_path": "./data/runtime",
        "event_log_path": "./data/runtime/event_log.jsonl",
        "resume_on_start": True,
    },
    "consciousness": {
        "idle_scheduler": {
            "threshold": 0.60,
            "weights": {
                "time_since_last_update": 0.25,
                "confidence_gap": 0.35,
                "hypothesis_age": 0.20,
                "attention_deficit": 0.20,
            },
        },
        "operators": {
            "capability_review": {
                "alpha": 2.0,
                "beta": 4.0,
            },
            "attention_rebalance": {
                "w_g": 0.30,
                "w_u": 0.30,
                "w_e": 0.25,
                "w_s": 0.15,
            },
            "trait_consolidation": {
                "lambda_k": 0.95,
                "eta_k": 0.10,
                "beta_k": 0.20,
                "min_delta": 0.01,
            },
        },
    },
}


def _merge_dicts(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge two config dictionaries."""

    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _merge_dicts(merged[key], value)
        else:
            merged[key] = value
    return merged


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
        return dict(DEFAULT_CONFIG)
    payload = yaml.safe_load(target.read_text(encoding="utf-8")) or {}
    return _merge_dicts(DEFAULT_CONFIG, payload)


def _resolve_path(base: Path, candidate: str | None, default: str) -> Path:
    target = Path(candidate or default)
    return target if target.is_absolute() else (base / target).resolve()


def _print_json(payload: dict[str, Any]) -> None:
    print(json.dumps(payload, indent=2, sort_keys=True))


def _budget_status(config: dict[str, Any], *, config_path: Path) -> dict[str, Any]:
    runtime_path = _runtime_path(config, config_path=config_path)
    policy = build_budget_policy(config)
    tracking = budget_tracking_enabled(config)
    ledger_path = runtime_path / "cost_ledger.jsonl"
    if tracking:
        ledger = BudgetLedger(ledger_path, policy=policy, auto_load=True)
        return ledger.summary()
    return {
        "tracking_enabled": False,
        "ledger_path": str(ledger_path.resolve()),
        "configured_usd_per_day": float(policy.max_compute_budget_per_session),
        "spent_usd": 0.0,
        "remaining_usd": float(policy.max_compute_budget_per_session),
        "entry_count": 0,
        "allowed_count": 0,
        "blocked_count": 0,
        "by_task_type": {},
        "stop_reasons": {},
    }


def _query_node_payload(node: KGNode) -> dict[str, Any]:
    """Serialize a node for CLI query responses without embedding payloads."""

    payload = node.snapshot()
    payload["embedding"] = []
    return payload


def _add_config_option(command_parser: argparse.ArgumentParser) -> None:
    """Attach the shared config option to a subcommand parser."""

    command_parser.add_argument("--config", "--config-path", dest="config_path", default=argparse.SUPPRESS)


def _memory_json_path(config: dict[str, Any], *, config_path: Path) -> Path:
    return _shared_memory_json_path(config, config_path=config_path)


def _runtime_path(config: dict[str, Any], *, config_path: Path) -> Path:
    return _shared_runtime_path(config, config_path=config_path)


def _event_log_path(config: dict[str, Any], *, config_path: Path) -> Path:
    return _shared_event_log_path(config, config_path=config_path)


def _build_vectorstore(config: dict[str, Any], *, config_path: Path):
    return _shared_build_vectorstore(config, config_path=config_path)


def _build_embedding_adapter(config: dict[str, Any], *, use_stub: bool = False) -> tuple[Any, str]:
    return _shared_build_embedding_adapter(config, use_stub=use_stub)


def _build_chat_client(config: dict[str, Any]) -> tuple[Any | None, str | None]:
    return _shared_build_chat_client(config)


def _build_sim_config(config: dict[str, Any]) -> SimConfig:
    sim_cfg = config.get("sim", {})
    return SimConfig(
        max_steps=int(sim_cfg.get("max_steps", 50)),
        dt=float(sim_cfg.get("dt", 1.0)),
        level2_check_every=int(sim_cfg.get("level2_check_every", 5)),
        level2_shock_delta=float(sim_cfg.get("level2_shock_delta", 0.01)),
        stop_on_hard_level2=bool(sim_cfg.get("stop_on_hard_level2", True)),
        convergence_check_steps=int(sim_cfg.get("convergence_check_steps", 20)),
        convergence_epsilon=float(sim_cfg.get("convergence_epsilon", 1.0e-4)),
        fixed_point_max_iter=int(sim_cfg.get("fixed_point_max_iter", 20)),
        fixed_point_alpha=float(sim_cfg.get("fixed_point_alpha", 0.1)),
        seed=int(sim_cfg.get("seed", 42)),
    )


def _bootstrap_storage(config: dict[str, Any], *, config_path: Path) -> dict[str, Any]:
    """Create the empty KG and storage directories for a fresh agent."""

    memory_cfg = config.get("memory", {})
    kg_path = _memory_json_path(config, config_path=config_path)
    session_log_path = _resolve_path(config_path.parent, memory_cfg.get("session_log_path"), "./data/sessions/")
    vector_cfg = memory_cfg.get("vector_store", {})
    vector_path = _resolve_path(config_path.parent, vector_cfg.get("path"), "./data/chroma_db")

    created: dict[str, Any] = {
        "knowledge_graph_path": str(kg_path),
        "session_log_path": str(session_log_path),
        "vector_store_path": str(vector_path) if vector_cfg.get("enabled", False) else None,
        "created": [],
    }
    session_log_path.mkdir(parents=True, exist_ok=True)
    if not kg_path.exists():
        knowledge_graph = KnowledgeGraph(json_path=kg_path, auto_load=False, auto_save=False)
        knowledge_graph.save()
        created["created"].append(str(kg_path))
    if vector_cfg.get("enabled", False):
        vector_path.mkdir(parents=True, exist_ok=True)
        created["created"].append(str(vector_path))
    created["created"].append(str(session_log_path))
    return created


def _write_default_config(path: Path, *, force: bool = False) -> Path:
    """Persist the default config template."""

    if path.exists() and not force:
        raise FileExistsError(f"{path} already exists. Use --force to overwrite it.")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(DEFAULT_CONFIG, sort_keys=False), encoding="utf-8")
    return path


def _build_knowledge_graph(
    config: dict[str, Any],
    *,
    config_path: Path,
    embedding_adapter: Any | None = None,
    vectorstore: Any | None = None,
) -> KnowledgeGraph:
    """Instantiate the KG using the resolved config paths."""

    return _shared_build_knowledge_graph(
        config,
        config_path=config_path,
        embedding_adapter=embedding_adapter,
        vectorstore=vectorstore,
        auto_load=True,
        auto_save=True,
    )


def _source_statuses(config: dict[str, Any]) -> list[dict[str, Any]]:
    """Summarize configured sources for CLI reporting."""

    statuses = []
    for source in config.get("agent", {}).get("sources", []):
        payload = dict(source)
        payload["status"] = "configured_optional_connector"
        statuses.append(payload)
    return statuses


def _retrieve_nodes(
    knowledge_graph: KnowledgeGraph,
    query: str,
    *,
    limit: int,
    status: str | None = None,
    node_type: str | None = None,
    min_confidence: float | None = None,
) -> list[KGNode]:
    """Retrieve KG nodes using semantic search when available."""

    if query.strip():
        candidates = knowledge_graph.semantic_query(query, top_k=limit)
        filtered = []
        for node in candidates:
            if status is not None and node.status != status:
                continue
            if node_type is not None and node.node_type != node_type:
                continue
            if min_confidence is not None and node.confidence < min_confidence:
                continue
            filtered.append(node)
        return filtered[:limit]
    return knowledge_graph.query(
        text=query,
        status=status,
        node_type=node_type,
        min_confidence=min_confidence,
    )[:limit]


def _summarize_query(
    query: str,
    nodes: list[KGNode],
    *,
    chat_client: Any | None,
) -> tuple[str | None, str | None]:
    """Generate a text answer from retrieved KG nodes when an LLM is configured."""

    if not nodes:
        return None, "no KG evidence matched the query"
    if chat_client is None:
        return None, "llm summarizer is not configured"
    context = [
        {
            "id": node.id,
            "label": node.label,
            "node_type": node.node_type,
            "content": node.content,
            "confidence": node.confidence,
            "metadata": node.metadata,
        }
        for node in nodes[:6]
    ]
    messages = [
        {
            "role": "system",
            "content": (
                "You answer user questions strictly from the provided Freeman knowledge-graph context. "
                "If the evidence is insufficient, say so explicitly."
            ),
        },
        {
            "role": "user",
            "content": json.dumps({"query": query, "kg_context": context}, ensure_ascii=False),
        },
    ]
    try:
        return str(chat_client.chat_text(messages, temperature=0.1, max_tokens=500)).strip(), None
    except Exception as exc:  # pragma: no cover - exercised only with live providers
        return None, str(exc)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="freeman")
    parser.add_argument("--config", "--config-path", dest="config_path", default="config.yaml")
    subparsers = parser.add_subparsers(dest="command", required=True)

    init_parser = subparsers.add_parser("init")
    _add_config_option(init_parser)
    init_parser.add_argument("--force", action="store_true")

    run_parser = subparsers.add_parser("run")
    _add_config_option(run_parser)
    run_parser.add_argument("--schema-path")
    run_parser.add_argument("--policies-path")

    ask_parser = subparsers.add_parser("ask")
    _add_config_option(ask_parser)
    ask_parser.add_argument("query")
    ask_parser.add_argument("--limit", type=int, default=5)
    ask_parser.add_argument("--status")
    ask_parser.add_argument("--node-type")
    ask_parser.add_argument("--min-confidence", type=float)

    status_parser = subparsers.add_parser("status")
    _add_config_option(status_parser)

    query_parser = subparsers.add_parser("query")
    _add_config_option(query_parser)
    query_parser.add_argument("--text")
    query_parser.add_argument("--limit", type=int)
    query_parser.add_argument("--status")
    query_parser.add_argument("--node-type")
    query_parser.add_argument("--min-confidence", type=float)

    export_parser = subparsers.add_parser("export-kg")
    _add_config_option(export_parser)
    export_parser.add_argument("format", choices=["html", "html-3d", "json-ld", "dot"])
    export_parser.add_argument("output_path")

    evolution_export_parser = subparsers.add_parser("export-kg-evolution")
    evolution_export_parser.add_argument("snapshot_source")
    evolution_export_parser.add_argument("output_path")

    reconcile_parser = subparsers.add_parser("reconcile")
    _add_config_option(reconcile_parser)
    reconcile_parser.add_argument("session_log_path")

    archive_parser = subparsers.add_parser("kg-archive")
    _add_config_option(archive_parser)
    archive_parser.add_argument("--node-id")
    archive_parser.add_argument("--reason", default="manual_archive")

    reindex_parser = subparsers.add_parser("kg-reindex")
    _add_config_option(reindex_parser)
    reindex_parser.add_argument("--batch-size", type=int, default=100)
    reindex_parser.add_argument("--use-stub-embeddings", action="store_true")

    override_param_parser = subparsers.add_parser("override-param")
    _add_config_option(override_param_parser)
    override_param_parser.add_argument("world_path")
    override_param_parser.add_argument("param_path")
    override_param_parser.add_argument("value")
    override_param_parser.add_argument("--output-path")

    override_sign_parser = subparsers.add_parser("override-sign")
    _add_config_option(override_sign_parser)
    override_sign_parser.add_argument("world_path")
    override_sign_parser.add_argument("edge_id")
    override_sign_parser.add_argument("expected_sign")
    override_sign_parser.add_argument("--output-path")

    rerun_parser = subparsers.add_parser("rerun-domain")
    _add_config_option(rerun_parser)
    rerun_parser.add_argument("world_path")
    rerun_parser.add_argument("--max-steps", type=int, default=5)
    rerun_parser.add_argument("--output-path")

    diff_parser = subparsers.add_parser("diff-domain")
    _add_config_option(diff_parser)
    diff_parser.add_argument("baseline_path")
    diff_parser.add_argument("current_path")
    diff_parser.add_argument("--output-path")

    identity_parser = subparsers.add_parser("identity")
    _add_config_option(identity_parser)
    identity_parser.add_argument("--narrative", action="store_true")

    explain_parser = subparsers.add_parser("explain")
    _add_config_option(explain_parser)
    explain_parser.add_argument("--trace-id", required=True)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "export-kg-evolution":
        exporter = KnowledgeGraphEvolutionExporter()
        output = exporter.export_html(args.snapshot_source, args.output_path)
        print(str(output))
        return 0

    config_path = Path(getattr(args, "config_path", "config.yaml")).resolve()

    if args.command == "init":
        written = _write_default_config(config_path, force=args.force)
        config = _load_config(written)
        storage = _bootstrap_storage(config, config_path=written)
        _print_json(
            {
                "status": "initialized",
                "config_path": str(written),
                **storage,
            }
        )
        return 0

    config = _load_config(config_path)
    storage = _bootstrap_storage(config, config_path=config_path)
    vectorstore = _build_vectorstore(config, config_path=config_path)
    needs_embeddings = (
        vectorstore is not None
        or args.command == "kg-reindex"
        or args.command == "ask"
        or (args.command == "query" and bool(getattr(args, "text", None)))
    )
    embedding_adapter = None
    embedding_backend = None
    if needs_embeddings:
        embedding_adapter, embedding_backend = _build_embedding_adapter(
            config,
            use_stub=getattr(args, "use_stub_embeddings", False),
        )
    knowledge_graph = _build_knowledge_graph(
        config,
        config_path=config_path,
        embedding_adapter=embedding_adapter,
        vectorstore=vectorstore,
    )

    if args.command == "run":
        agent_cfg = config.get("agent", {})
        bootstrap_cfg = agent_cfg.get("bootstrap", {})
        schema_path = args.schema_path or bootstrap_cfg.get("schema_path")
        policies_path = args.policies_path or bootstrap_cfg.get("policies_path")
        if schema_path:
            schema = _load_payload(schema_path)
            policies = _load_payload(policies_path) if policies_path else []
            pipeline = AnalysisPipeline(
                knowledge_graph=knowledge_graph,
                sim_config=_build_sim_config(config),
                config_path=config_path,
            )
            result = pipeline.run(schema, policies=policies)
            warnings = list(result.metadata.get("warnings", []))
            for message in warnings:
                print(message, file=sys.stderr)
            _print_json(
                {
                    "status": "completed",
                    "mode": "bootstrap_schema_run",
                    "knowledge_graph_path": str(knowledge_graph.json_path),
                    "domain_id": result.world.domain_id,
                    "dominant_outcome": result.dominant_outcome,
                    "confidence": result.simulation["confidence"],
                    "forecast_count": result.metadata.get("forecast_count", 0),
                    "epistemic_event_count": len(result.metadata.get("epistemic_event_ids", [])),
                    "warnings": warnings,
                    "simulation": result.simulation,
                }
            )
            return 0
        _print_json(
            {
                "status": "idle",
                "mode": "config_first_bootstrap",
                "knowledge_graph_path": str(knowledge_graph.json_path),
                "configured_source_count": len(agent_cfg.get("sources", [])),
                "sources": _source_statuses(config),
                "budget_usd_per_day": float(agent_cfg.get("budget_usd_per_day", 0.0)),
                "note": (
                    "Core Freeman does not ship live source connectors. "
                    "Install the optional connectors package to ingest RSS/arXiv/HTTP streams."
                ),
            }
        )
        return 0

    if args.command == "ask":
        runtime_artifacts = load_runtime_artifacts(config_path)
        chat_client, llm_error = _build_chat_client(config)
        payload = RuntimeAnswerEngine(runtime_artifacts).answer(
            args.query,
            limit=args.limit,
            chat_client=chat_client,
        )
        payload["llm_error"] = payload.get("llm_error") or llm_error
        payload["match_count"] = payload.get("count", 0)
        _print_json(payload)
        return 0

    if args.command == "status":
        payload = InterfaceAPI(knowledge_graph).get_status()
        payload["budget"] = _budget_status(config, config_path=config_path)
        payload["sources"] = _source_statuses(config)
        payload["vector_store_enabled"] = bool(config.get("memory", {}).get("vector_store", {}).get("enabled", False))
        payload["storage"] = storage
        _print_json(payload)
        return 0

    if args.command == "query":
        if args.text:
            runtime_artifacts = load_runtime_artifacts(config_path)
            payload = RuntimeQueryEngine(runtime_artifacts).semantic_query(
                args.text,
                limit=args.limit or int(config.get("memory", {}).get("retrieval_top_k", 15)),
            ).to_dict()
        else:
            payload = InterfaceAPI(knowledge_graph).post_query(
                text=args.text,
                status=args.status,
                node_type=args.node_type,
                min_confidence=args.min_confidence,
                limit=args.limit,
            )
        _print_json(payload)
        return 0

    if args.command == "export-kg":
        exporter = KnowledgeGraphExporter()
        if args.format == "html":
            output = exporter.export_html(knowledge_graph, args.output_path)
        elif args.format == "html-3d":
            output = exporter.export_html_3d(knowledge_graph, args.output_path)
        elif args.format == "json-ld":
            output = exporter.export_json_ld(knowledge_graph, args.output_path)
        else:
            output = exporter.export_dot(knowledge_graph, args.output_path)
        print(str(output))
        return 0

    if args.command == "reconcile":
        session_log = SessionLog.load(args.session_log_path)
        result = Reconciler().reconcile(knowledge_graph, session_log)
        _print_json(result.__dict__)
        return 0

    if args.command == "kg-archive":
        if args.node_id:
            archived = knowledge_graph.archive(args.node_id, reason=args.reason)
            _print_json(archived.snapshot())
            return 0
        archived_ids = []
        for node in knowledge_graph.nodes():
            if node.confidence < 0.15 and node.status != "archived":
                knowledge_graph.archive(node.id, reason=args.reason)
                archived_ids.append(node.id)
        _print_json({"archived_ids": archived_ids})
        return 0

    if args.command == "kg-reindex":
        if vectorstore is None:
            _print_json({"reembedded": 0, "synced": 0, "status": "vector_store_disabled"})
            return 0
        missing = [
            node
            for node in knowledge_graph.nodes(lazy_embed=False)
            if node.status != "archived" and node.content and not node.embedding
        ]
        reembedded = 0
        for batch_start in range(0, len(missing), args.batch_size):
            batch = missing[batch_start : batch_start + args.batch_size]
            if hasattr(embedding_adapter, "embed_many"):
                embeddings = embedding_adapter.embed_many(node.content for node in batch)
                for node, embedding in zip(batch, embeddings, strict=False):
                    node.embedding = list(embedding)
                    knowledge_graph.update_node(node)
                    reembedded += 1
                continue
            for node in batch:
                node.embedding = []
                knowledge_graph.update_node(node)
                reembedded += 1
        synced = vectorstore.sync_from_kg(knowledge_graph)
        knowledge_graph.save()
        _print_json(
            {
                "embedding_backend": embedding_backend,
                "reembedded": reembedded,
                "synced": synced,
                "vectorstore_path": str(vectorstore.path),
            }
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
        _print_json(payload)
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
        _print_json(payload)
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
        _print_json(payload)
        return 0

    if args.command == "diff-domain":
        before = _load_payload(args.baseline_path)
        after = _load_payload(args.current_path)
        domain_id = before.get("domain_id", after.get("domain_id", "unknown"))
        report = build_simulation_diff(domain_id=domain_id, before=before, after=after)
        if args.output_path:
            export_simulation_diff(report, args.output_path)
        _print_json(report.snapshot())
        return 0

    if args.command == "identity":
        state = build_identity_state(knowledge_graph)
        narrator = IdentityNarrator(_build_chat_client(config)[0] if args.narrative else None)
        payload = narrator.structured_snapshot(state)
        if args.narrative:
            payload["narrative"] = narrator.render(state)
        _print_json(payload)
        return 0

    if args.command == "explain":
        runtime_path = _runtime_path(config, config_path=config_path)
        checkpoint_path = runtime_path / "checkpoint.json"
        event_log = EventLog(_event_log_path(config, config_path=config_path))
        event = event_log.lookup(args.trace_id)
        if event is not None:
            trace_slice = [event]
        elif checkpoint_path.exists():
            checkpoint_state = CheckpointManager().load(checkpoint_path)
            trace_slice = [item for item in checkpoint_state.trace_state if item.event_id == args.trace_id]
        else:
            trace_slice = []
        renderer = ExplanationRenderer(_build_chat_client(config)[0])
        _print_json(
            {
                "trace_id": args.trace_id,
                "found": bool(trace_slice),
                "explanation": renderer.explain_trace(trace_slice) if trace_slice else None,
            }
        )
        return 0

    return 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
