"""OpenAI-compatible tool functions for Freeman."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List

import yaml

from freeman.agent.analysispipeline import AnalysisPipeline
from freeman.agent.consciousness import ConsciousState
from freeman.agent.forecastregistry import ForecastRegistry
from freeman.core.transition import step_world
from freeman.core.types import Policy
from freeman.core.world import WorldState
from freeman.domain.compiler import DomainCompiler
from freeman.exceptions import HardStopException
from freeman.game.runner import GameRunner, SimConfig
from freeman.memory.knowledgegraph import KGEdge, KGNode, KnowledgeGraph
from freeman.runtime.checkpoint import CheckpointManager
from freeman.verifier.level1 import level1_check
from freeman.verifier.level2 import level2_check
from freeman.verifier.report import VerificationReport

WORLD_REGISTRY: Dict[str, WorldState] = {}
TRAJECTORY_REGISTRY: Dict[str, List[Dict[str, Any]]] = {}


@dataclass
class RuntimeArtifacts:
    """Resolved runtime artifacts for persistent query tools."""

    config: dict[str, Any]
    config_path: Path
    runtime_path: Path
    kg_path: Path
    world_state_path: Path
    snapshot_dir: Path
    knowledge_graph: KnowledgeGraph
    pipeline: AnalysisPipeline
    world_state: WorldState | None


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _default_config_path() -> Path:
    return (_repo_root() / "config.yaml").resolve()


def _resolve_config_path(config_path: str | Path | None = None) -> Path:
    if config_path is None:
        return _default_config_path()
    candidate = Path(config_path).expanduser()
    return candidate.resolve() if candidate.is_absolute() else (_repo_root() / candidate).resolve()


def _resolve_path(base: Path, candidate: str | None, default: str) -> Path:
    target = Path(candidate or default).expanduser()
    return target if target.is_absolute() else (base / target).resolve()


def _load_config(config_path: str | Path | None = None) -> tuple[dict[str, Any], Path]:
    resolved = _resolve_config_path(config_path)
    if not resolved.exists():
        return {}, resolved
    payload = yaml.safe_load(resolved.read_text(encoding="utf-8")) or {}
    return dict(payload), resolved


def _build_sim_config(config: dict[str, Any]) -> SimConfig:
    sim_cfg = dict(config.get("sim", {}))
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


def _load_runtime_artifacts(config_path: str | Path | None = None) -> RuntimeArtifacts:
    config, resolved_config_path = _load_config(config_path)
    config_base = resolved_config_path.parent
    runtime_cfg = dict(config.get("runtime", {}))
    memory_cfg = dict(config.get("memory", {}))
    runtime_path = _resolve_path(config_base, runtime_cfg.get("runtime_path"), "./data/runtime")
    kg_path = _resolve_path(config_base, memory_cfg.get("json_path"), "./data/kg_state.json")
    world_state_path = runtime_path / "world_state.json"
    snapshot_cfg = dict(runtime_cfg.get("kg_snapshots", {}))
    snapshot_dir = _resolve_path(config_base, snapshot_cfg.get("path"), str(runtime_path / "kg_snapshots"))

    knowledge_graph = KnowledgeGraph(
        json_path=kg_path,
        auto_load=kg_path.exists(),
        auto_save=False,
    )
    forecasts_path = runtime_path / "forecasts.json"
    forecast_registry = ForecastRegistry(
        json_path=forecasts_path,
        auto_load=forecasts_path.exists(),
        auto_save=False,
    )
    pipeline = AnalysisPipeline(
        knowledge_graph=knowledge_graph,
        forecast_registry=forecast_registry,
        sim_config=_build_sim_config(config),
        config_path=resolved_config_path,
    )
    checkpoint_path = runtime_path / "checkpoint.json"
    if checkpoint_path.exists():
        checkpoint_state = CheckpointManager().load(checkpoint_path)
        pipeline.conscious_state = ConsciousState.from_dict(checkpoint_state.to_dict(), knowledge_graph)
    world_state = None
    if world_state_path.exists():
        world_state = WorldState.from_snapshot(json.loads(world_state_path.read_text(encoding="utf-8")))
    return RuntimeArtifacts(
        config=config,
        config_path=resolved_config_path,
        runtime_path=runtime_path,
        kg_path=kg_path,
        world_state_path=world_state_path,
        snapshot_dir=snapshot_dir,
        knowledge_graph=knowledge_graph,
        pipeline=pipeline,
        world_state=world_state,
    )


def _read_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def _snapshot_entries(snapshot_dir: Path) -> list[dict[str, Any]]:
    manifest_path = snapshot_dir / "manifest.jsonl"
    entries: list[dict[str, Any]] = []
    if manifest_path.exists():
        for line in manifest_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            payload = json.loads(line)
            payload["path"] = str(Path(payload["path"]).resolve())
            entries.append(payload)
        return sorted(
            entries,
            key=lambda item: (
                int(item.get("runtime_step", -1)),
                str(item.get("timestamp", "")),
                str(item.get("snapshot_id", "")),
            ),
        )
    for path in sorted(snapshot_dir.glob("*.json")):
        payload = _read_json(path, {})
        meta = dict(payload.get("snapshot_meta", {}))
        meta["path"] = str(path.resolve())
        entries.append(meta)
    return sorted(
        entries,
        key=lambda item: (
            int(item.get("runtime_step", -1)),
            str(item.get("timestamp", "")),
            str(item.get("snapshot_id", "")),
        ),
    )


def _contains(haystack: str, needle: str | None) -> bool:
    if not needle:
        return True
    return str(needle).strip().lower() in str(haystack).lower()


def _node_matches(knowledge_graph: KnowledgeGraph, node_id: str, query: str | None) -> bool:
    if not query:
        return True
    node = knowledge_graph.get_node(node_id, lazy_embed=False)
    label = node.label if node is not None else ""
    content = node.content if node is not None else ""
    haystack = " ".join([node_id, label, content])
    return _contains(haystack, query)


def _edge_matches(
    knowledge_graph: KnowledgeGraph,
    edge: KGEdge,
    *,
    source: str | None = None,
    target: str | None = None,
    relation_type: str | None = None,
) -> bool:
    if relation_type and str(edge.relation_type) != str(relation_type):
        return False
    if not _node_matches(knowledge_graph, edge.source, source):
        return False
    if not _node_matches(knowledge_graph, edge.target, target):
        return False
    return True


def _edge_dict(knowledge_graph: KnowledgeGraph, edge: KGEdge) -> dict[str, Any]:
    source_node = knowledge_graph.get_node(edge.source, lazy_embed=False)
    target_node = knowledge_graph.get_node(edge.target, lazy_embed=False)
    payload = edge.snapshot()
    payload["source_label"] = source_node.label if source_node is not None else edge.source
    payload["target_label"] = target_node.label if target_node is not None else edge.target
    payload["runtime_step"] = int((edge.metadata or {}).get("runtime_step", -1))
    return payload


def _node_snapshot(node: KGNode) -> dict[str, Any]:
    payload = node.snapshot()
    payload["cluster_topics"] = list(node.metadata.get("payload", {}).get("cluster_topics", []))
    return payload


def _next_world_id(domain_id: str) -> str:
    """Return a new in-memory world id."""

    return f"{domain_id}:{len(WORLD_REGISTRY) + 1}"


def _coerce_policies(policies: Iterable[Policy | Dict[str, Any]]) -> List[Policy]:
    """Convert policy-like inputs into ``Policy`` instances."""

    return [policy if isinstance(policy, Policy) else Policy.from_snapshot(policy) for policy in policies]


def freeman_compile_domain(schema: Dict[str, Any]) -> Dict[str, Any]:
    """Compile ``schema`` into a world and store it in the in-memory registry."""

    compiler = DomainCompiler()
    world = compiler.compile(schema)
    world_id = _next_world_id(world.domain_id)
    WORLD_REGISTRY[world_id] = world
    TRAJECTORY_REGISTRY[world_id] = [world.snapshot()]
    return {"world_id": world_id, "validation_result": {"valid": True, "domain_id": world.domain_id}}


def freeman_run_simulation(
    world_id: str,
    policies: Iterable[Policy | Dict[str, Any]],
    max_steps: int = 50,
    seed: int = 42,
) -> str:
    """Run a simulation for a compiled world and return serialized JSON."""

    world = WORLD_REGISTRY[world_id].clone()
    world.seed = seed
    config = SimConfig(max_steps=max_steps, seed=seed)
    runner = GameRunner(config)
    result = runner.run(world, _coerce_policies(policies))
    TRAJECTORY_REGISTRY[world_id] = result.trajectory
    WORLD_REGISTRY[world_id] = WorldState.from_snapshot(result.trajectory[-1])
    return result.to_json()


def freeman_get_world_state(world_id: str, t: int = -1) -> Dict[str, Any]:
    """Return a stored world snapshot, defaulting to the latest timestep."""

    trajectory = TRAJECTORY_REGISTRY.get(world_id)
    if trajectory:
        index = len(trajectory) - 1 if t == -1 else t
        return trajectory[index]
    return WORLD_REGISTRY[world_id].snapshot()


def freeman_verify_domain(world_id: str, levels: Iterable[int] = (0, 1, 2)) -> Dict[str, Any]:
    """Run selected verification levels for a compiled world."""

    world = WORLD_REGISTRY[world_id].clone()
    requested_levels = list(levels)
    violations = []

    if 0 in requested_levels:
        try:
            _, level0_violations = step_world(world.clone(), [])
            violations.extend(level0_violations)
        except HardStopException as exc:
            violations.extend(exc.violations)

    if 1 in requested_levels:
        violations.extend(level1_check(world.clone(), SimConfig(seed=world.seed)))

    if 2 in requested_levels:
        violations.extend(
            level2_check(world.clone(), world.causal_dag, base_delta=SimConfig().level2_shock_delta)
        )

    report = VerificationReport(
        world_id=world_id,
        domain_id=world.domain_id,
        levels_run=requested_levels,
        violations=violations,
        passed=not any(violation.severity == "hard" for violation in violations),
        metadata={"violation_count": len(violations)},
    )
    return report.snapshot()


def freeman_get_runtime_summary(config_path: str = "config.yaml") -> dict[str, Any]:
    """Return a compact summary of the persisted daemon state."""

    artifacts = _load_runtime_artifacts(config_path)
    forecasts = artifacts.pipeline.list_forecasts()
    forecast_counts: dict[str, int] = {}
    for summary in forecasts:
        forecast_counts[summary.status] = forecast_counts.get(summary.status, 0) + 1

    event_log_path = artifacts.runtime_path / "event_log.jsonl"
    pending_signals = _read_json(artifacts.runtime_path / "pending_signals.json", {"signals": []})
    snapshots = _snapshot_entries(artifacts.snapshot_dir) if artifacts.snapshot_dir.exists() else []

    return {
        "config_path": str(artifacts.config_path),
        "runtime_path": str(artifacts.runtime_path),
        "kg_path": str(artifacts.kg_path),
        "world_state_path": str(artifacts.world_state_path),
        "domain_id": artifacts.world_state.domain_id if artifacts.world_state is not None else None,
        "world_t": int(artifacts.world_state.t) if artifacts.world_state is not None else None,
        "runtime_step": int(artifacts.world_state.runtime_step) if artifacts.world_state is not None else None,
        "kg_node_count": len(artifacts.knowledge_graph.nodes(lazy_embed=False)),
        "kg_edge_count": len(artifacts.knowledge_graph.edges()),
        "forecast_counts": forecast_counts,
        "pending_signal_count": len(pending_signals.get("signals", [])),
        "event_count": sum(1 for line in event_log_path.read_text(encoding="utf-8").splitlines() if line.strip())
        if event_log_path.exists()
        else 0,
        "snapshot_count": len(snapshots),
        "snapshots_enabled": bool(artifacts.config.get("runtime", {}).get("kg_snapshots", {}).get("enabled", False)),
    }


def freeman_query_forecasts(
    config_path: str = "config.yaml",
    status: str | None = None,
    outcome_id: str | None = None,
    limit: int = 20,
) -> dict[str, Any]:
    """Return saved forecasts for one runtime."""

    artifacts = _load_runtime_artifacts(config_path)
    summaries = []
    for summary in artifacts.pipeline.list_forecasts(status=status):
        if outcome_id and not _contains(summary.outcome_id, outcome_id):
            continue
        summaries.append(summary.to_dict())
    return {
        "count": len(summaries),
        "items": summaries[: max(int(limit), 1)],
    }


def freeman_explain_forecast(config_path: str = "config.yaml", forecast_id: str = "") -> dict[str, Any]:
    """Return one structured causal explanation for a saved forecast."""

    if not forecast_id:
        raise ValueError("forecast_id is required.")
    artifacts = _load_runtime_artifacts(config_path)
    explanation = artifacts.pipeline.explain_forecast(str(forecast_id))
    payload = explanation.to_dict()
    payload["text"] = explanation.to_text()
    return payload


def freeman_query_anomalies(config_path: str = "config.yaml", limit: int = 20) -> dict[str, Any]:
    """Return anomaly candidates plus ontology-gap traits from the KG."""

    artifacts = _load_runtime_artifacts(config_path)
    anomaly_candidates = [
        _node_snapshot(node)
        for node in artifacts.knowledge_graph.query(node_type="anomaly_candidate")
    ][: max(int(limit), 1)]
    ontology_gap_traits = [
        _node_snapshot(node)
        for node in artifacts.knowledge_graph.query(
            node_type="identity_trait",
            metadata_filters={"payload.trait_key": "ontology_gap"},
        )
    ][: max(int(limit), 1)]
    return {
        "anomaly_candidates": anomaly_candidates,
        "ontology_gap_traits": ontology_gap_traits,
    }


def freeman_query_causal_edges(
    config_path: str = "config.yaml",
    source: str | None = None,
    target: str | None = None,
    relation_type: str | None = None,
    limit: int = 20,
) -> dict[str, Any]:
    """Return current KG causal edges, optionally filtered by source, target, or relation."""

    artifacts = _load_runtime_artifacts(config_path)
    relation_filter = relation_type
    allowed_relations = {"causes", "propagates_to", "threshold_exceeded"}
    edges = [
        _edge_dict(artifacts.knowledge_graph, edge)
        for edge in artifacts.knowledge_graph.edges()
        if (relation_filter or edge.relation_type) in allowed_relations
        and _edge_matches(
            artifacts.knowledge_graph,
            edge,
            source=source,
            target=target,
            relation_type=relation_filter,
        )
    ]
    edges.sort(
        key=lambda item: (
            int(item.get("runtime_step", -1)),
            str(item.get("updated_at", "")),
            str(item.get("id", "")),
        ),
        reverse=True,
    )
    return {
        "count": len(edges),
        "items": edges[: max(int(limit), 1)],
    }


def freeman_trace_relation_learning(
    config_path: str = "config.yaml",
    source: str = "",
    target: str = "",
    relation_type: str | None = None,
    last_n_steps: int = 10,
) -> dict[str, Any]:
    """Trace how a relation appears across recent KG snapshots."""

    if not source or not target:
        raise ValueError("source and target are required.")

    artifacts = _load_runtime_artifacts(config_path)
    current_step = int(artifacts.world_state.runtime_step) if artifacts.world_state is not None else -1
    min_runtime_step = current_step - max(int(last_n_steps), 0) + 1 if current_step >= 0 else None
    timeline: list[dict[str, Any]] = []

    for entry in _snapshot_entries(artifacts.snapshot_dir):
        runtime_step = int(entry.get("runtime_step", -1))
        if min_runtime_step is not None and runtime_step < min_runtime_step:
            continue
        snapshot_path = Path(entry["path"]).resolve()
        snapshot_payload = _read_json(snapshot_path, {})
        snapshot_graph = KnowledgeGraph(auto_load=False, auto_save=False)
        snapshot_graph.load(snapshot_path)
        matches = [
            _edge_dict(snapshot_graph, edge)
            for edge in snapshot_graph.edges()
            if _edge_matches(
                snapshot_graph,
                edge,
                source=source,
                target=target,
                relation_type=relation_type,
            )
        ]
        if not matches:
            continue
        timeline.append(
            {
                "snapshot_id": entry.get("snapshot_id") or snapshot_payload.get("snapshot_meta", {}).get("snapshot_id"),
                "timestamp": entry.get("timestamp") or snapshot_payload.get("snapshot_meta", {}).get("timestamp"),
                "runtime_step": runtime_step,
                "reason": entry.get("reason") or snapshot_payload.get("snapshot_meta", {}).get("reason"),
                "signal_id": entry.get("signal_id") or snapshot_payload.get("snapshot_meta", {}).get("signal_id"),
                "match_count": len(matches),
                "matches": matches,
            }
        )

    current_matches = [
        _edge_dict(artifacts.knowledge_graph, edge)
        for edge in artifacts.knowledge_graph.edges()
        if _edge_matches(
            artifacts.knowledge_graph,
            edge,
            source=source,
            target=target,
            relation_type=relation_type,
        )
    ]
    current_matches.sort(
        key=lambda item: (
            int(item.get("runtime_step", -1)),
            str(item.get("updated_at", "")),
            str(item.get("id", "")),
        ),
        reverse=True,
    )
    return {
        "source_query": source,
        "target_query": target,
        "relation_type": relation_type,
        "last_n_steps": int(last_n_steps),
        "current_runtime_step": current_step,
        "timeline": timeline,
        "current_matches": current_matches,
    }


FREEMAN_TOOLS = [
    {
        "name": "freeman_compile_domain",
        "description": "Compile a domain schema into a simulation world. Returns world_id.",
        "parameters": {
            "type": "object",
            "properties": {
                "schema": {"type": "object", "description": "Domain schema JSON"},
            },
            "required": ["schema"],
        },
    },
    {
        "name": "freeman_run_simulation",
        "description": "Run simulation on a compiled world. Returns SimResult JSON.",
        "parameters": {
            "type": "object",
            "properties": {
                "world_id": {"type": "string"},
                "policies": {"type": "array"},
                "max_steps": {"type": "integer", "default": 50},
                "seed": {"type": "integer", "default": 42},
            },
            "required": ["world_id", "policies"],
        },
    },
    {
        "name": "freeman_get_world_state",
        "description": "Get current state of a world at timestep t.",
        "parameters": {
            "type": "object",
            "properties": {
                "world_id": {"type": "string"},
                "t": {"type": "integer", "default": -1},
            },
            "required": ["world_id"],
        },
    },
    {
        "name": "freeman_verify_domain",
        "description": "Run verification levels on a compiled world.",
        "parameters": {
            "type": "object",
            "properties": {
                "world_id": {"type": "string"},
                "levels": {"type": "array", "items": {"type": "integer"}, "default": [0, 1, 2]},
            },
            "required": ["world_id"],
        },
    },
    {
        "name": "freeman_get_runtime_summary",
        "description": "Summarize the current Freeman daemon state from persisted runtime artifacts.",
        "parameters": {
            "type": "object",
            "properties": {
                "config_path": {
                    "type": "string",
                    "default": "config.yaml",
                    "description": "Path to the Freeman config that points at the runtime artifacts.",
                },
            },
        },
    },
    {
        "name": "freeman_query_forecasts",
        "description": "List saved runtime forecasts, filtered by status or outcome id.",
        "parameters": {
            "type": "object",
            "properties": {
                "config_path": {"type": "string", "default": "config.yaml"},
                "status": {"type": "string", "description": "Optional forecast status filter."},
                "outcome_id": {"type": "string", "description": "Optional outcome-id substring filter."},
                "limit": {"type": "integer", "default": 20},
            },
        },
    },
    {
        "name": "freeman_explain_forecast",
        "description": "Explain one saved forecast and its causal path.",
        "parameters": {
            "type": "object",
            "properties": {
                "config_path": {"type": "string", "default": "config.yaml"},
                "forecast_id": {"type": "string"},
            },
            "required": ["forecast_id"],
        },
    },
    {
        "name": "freeman_query_anomalies",
        "description": "Return anomaly candidates plus escalated ontology-gap traits from the current KG.",
        "parameters": {
            "type": "object",
            "properties": {
                "config_path": {"type": "string", "default": "config.yaml"},
                "limit": {"type": "integer", "default": 20},
            },
        },
    },
    {
        "name": "freeman_query_causal_edges",
        "description": "Return current KG causal edges, optionally filtered by source, target, or relation type.",
        "parameters": {
            "type": "object",
            "properties": {
                "config_path": {"type": "string", "default": "config.yaml"},
                "source": {"type": "string", "description": "Optional source node id or label substring."},
                "target": {"type": "string", "description": "Optional target node id or label substring."},
                "relation_type": {"type": "string", "description": "Optional exact relation filter."},
                "limit": {"type": "integer", "default": 20},
            },
        },
    },
    {
        "name": "freeman_trace_relation_learning",
        "description": "Trace how a relation between two concepts appeared across recent KG snapshots.",
        "parameters": {
            "type": "object",
            "properties": {
                "config_path": {"type": "string", "default": "config.yaml"},
                "source": {"type": "string"},
                "target": {"type": "string"},
                "relation_type": {"type": "string", "description": "Optional exact relation filter."},
                "last_n_steps": {"type": "integer", "default": 10},
            },
            "required": ["source", "target"],
        },
    },
]

FREEMAN_TOOL_FUNCTIONS: dict[str, Callable[..., Any]] = {
    "freeman_compile_domain": freeman_compile_domain,
    "freeman_run_simulation": freeman_run_simulation,
    "freeman_get_world_state": freeman_get_world_state,
    "freeman_verify_domain": freeman_verify_domain,
    "freeman_get_runtime_summary": freeman_get_runtime_summary,
    "freeman_query_forecasts": freeman_query_forecasts,
    "freeman_explain_forecast": freeman_explain_forecast,
    "freeman_query_anomalies": freeman_query_anomalies,
    "freeman_query_causal_edges": freeman_query_causal_edges,
    "freeman_trace_relation_learning": freeman_trace_relation_learning,
}


def resolve_tool(name: str) -> Callable[..., Any]:
    """Resolve a Freeman tool by name."""

    try:
        return FREEMAN_TOOL_FUNCTIONS[str(name)]
    except KeyError as exc:  # pragma: no cover - exercised via callers
        raise KeyError(f"Unknown Freeman tool: {name}") from exc


def invoke_tool(name: str, arguments: dict[str, Any] | None = None) -> Any:
    """Invoke one Freeman tool by name."""

    return resolve_tool(name)(**dict(arguments or {}))


__all__ = [
    "FREEMAN_TOOLS",
    "FREEMAN_TOOL_FUNCTIONS",
    "freeman_compile_domain",
    "freeman_run_simulation",
    "freeman_get_world_state",
    "freeman_verify_domain",
    "freeman_get_runtime_summary",
    "freeman_query_forecasts",
    "freeman_explain_forecast",
    "freeman_query_anomalies",
    "freeman_query_causal_edges",
    "freeman_trace_relation_learning",
    "invoke_tool",
    "resolve_tool",
]
