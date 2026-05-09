"""Transparent benchmark for ETL bootstrap + KG growth on random live RSS news."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
import random
import sys
import time
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    from freeman_connectors import build_signal_source
except ImportError:
    connectors_root = REPO_ROOT / "packages" / "freeman-connectors"
    if str(connectors_root) not in sys.path:
        sys.path.insert(0, str(connectors_root))
    from freeman_connectors import build_signal_source

from freeman.agent.signalingestion import Signal
from freeman.interface.kgexport import KnowledgeGraphExporter
from freeman.memory.knowledgegraph import KnowledgeGraph
from freeman.runtime import stream_runtime

LOGGER = logging.getLogger("random_news_etl_benchmark")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark Freeman ETL on a random live RSS news sample.")
    parser.add_argument("--config-path", default="config.climate.clean.yaml")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--sample-size", type=int, default=12)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-sources", type=int, default=6)
    parser.add_argument("--per-source-max", type=int, default=12)
    parser.add_argument("--bootstrap-max-retries", type=int, default=4)
    parser.add_argument("--bootstrap-trial-steps", type=int, default=2)
    parser.add_argument("--model", default="qwen2.5-coder:14b")
    parser.add_argument("--ollama-base-url", default="http://127.0.0.1:11434")
    parser.add_argument("--llm-timeout-seconds", type=float, default=45.0)
    parser.add_argument("--log-level", default="INFO")
    parser.add_argument("--include-watch", action="store_true")
    parser.add_argument("--allow-fallback", action="store_true")
    return parser.parse_args()


def _load_config(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"Config at {path} must be a mapping.")
    return payload


def _rss_sources(config: dict[str, Any], *, max_sources: int, per_source_max: int) -> list[dict[str, Any]]:
    sources = list(((config.get("agent") or {}).get("sources") or []))
    rss_sources = [dict(source) for source in sources if str(source.get("type", "")).strip().lower() == "rss"]
    selected = rss_sources[: max(int(max_sources), 1)]
    for source in selected:
        source["max_entries"] = min(int(source.get("max_entries", per_source_max)), int(per_source_max))
    if not selected:
        raise RuntimeError("No RSS sources configured for benchmark.")
    return selected


def _signal_snapshot(signal: Signal) -> dict[str, Any]:
    return {
        "signal_id": signal.signal_id,
        "source_type": signal.source_type,
        "topic": signal.topic,
        "text": signal.text,
        "entities": list(signal.entities),
        "sentiment": float(signal.sentiment),
        "timestamp": signal.timestamp,
        "metadata": signal.metadata,
    }


def _fetch_signal_pool(source_specs: list[dict[str, Any]]) -> tuple[list[Signal], list[dict[str, Any]]]:
    pool: list[Signal] = []
    fetch_log: list[dict[str, Any]] = []
    for index, source_spec in enumerate(source_specs, start=1):
        source = build_signal_source(source_spec)
        started = time.perf_counter()
        signals = source.fetch()
        elapsed = time.perf_counter() - started
        for signal in signals:
            signal.metadata = {
                **dict(signal.metadata),
                "benchmark_source_index": index,
                "benchmark_feed_url": str(source_spec.get("url", "")),
                "benchmark_default_topic": str(source_spec.get("default_topic", "")),
            }
        pool.extend(signals)
        fetch_log.append(
            {
                "source_index": index,
                "source_type": str(source_spec.get("source_type", source_spec.get("type", "rss"))),
                "url": str(source_spec.get("url", "")),
                "fetched_count": len(signals),
                "elapsed_seconds": round(elapsed, 4),
            }
        )
    deduped = list({signal.signal_id: signal for signal in pool}.values())
    return deduped, fetch_log


def _sample_signals(pool: list[Signal], *, sample_size: int, seed: int) -> list[Signal]:
    if len(pool) < sample_size:
        raise RuntimeError(f"Requested sample_size={sample_size}, but fetched only {len(pool)} unique signals.")
    rng = random.Random(seed)
    indices = list(range(len(pool)))
    rng.shuffle(indices)
    chosen = sorted(indices[:sample_size])
    return [pool[index] for index in chosen]


def _brief_from_signals(signals: list[Signal]) -> str:
    lines = [
        "Build a compact Freeman domain that explains the following observed live news cluster.",
        "Treat the bullet list as empirical evidence from a random RSS sample, not as an opinionated narrative.",
        "Prefer stable latent drivers, actor/resource separation, and causal links that can explain repeated topics across sources.",
        "",
        "Observed signals:",
    ]
    for index, signal in enumerate(signals, start=1):
        link = str(signal.metadata.get("link", "") or signal.metadata.get("benchmark_feed_url", "")).strip()
        title = str(signal.text).splitlines()[0].strip()
        source_label = str(signal.metadata.get("feed_title", signal.source_type)).strip()
        lines.append(
            f"{index}. topic={signal.topic} | source={source_label} | timestamp={signal.timestamp} | title={title}"
        )
        if link:
            lines.append(f"   link={link}")
    lines.extend(
        [
            "",
            "Benchmark constraints:",
            "- Keep the domain compact and executable under Freeman verifier checks.",
            "- Use latent resources that can absorb multiple observed signals rather than one node per headline.",
            "- Outcomes should reflect the dominant regimes implied by the sample.",
        ]
    )
    return "\n".join(lines).strip()


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")


def _build_benchmark_config(
    base_config: dict[str, Any],
    *,
    output_dir: Path,
    domain_brief: str,
    model: str,
    ollama_base_url: str,
    llm_timeout_seconds: float,
    bootstrap_max_retries: int,
    bootstrap_trial_steps: int,
    allow_fallback: bool,
    config_base: Path,
) -> dict[str, Any]:
    config = json.loads(json.dumps(base_config, ensure_ascii=False))
    config.setdefault("agent", {})
    config["agent"].setdefault("bootstrap", {})
    config["agent"]["sources"] = []
    config["agent"]["bootstrap"]["mode"] = "llm_synthesize"
    config["agent"]["bootstrap"]["domain_brief"] = domain_brief
    config["agent"]["bootstrap"]["domain_brief_path"] = ""
    config["agent"]["bootstrap"]["max_retries"] = int(bootstrap_max_retries)
    config["agent"]["bootstrap"]["trial_steps"] = int(bootstrap_trial_steps)
    fallback_candidate = str(config["agent"]["bootstrap"].get("fallback_schema_path", "")).strip()
    if allow_fallback and fallback_candidate:
        fallback_path = Path(fallback_candidate)
        config["agent"]["bootstrap"]["fallback_schema_path"] = str(
            fallback_path if fallback_path.is_absolute() else (config_base / fallback_path).resolve()
        )
    else:
        config["agent"]["bootstrap"]["fallback_schema_path"] = ""
    config.setdefault("runtime", {})
    config["runtime"]["runtime_path"] = str((output_dir / "runtime").resolve())
    config["runtime"]["event_log_path"] = str((output_dir / "runtime" / "event_log.jsonl").resolve())
    config["runtime"].setdefault("kg_snapshots", {})
    config["runtime"]["kg_snapshots"]["enabled"] = True
    config["runtime"]["kg_snapshots"]["path"] = str((output_dir / "runtime" / "kg_snapshots").resolve())
    config.setdefault("memory", {})
    config["memory"]["json_path"] = str((output_dir / "kg_state.json").resolve())
    config.setdefault("llm", {})
    config["llm"]["provider"] = "ollama"
    config["llm"]["model"] = model
    config["llm"]["base_url"] = ollama_base_url
    config["llm"]["timeout_seconds"] = float(llm_timeout_seconds)
    return config


def _runtime_args(config_path: Path, *, model: str, ollama_base_url: str, include_watch: bool) -> argparse.Namespace:
    parser = stream_runtime._build_parser(default_config_path=str(config_path))
    argv = [
        "--config-path",
        str(config_path),
        "--bootstrap-mode",
        "llm_synthesize",
        "--model",
        model,
        "--ollama-base-url",
        ollama_base_url,
        "--hours",
        "0",
        "--analysis-interval-seconds",
        "0.1",
        "--max-signals-per-poll",
        "0",
        "--no-resume",
        "--log-level",
        "INFO",
    ]
    if include_watch:
        argv.append("--include-watch")
    return parser.parse_args(argv)


def _node_type_counts(knowledge_graph: KnowledgeGraph) -> dict[str, int]:
    counts: dict[str, int] = {}
    for node in knowledge_graph.nodes():
        counts[node.node_type] = counts.get(node.node_type, 0) + 1
    return dict(sorted(counts.items()))


def _edge_type_counts(knowledge_graph: KnowledgeGraph) -> dict[str, int]:
    counts: dict[str, int] = {}
    for edge in knowledge_graph.edges():
        counts[edge.relation_type] = counts.get(edge.relation_type, 0) + 1
    return dict(sorted(counts.items()))


def _phase_counts(bootstrap_attempts: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for attempt in bootstrap_attempts:
        phase = str(attempt.get("etl_phase", attempt.get("phase", "unknown")))
        counts[phase] = counts.get(phase, 0) + 1
    return dict(sorted(counts.items()))


def _report_text(summary: dict[str, Any]) -> str:
    metrics = summary["metrics"]
    lines = [
        "# Random News ETL Benchmark",
        "",
        "## Setup",
        f"- model: `{summary['model']}`",
        f"- sample size: `{summary['sample_size']}`",
        f"- random seed: `{summary['seed']}`",
        f"- pool size: `{summary['pool_size']}`",
        f"- source count: `{summary['source_count']}`",
        "",
        "## Formal Metrics",
        r"- compile success indicator: \( I_c = 1[\text{bootstrap_mode} = \text{llm\_synthesize}] \)",
        r"- ETL repair burden: \( b = \max(|A| - 2, 0) \), where \(A\) is `bootstrap_attempts` and the first two entries are baseline `skeleton` and `edges`",
        r"- world update rate: \( r_u = n_{updated} / n_{signals} \)",
        r"- graph growth per signal: \( g_V = \Delta |V| / n_{processed},\; g_E = \Delta |E| / n_{processed} \)",
        "",
        "## Results",
        f"- bootstrap mode: `{summary['bootstrap_mode']}`",
        f"- ETL time seconds: `{metrics['etl_seconds']:.3f}`",
        f"- total processing time seconds: `{metrics['total_seconds']:.3f}`",
        f"- bootstrap attempts: `{metrics['bootstrap_attempt_count']}`",
        f"- ETL phase counts: `{json.dumps(summary['etl_phase_counts'], ensure_ascii=False)}`",
        f"- schema sizes: `{json.dumps(summary['schema_sizes'], ensure_ascii=False)}`",
        f"- processed signals: `{metrics['processed_signals']}`",
        f"- world updates: `{metrics['world_updates']}`",
        f"- update rate: `{metrics['world_update_rate']:.3f}`",
        f"- filtered out: `{metrics['filtered_out']}`",
        f"- KG nodes: `{metrics['kg_nodes']}`",
        f"- KG edges: `{metrics['kg_edges']}`",
        f"- nodes per processed signal: `{metrics['nodes_per_processed_signal']:.3f}`",
        f"- edges per processed signal: `{metrics['edges_per_processed_signal']:.3f}`",
        "",
        "## Artifacts",
        "- `raw_signal_pool.json`",
        "- `selected_signals.json`",
        "- `domain_brief.md`",
        "- `benchmark_config.yaml`",
        "- `runtime/bootstrap_package.json`",
        "- `benchmark_summary.json`",
        "- `report.md`",
        "- `kg_3d.html`",
        "- `kg.json`",
    ]
    return "\n".join(lines) + "\n"


def main() -> int:
    args = _parse_args()
    logging.basicConfig(
        level=getattr(logging, str(args.log_level).upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    config_path = Path(args.config_path).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    base_config = _load_config(config_path)
    rss_sources = _rss_sources(base_config, max_sources=args.max_sources, per_source_max=args.per_source_max)
    pool, fetch_log = _fetch_signal_pool(rss_sources)
    selected_signals = _sample_signals(pool, sample_size=int(args.sample_size), seed=int(args.seed))
    domain_brief = _brief_from_signals(selected_signals)

    _write_json(output_dir / "fetch_log.json", fetch_log)
    _write_json(output_dir / "raw_signal_pool.json", [_signal_snapshot(signal) for signal in pool])
    _write_json(output_dir / "selected_signals.json", [_signal_snapshot(signal) for signal in selected_signals])
    (output_dir / "domain_brief.md").write_text(domain_brief + "\n", encoding="utf-8")

    benchmark_config = _build_benchmark_config(
        base_config,
        output_dir=output_dir,
        domain_brief=domain_brief,
        model=str(args.model),
        ollama_base_url=str(args.ollama_base_url),
        llm_timeout_seconds=float(args.llm_timeout_seconds),
        bootstrap_max_retries=int(args.bootstrap_max_retries),
        bootstrap_trial_steps=int(args.bootstrap_trial_steps),
        allow_fallback=bool(args.allow_fallback),
        config_base=config_path.parent,
    )
    benchmark_config_path = output_dir / "benchmark_config.yaml"
    benchmark_config_path.write_text(yaml.safe_dump(benchmark_config, sort_keys=False), encoding="utf-8")

    runtime_args = _runtime_args(
        benchmark_config_path,
        model=str(args.model),
        ollama_base_url=str(args.ollama_base_url),
        include_watch=bool(args.include_watch),
    )
    paths = stream_runtime._resolve_runtime_paths(runtime_args, benchmark_config, benchmark_config_path)
    storage = stream_runtime._initialize_runtime_storage(runtime_args, paths.runtime_path, paths.event_log_path)

    start_total = time.perf_counter()
    start_etl = time.perf_counter()
    try:
        bootstrap = stream_runtime._bootstrap(args=runtime_args, config=benchmark_config, paths=paths, storage=storage)
    except Exception as exc:  # noqa: BLE001
        failure_summary = {
            "status": "bootstrap_failed",
            "model": str(args.model),
            "seed": int(args.seed),
            "sample_size": len(selected_signals),
            "pool_size": len(pool),
            "source_count": len(rss_sources),
            "error": str(exc),
            "artifacts": {
                "output_dir": str(output_dir),
                "benchmark_config_path": str(benchmark_config_path),
            },
        }
        _write_json(output_dir / "benchmark_summary.json", failure_summary)
        (output_dir / "report.md").write_text("# Random News ETL Benchmark\n\nBootstrap failed.\n", encoding="utf-8")
        LOGGER.error("Benchmark bootstrap failed: %s", exc)
        return 1
    etl_seconds = time.perf_counter() - start_etl
    ctx = stream_runtime._build_runtime_context(
        args=runtime_args,
        config=benchmark_config,
        paths=paths,
        storage=storage,
        bootstrap=bootstrap,
        default_sources=[],
        default_keywords=None,
    )

    per_signal_results: list[dict[str, Any]] = []
    for signal in selected_signals:
        before_nodes = len(ctx.pipeline.knowledge_graph.nodes())
        before_edges = len(ctx.pipeline.knowledge_graph.edges())
        before_t = int(ctx.current_world.t)
        before_runtime_step = int(ctx.current_world.runtime_step)
        started = time.perf_counter()
        signal_result = stream_runtime._process_one_signal(signal, ctx=ctx)
        elapsed = time.perf_counter() - started
        ctx.stats["signals_seen"] += 1
        ctx.stats["signals_committed"] += signal_result.processed
        ctx.stats["world_updates"] += signal_result.updated
        ctx.stats["world_update_failures"] += signal_result.update_failures
        ctx.stats["verified_forecasts"] += signal_result.verified_forecasts
        ctx.stats["watch_skipped"] += signal_result.skipped_watch
        ctx.stats["filtered_out_count"] += signal_result.filtered_out
        per_signal_results.append(
            {
                "signal_id": signal.signal_id,
                "topic": signal.topic,
                "processed": int(signal_result.processed),
                "updated": int(signal_result.updated),
                "update_failures": int(signal_result.update_failures),
                "filtered_out": int(signal_result.filtered_out),
                "elapsed_seconds": round(elapsed, 4),
                "world_t_before": before_t,
                "world_t_after": int(ctx.current_world.t),
                "runtime_step_before": before_runtime_step,
                "runtime_step_after": int(ctx.current_world.runtime_step),
                "kg_nodes_delta": len(ctx.pipeline.knowledge_graph.nodes()) - before_nodes,
                "kg_edges_delta": len(ctx.pipeline.knowledge_graph.edges()) - before_edges,
            }
        )
    total_seconds = time.perf_counter() - start_total

    kg = ctx.pipeline.knowledge_graph
    exporter = KnowledgeGraphExporter()
    exporter.export_html_3d(kg, output_dir / "kg_3d.html")
    kg.export_json(output_dir / "kg.json")

    package_payload = json.loads((paths.runtime_path / "bootstrap_package.json").read_text(encoding="utf-8"))
    bootstrap_attempts = list(package_payload.get("bootstrap_attempts", []))
    processed_signals = max(int(ctx.stats["signals_committed"]), 1)
    summary = {
        "model": bootstrap.model_name,
        "llm_provider": bootstrap.provider,
        "bootstrap_mode": package_payload.get("bootstrap_mode"),
        "seed": int(args.seed),
        "sample_size": len(selected_signals),
        "pool_size": len(pool),
        "source_count": len(rss_sources),
        "source_urls": [str(source.get("url", "")) for source in rss_sources],
        "etl_phase_counts": _phase_counts(bootstrap_attempts),
        "schema_sizes": {
            "actors": len(package_payload["schema"].get("actors", [])),
            "resources": len(package_payload["schema"].get("resources", [])),
            "outcomes": len(package_payload["schema"].get("outcomes", [])),
            "causal_dag": len(package_payload["schema"].get("causal_dag", [])),
        },
        "node_type_counts": _node_type_counts(kg),
        "edge_type_counts": _edge_type_counts(kg),
        "metrics": {
            "etl_seconds": etl_seconds,
            "total_seconds": total_seconds,
            "bootstrap_attempt_count": len(bootstrap_attempts),
            "processed_signals": int(ctx.stats["signals_committed"]),
            "world_updates": int(ctx.stats["world_updates"]),
            "world_update_failures": int(ctx.stats["world_update_failures"]),
            "filtered_out": int(ctx.stats["filtered_out_count"]),
            "watch_skipped": int(ctx.stats["watch_skipped"]),
            "world_update_rate": float(ctx.stats["world_updates"]) / len(selected_signals),
            "kg_nodes": len(kg.nodes()),
            "kg_edges": len(kg.edges()),
            "nodes_per_processed_signal": len(kg.nodes()) / processed_signals,
            "edges_per_processed_signal": len(kg.edges()) / processed_signals,
        },
        "artifacts": {
            "output_dir": str(output_dir),
            "benchmark_config_path": str(benchmark_config_path),
            "bootstrap_package_path": str(paths.runtime_path / "bootstrap_package.json"),
            "kg_json_path": str(output_dir / "kg.json"),
            "kg_3d_path": str(output_dir / "kg_3d.html"),
        },
        "per_signal_results": per_signal_results,
    }
    _write_json(output_dir / "benchmark_summary.json", summary)
    (output_dir / "report.md").write_text(_report_text(summary), encoding="utf-8")
    LOGGER.info("Benchmark complete. Summary written to %s", output_dir / "benchmark_summary.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
