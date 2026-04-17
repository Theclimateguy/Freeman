"""Semantic runtime retrieval and answer synthesis over persisted artifacts."""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
import re
from typing import Any, Dict, List, Sequence

import yaml

from freeman.agent.costmodel import (
    BudgetLedger,
    BudgetPolicy,
    CostModel,
    build_budget_policy,
    budget_tracking_enabled,
    resolve_budget_decision,
)
from freeman.agent.analysispipeline import AnalysisPipeline
from freeman.agent.consciousness import ConsciousState
from freeman.agent.forecastregistry import Forecast, ForecastRegistry
from freeman.core.world import WorldState
from freeman.game.runner import SimConfig
from freeman.interface.factory import (
    build_chat_client,
    build_embedding_adapter,
    build_knowledge_graph,
    build_vectorstore,
    resolve_memory_json_path,
    resolve_runtime_path,
    resolve_semantic_min_score,
)
from freeman.memory.knowledgegraph import KGEdge, KGSemanticSearchResult, KGNode
from freeman.runtime.checkpoint import CheckpointManager


DEFAULT_QUERY_CONFIG: dict[str, Any] = {
    "agent": {
        "budget_usd_per_day": 0.50,
        "cost_governance": {},
    },
    "llm": {
        "provider": "",
        "model": "",
        "base_url": "",
        "timeout_seconds": 90.0,
    },
    "memory": {
        "json_path": "./data/kg_state.json",
        "vector_store": {"enabled": False},
        "retrieval_top_k": 15,
    },
    "runtime": {
        "runtime_path": "./data/runtime",
    },
}


def _merge_dicts(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _merge_dicts(merged[key], value)
        else:
            merged[key] = value
    return merged


@dataclass
class RuntimeArtifacts:
    """Resolved persisted runtime state plus semantic backends."""

    config: dict[str, Any]
    config_path: Path
    runtime_path: Path
    kg_path: Path
    knowledge_graph: Any
    pipeline: AnalysisPipeline
    world_state: WorldState | None
    embedding_adapter: Any | None
    embedding_backend: str | None
    vectorstore: Any | None
    semantic_min_score: float
    cost_model: CostModel
    budget_policy: BudgetPolicy
    budget_ledger: BudgetLedger | None
    budget_tracking_enabled: bool


@dataclass
class RuntimeEvidence:
    """One evidence item used for semantic answers."""

    kind: str
    item_id: str
    label: str
    score: float
    payload: dict[str, Any]

    def snapshot(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "id": self.item_id,
            "label": self.label,
            "score": float(self.score),
            "payload": self.payload,
        }


@dataclass
class RuntimeQueryResult:
    """Structured runtime semantic-query output."""

    query: str
    semantic: bool
    matched: bool
    matches: list[dict[str, Any]]
    evidence: list[RuntimeEvidence]
    semantic_trace: list[dict[str, Any]]
    no_match_reason: str | None
    semantic_backend: str | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "query": self.query,
            "semantic": self.semantic,
            "matched": self.matched,
            "count": len(self.matches),
            "matches": self.matches,
            "evidence_count": len(self.evidence),
            "evidence": [item.snapshot() for item in self.evidence],
            "semantic_trace": list(self.semantic_trace),
            "semantic_backend": self.semantic_backend,
            "no_match_reason": self.no_match_reason,
        }


def _resolve_config_path(config_path: str | Path | None = None) -> Path:
    if config_path is None:
        return (Path(__file__).resolve().parents[2] / "config.yaml").resolve()
    candidate = Path(config_path).expanduser()
    return candidate.resolve()


def _load_config(config_path: str | Path | None = None) -> tuple[dict[str, Any], Path]:
    resolved = _resolve_config_path(config_path)
    if not resolved.exists():
        return dict(DEFAULT_QUERY_CONFIG), resolved
    payload = dict(yaml.safe_load(resolved.read_text(encoding="utf-8")) or {})
    return _merge_dicts(DEFAULT_QUERY_CONFIG, payload), resolved


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


def load_runtime_artifacts(config_path: str | Path | None = None) -> RuntimeArtifacts:
    """Load persisted runtime state with the same semantic backends as the CLI."""

    config, resolved_config_path = _load_config(config_path)
    memory_cfg = dict(config.get("memory", {}))
    vectorstore = build_vectorstore(config, config_path=resolved_config_path)
    embedding_adapter = None
    embedding_backend = None
    if vectorstore is not None or str(memory_cfg.get("embedding_provider", "")).strip():
        embedding_adapter, embedding_backend = build_embedding_adapter(config)

    runtime_path = resolve_runtime_path(config, config_path=resolved_config_path)
    kg_path = resolve_memory_json_path(config, config_path=resolved_config_path)
    knowledge_graph = build_knowledge_graph(
        config,
        config_path=resolved_config_path,
        embedding_adapter=embedding_adapter,
        vectorstore=vectorstore,
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
    world_state_path = runtime_path / "world_state.json"
    world_state = None
    if world_state_path.exists():
        world_state = WorldState.from_snapshot(json.loads(world_state_path.read_text(encoding="utf-8")))
    policy = build_budget_policy(config)
    tracking_enabled = budget_tracking_enabled(config)
    budget_ledger = BudgetLedger(runtime_path / "cost_ledger.jsonl", policy=policy, auto_load=True) if tracking_enabled else None
    return RuntimeArtifacts(
        config=config,
        config_path=resolved_config_path,
        runtime_path=runtime_path,
        kg_path=kg_path,
        knowledge_graph=knowledge_graph,
        pipeline=pipeline,
        world_state=world_state,
        embedding_adapter=embedding_adapter,
        embedding_backend=embedding_backend,
        vectorstore=vectorstore,
        semantic_min_score=resolve_semantic_min_score(config),
        cost_model=CostModel(policy),
        budget_policy=policy,
        budget_ledger=budget_ledger,
        budget_tracking_enabled=tracking_enabled,
    )


def _semantic_tokens(text: str) -> List[str]:
    normalized = str(text).lower().replace("_", " ")
    return re.findall(r"[a-z0-9]+", normalized)


def _semantic_ngrams(tokens: Sequence[str], *, size: int) -> set[str]:
    values = [str(token) for token in tokens if str(token)]
    if len(values) < size:
        return set()
    return {" ".join(values[index : index + size]) for index in range(len(values) - size + 1)}


def _lexical_score(query_text: str, document: str) -> float:
    haystack = str(document).strip().lower()
    query_text = str(query_text).strip().lower()
    if not haystack or not query_text:
        return 0.0
    score = 0.0
    if query_text in haystack:
        score += 1.0
    query_tokens = _semantic_tokens(query_text)
    haystack_token_list = _semantic_tokens(haystack)
    haystack_tokens = set(haystack_token_list)
    if query_tokens:
        overlap = sum(1 for token in query_tokens if token in haystack_tokens)
        score += 0.6 * (overlap / len(query_tokens))
    query_bigrams = _semantic_ngrams(query_tokens, size=2)
    haystack_bigrams = _semantic_ngrams(haystack_token_list, size=2)
    if query_bigrams:
        score += 0.3 * (len(query_bigrams & haystack_bigrams) / len(query_bigrams))
    query_trigrams = _semantic_ngrams(query_tokens, size=3)
    haystack_trigrams = _semantic_ngrams(haystack_token_list, size=3)
    if query_trigrams:
        score += 0.2 * (len(query_trigrams & haystack_trigrams) / len(query_trigrams))
    return float(score)


def _cosine_similarity(left: Sequence[float], right: Sequence[float]) -> float | None:
    if not left or not right or len(left) != len(right):
        return None
    left_norm = math.sqrt(sum(float(value) * float(value) for value in left))
    right_norm = math.sqrt(sum(float(value) * float(value) for value in right))
    if left_norm == 0.0 or right_norm == 0.0:
        return None
    dot = sum(float(a) * float(b) for a, b in zip(left, right, strict=False))
    return float(dot / (left_norm * right_norm))


class RuntimeQueryEngine:
    """Semantic retrieval across KG nodes, forecasts, causal traces, and world state."""

    def __init__(self, artifacts: RuntimeArtifacts) -> None:
        self.artifacts = artifacts
        self._embedding_cache: dict[str, list[float]] = {}

    def semantic_query(self, query_text: str, *, limit: int = 10) -> RuntimeQueryResult:
        limit = max(int(limit), 1)
        query = str(query_text).strip()
        query_embedding = self._query_embedding(query)
        kg_result = self.artifacts.knowledge_graph.semantic_search(
            query,
            top_k=limit,
            min_score=self.artifacts.semantic_min_score,
        )
        matches = [self._node_payload(hit.node) for hit in kg_result.hits[:limit]]
        matched_node_ids = {hit["id"] for hit in matches}

        evidence: list[RuntimeEvidence] = []
        for hit in kg_result.hits:
            evidence.append(
                RuntimeEvidence(
                    kind="kg_node",
                    item_id=hit.node.id,
                    label=hit.node.label,
                    score=float(hit.score),
                    payload={
                        "matched_directly": bool(hit.matched_directly),
                        "source": hit.source,
                        "seed_id": hit.seed_id,
                        "node": self._node_payload(hit.node),
                    },
                )
            )
        evidence.extend(self._forecast_evidence(query, query_embedding, matched_node_ids))
        evidence.extend(self._causal_edge_evidence(query, query_embedding, matched_node_ids))
        world_evidence = self._world_state_evidence(query, query_embedding)
        if world_evidence is not None:
            evidence.append(world_evidence)

        deduped: dict[tuple[str, str], RuntimeEvidence] = {}
        for item in evidence:
            key = (item.kind, item.item_id)
            existing = deduped.get(key)
            if existing is None or float(item.score) > float(existing.score):
                deduped[key] = item
        ranked = sorted(
            deduped.values(),
            key=lambda item: (-float(item.score), item.kind, item.item_id),
        )[: max(limit, len(matches))]
        matched = bool(ranked)
        return RuntimeQueryResult(
            query=query,
            semantic=True,
            matched=matched,
            matches=matches,
            evidence=ranked,
            semantic_trace=kg_result.trace(),
            no_match_reason=None if matched else "no_runtime_evidence_matched",
            semantic_backend=self.artifacts.embedding_backend or "lexical",
        )

    def _query_embedding(self, query_text: str) -> list[float] | None:
        if self.artifacts.embedding_adapter is None or not query_text:
            return None
        return [float(value) for value in self.artifacts.embedding_adapter.embed(query_text)]

    def _score_text(
        self,
        query_text: str,
        document: str,
        *,
        query_embedding: list[float] | None,
        boost: float = 0.0,
    ) -> float:
        lexical = _lexical_score(query_text, document)
        embedding_similarity = self._document_similarity(document, query_embedding=query_embedding)
        if embedding_similarity is not None:
            embedding_weight, lexical_weight = self._score_weights()
            score = (embedding_weight * embedding_similarity) + (lexical_weight * lexical)
        else:
            score = lexical
        return float(score + boost)

    def _score_weights(self) -> tuple[float, float]:
        if self._uses_hashing_backend():
            return 0.45, 0.55
        return 0.75, 0.25

    def _passes_text_acceptance(
        self,
        *,
        lexical: float,
        embedding_similarity: float | None,
        score: float,
        boost: float = 0.0,
    ) -> bool:
        if score < self.artifacts.semantic_min_score:
            return False
        if lexical > 0.0 or boost > 0.0:
            return True
        if embedding_similarity is None:
            return False
        floor = max(self.artifacts.semantic_min_score, 0.10)
        if self._uses_hashing_backend():
            floor = max(floor, 0.18)
        return float(embedding_similarity) >= floor

    def _uses_hashing_backend(self) -> bool:
        backend = str(self.artifacts.embedding_backend or "").lower()
        if "hashing" in backend:
            return True
        adapter = self.artifacts.embedding_adapter
        return adapter is not None and "hashing" in adapter.__class__.__name__.lower()

    def _document_similarity(self, document: str, *, query_embedding: list[float] | None) -> float | None:
        if self.artifacts.embedding_adapter is None or not query_embedding or not document:
            return None
        cached = self._embedding_cache.get(document)
        if cached is None:
            cached = [float(value) for value in self.artifacts.embedding_adapter.embed(document)]
            self._embedding_cache[document] = cached
        return _cosine_similarity(query_embedding, cached)

    def _node_payload(self, node: KGNode) -> dict[str, Any]:
        payload = node.snapshot()
        payload["embedding"] = []
        return payload

    def _forecast_evidence(
        self,
        query_text: str,
        query_embedding: list[float] | None,
        matched_node_ids: set[str],
    ) -> list[RuntimeEvidence]:
        registry = self.artifacts.pipeline.forecast_registry
        if registry is None:
            return []
        evidence: list[RuntimeEvidence] = []
        for forecast in registry.all():
            analysis_node_id = str(forecast.metadata.get("analysis_node_id", "")).strip()
            boost = 0.25 if analysis_node_id and analysis_node_id in matched_node_ids else 0.0
            document = json.dumps(
                {
                    "forecast_id": forecast.forecast_id,
                    "outcome_id": forecast.outcome_id,
                    "status": forecast.status,
                    "predicted_prob": forecast.predicted_prob,
                    "actual_prob": forecast.actual_prob,
                    "error": forecast.error,
                    "analysis_node_id": analysis_node_id,
                    "causal_path": list(forecast.causal_path),
                    "metadata": forecast.metadata,
                },
                ensure_ascii=False,
                sort_keys=True,
            )
            lexical = _lexical_score(query_text, document)
            embedding_similarity = self._document_similarity(document, query_embedding=query_embedding)
            score = self._score_text(query_text, document, query_embedding=query_embedding, boost=boost)
            if not self._passes_text_acceptance(
                lexical=lexical,
                embedding_similarity=embedding_similarity,
                score=score,
                boost=boost,
            ):
                continue
            payload = forecast.snapshot()
            payload["analysis_node_id"] = analysis_node_id or None
            evidence.append(
                RuntimeEvidence(
                    kind="forecast",
                    item_id=str(forecast.forecast_id),
                    label=f"Forecast {forecast.forecast_id}",
                    score=score,
                    payload=payload,
                )
            )
        return evidence

    def _causal_edge_evidence(
        self,
        query_text: str,
        query_embedding: list[float] | None,
        matched_node_ids: set[str],
    ) -> list[RuntimeEvidence]:
        evidence: list[RuntimeEvidence] = []
        for edge in self.artifacts.knowledge_graph.edges():
            if edge.relation_type not in {"causes", "propagates_to", "threshold_exceeded"}:
                continue
            source_node = self.artifacts.knowledge_graph.get_node(edge.source, lazy_embed=False)
            target_node = self.artifacts.knowledge_graph.get_node(edge.target, lazy_embed=False)
            source_label = source_node.label if source_node is not None else edge.source
            target_label = target_node.label if target_node is not None else edge.target
            boost = 0.0
            if edge.source in matched_node_ids or edge.target in matched_node_ids:
                boost += 0.2
            document = json.dumps(
                {
                    "edge_id": edge.id,
                    "relation_type": edge.relation_type,
                    "source_id": edge.source,
                    "source_label": source_label,
                    "target_id": edge.target,
                    "target_label": target_label,
                    "metadata": edge.metadata,
                },
                ensure_ascii=False,
                sort_keys=True,
            )
            lexical = _lexical_score(query_text, document)
            embedding_similarity = self._document_similarity(document, query_embedding=query_embedding)
            score = self._score_text(query_text, document, query_embedding=query_embedding, boost=boost)
            if not self._passes_text_acceptance(
                lexical=lexical,
                embedding_similarity=embedding_similarity,
                score=score,
                boost=boost,
            ):
                continue
            payload = edge.snapshot()
            payload["source_label"] = source_label
            payload["target_label"] = target_label
            evidence.append(
                RuntimeEvidence(
                    kind="causal_edge",
                    item_id=str(edge.id),
                    label=f"{source_label} {edge.relation_type} {target_label}",
                    score=score,
                    payload=payload,
                )
            )
        return evidence

    def _world_state_evidence(
        self,
        query_text: str,
        query_embedding: list[float] | None,
    ) -> RuntimeEvidence | None:
        world = self.artifacts.world_state
        if world is None:
            return None
        payload = {
            "domain_id": world.domain_id,
            "world_t": int(world.t),
            "runtime_step": int(world.runtime_step),
            "actors": sorted(world.actors),
            "resources": sorted(world.resources),
            "outcomes": sorted(world.outcomes),
            "metadata": dict(world.metadata),
        }
        document = json.dumps(payload, ensure_ascii=False, sort_keys=True)
        lexical = _lexical_score(query_text, document)
        embedding_similarity = self._document_similarity(document, query_embedding=query_embedding)
        score = self._score_text(query_text, document, query_embedding=query_embedding)
        if not self._passes_text_acceptance(
            lexical=lexical,
            embedding_similarity=embedding_similarity,
            score=score,
        ):
            return None
        return RuntimeEvidence(
            kind="world_state",
            item_id=f"{world.domain_id}:{world.runtime_step}",
            label=f"World {world.domain_id} @ runtime_step={world.runtime_step}",
            score=score,
            payload=payload,
        )


class RuntimeAnswerEngine:
    """Generate answer text strictly from retrieved runtime evidence."""

    def __init__(self, artifacts: RuntimeArtifacts) -> None:
        self.artifacts = artifacts
        self.query_engine = RuntimeQueryEngine(artifacts)

    def answer(self, query_text: str, *, limit: int = 10, chat_client: Any | None = None) -> dict[str, Any]:
        result = self.query_engine.semantic_query(query_text, limit=limit)
        payload = result.to_dict()
        payload["budget"] = self._budget_summary()
        if not result.matched:
            payload.update(
                {
                    "answer": None,
                    "answer_generated": False,
                    "llm_error": "no runtime evidence matched the query",
                }
            )
            return payload
        if chat_client is None:
            chat_client, llm_error = build_chat_client(self.artifacts.config)
            if chat_client is None:
                payload.update(
                    {
                        "answer": None,
                        "answer_generated": False,
                        "llm_error": llm_error or "llm provider is not configured",
                    }
                )
                return payload

        budget_decision, approved_cost = self._answer_budget_decision(query_text)
        if budget_decision is not None and (not budget_decision.allowed or budget_decision.approved_mode == "WATCH"):
            if self.artifacts.budget_ledger is not None:
                self.artifacts.budget_ledger.record(
                    task_type="answer_generation",
                    requested_mode="ANALYZE",
                    decision=budget_decision,
                    actual_cost=0.0,
                    metadata={
                        "query": query_text,
                        "approved_estimated_cost": float(approved_cost),
                        "match_count": len(result.matches),
                        "evidence_count": len(result.evidence),
                    },
                )
                payload["budget"] = self._budget_summary()
            payload.update(
                {
                    "answer": None,
                    "answer_generated": False,
                    "llm_error": f"budget gate blocked answer generation: {budget_decision.stop_reason or 'budget_policy'}",
                }
            )
            return payload

        context_items = [self._answer_context_item(item) for item in result.evidence[: min(max(limit, 1), 8)]]
        messages = [
            {
                "role": "system",
                "content": (
                    "You are Freeman's runtime answer engine. "
                    "Answer the user's question directly from the provided evidence summaries only. "
                    "Do not describe the JSON structure, schema, or metadata mechanically. "
                    "Start with the direct answer in 1-2 sentences, then give up to 3 short evidence-backed points. "
                    "When useful, name concrete mechanisms such as outcomes, variables, thresholds, signal excerpts, or rationale. "
                    "If the evidence is insufficient, say that explicitly."
                ),
            },
            {
                "role": "user",
                "content": json.dumps(
                    {
                        "query": query_text,
                        "runtime_evidence": context_items,
                        "world_state": self._world_summary(),
                    },
                    ensure_ascii=False,
                ),
            },
        ]
        attempted_generation = False
        try:
            attempted_generation = True
            answer = str(chat_client.chat_text(messages, temperature=0.1, max_tokens=700)).strip()
            payload.update(
                {
                    "answer": answer,
                    "answer_generated": True,
                    "llm_error": None,
                }
            )
        except Exception as exc:  # pragma: no cover - exercised only with live providers
            payload.update(
                {
                    "answer": None,
                    "answer_generated": False,
                    "llm_error": str(exc),
                }
            )
        if budget_decision is not None and self.artifacts.budget_ledger is not None:
            self.artifacts.budget_ledger.record(
                task_type="answer_generation",
                requested_mode="ANALYZE",
                decision=budget_decision,
                actual_cost=float(approved_cost if attempted_generation else 0.0),
                metadata={
                    "query": query_text,
                    "approved_estimated_cost": float(approved_cost),
                    "match_count": len(result.matches),
                    "evidence_count": len(result.evidence),
                    "answer_generated": bool(payload.get("answer_generated")),
                },
            )
            payload["budget"] = self._budget_summary()
        return payload

    def _answer_budget_decision(self, query_text: str) -> tuple[Any | None, float]:
        if not self.artifacts.budget_tracking_enabled or self.artifacts.budget_ledger is None:
            return None, 0.0
        world = self.artifacts.world_state
        actors = len(world.actors) if world is not None else 0
        resources = len(world.resources) if world is not None else 0
        domains = max(len(world.outcomes), 1) if world is not None else 1
        query_tokens = max(len(str(query_text).split()), 1)

        def _estimate_for_mode(mode: str):
            idle_mode = str(mode or "WATCH").upper() == "WATCH"
            return self.artifacts.cost_model.estimate(
                task_id=f"answer:{query_tokens}:{mode.lower()}",
                llm_calls=0 if idle_mode else 1,
                sim_steps=0,
                actors=0 if idle_mode else actors,
                resources=0 if idle_mode else resources,
                domains=0 if idle_mode else domains,
                kg_updates=0,
                embedding_tokens_used=0 if idle_mode else query_tokens * 16,
            )

        decision = resolve_budget_decision(
            cost_model=self.artifacts.cost_model,
            requested_mode="ANALYZE",
            estimate_for_mode=_estimate_for_mode,
            budget_spent=self.artifacts.budget_ledger.spent_usd,
            deep_dive_depth=0,
        )
        approved_cost = 0.0
        if decision.allowed and decision.approved_mode != "WATCH":
            approved_cost = float(_estimate_for_mode(decision.approved_mode).estimated_cost)
        return decision, approved_cost

    def _budget_summary(self) -> dict[str, Any]:
        if self.artifacts.budget_ledger is not None:
            return self.artifacts.budget_ledger.summary()
        return {
            "tracking_enabled": bool(self.artifacts.budget_tracking_enabled),
            "ledger_path": str((self.artifacts.runtime_path / "cost_ledger.jsonl").resolve()),
            "configured_usd_per_day": float(self.artifacts.budget_policy.max_compute_budget_per_session),
            "spent_usd": 0.0,
            "remaining_usd": float(self.artifacts.budget_policy.max_compute_budget_per_session),
            "entry_count": 0,
            "allowed_count": 0,
            "blocked_count": 0,
            "by_task_type": {},
            "stop_reasons": {},
        }

    def _world_summary(self) -> dict[str, Any] | None:
        world = self.artifacts.world_state
        if world is None:
            return None
        return {
            "domain_id": world.domain_id,
            "world_t": int(world.t),
            "runtime_step": int(world.runtime_step),
            "actors": sorted(world.actors),
            "resources": sorted(world.resources),
            "outcomes": sorted(world.outcomes),
        }

    def _answer_context_item(self, item: RuntimeEvidence) -> dict[str, Any]:
        payload = item.payload
        if item.kind == "forecast":
            metadata = dict(payload.get("metadata", {}))
            parameter_vector = dict(metadata.get("parameter_vector", {}))
            thresholds = [
                str(step)
                for step in payload.get("causal_path", [])
                if str(step).startswith("threshold_exceeded:")
            ][:6]
            return {
                "kind": item.kind,
                "id": item.item_id,
                "label": item.label,
                "score": float(item.score),
                "outcome_id": payload.get("outcome_id"),
                "predicted_prob": payload.get("predicted_prob"),
                "status": payload.get("status"),
                "analysis_node_id": payload.get("analysis_node_id"),
                "rationale": parameter_vector.get("rationale") or metadata.get("rationale_at_time"),
                "thresholds": thresholds,
            }
        if item.kind == "kg_node":
            node = dict(payload.get("node", {}))
            metadata = dict(node.get("metadata", {}))
            summary = {
                "kind": item.kind,
                "id": item.item_id,
                "label": item.label,
                "score": float(item.score),
                "node_type": node.get("node_type"),
                "content": node.get("content"),
            }
            for key in (
                "rationale",
                "signal_excerpt",
                "dominant_outcome",
                "posterior_dominant_outcome",
                "final_outcome_probs",
                "posterior_outcome_probs",
                "strongest_mismatch",
            ):
                if key in metadata:
                    summary[key] = metadata[key]
            return summary
        if item.kind == "causal_edge":
            return {
                "kind": item.kind,
                "id": item.item_id,
                "label": item.label,
                "score": float(item.score),
                "relation_type": payload.get("relation_type"),
                "source_label": payload.get("source_label"),
                "target_label": payload.get("target_label"),
                "confidence": payload.get("confidence"),
                "metadata": payload.get("metadata", {}),
            }
        if item.kind == "world_state":
            return {
                "kind": item.kind,
                "id": item.item_id,
                "label": item.label,
                "score": float(item.score),
                "payload": payload,
            }
        return {
            "kind": item.kind,
            "id": item.item_id,
            "label": item.label,
            "score": float(item.score),
            "payload": payload,
        }


__all__ = [
    "RuntimeAnswerEngine",
    "RuntimeArtifacts",
    "RuntimeEvidence",
    "RuntimeQueryEngine",
    "RuntimeQueryResult",
    "load_runtime_artifacts",
]
