"""Tests for KG reconciliation and persistence."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
import json

import pytest

from freeman.agent.forecastregistry import Forecast, ForecastRegistry
from freeman.memory.knowledgegraph import KGNode, KnowledgeGraph
from freeman.memory.reconciler import Reconciler
from freeman.memory.sessionlog import KGDelta, SessionLog


@dataclass
class SpyVectorStore:
    deleted_ids: list[str] = field(default_factory=list)
    upserted: list[tuple[str, list[float]]] = field(default_factory=list)

    def upsert(self, node: KGNode) -> None:
        self.upserted.append((node.id, list(node.embedding)))

    def delete(self, node_id: str) -> None:
        self.deleted_ids.append(node_id)

    def query(self, query_embedding: list[float], top_k: int = 15, min_confidence: float = 0.0) -> list[str]:
        del query_embedding, top_k, min_confidence
        return []

    def sync_from_kg(self, kg: KnowledgeGraph) -> int:
        del kg
        return 0


class MappingEmbeddingAdapter:
    def __init__(self, mapping: dict[str, list[float]]) -> None:
        self.mapping = {key: list(value) for key, value in mapping.items()}
        self.calls: list[str] = []

    def embed(self, text: str) -> list[float]:
        self.calls.append(text)
        return list(self.mapping[text])


def test_log_odds_convergence_under_support() -> None:
    reconciler = Reconciler(mode="log_odds")
    confidence = 0.5
    history = []

    for _ in range(8):
        confidence = reconciler.beta_update(confidence, support=1, contradiction=0)
        history.append(confidence)

    assert history == sorted(history)
    assert history[-1] > 0.99


def test_log_odds_convergence_under_conflict() -> None:
    reconciler = Reconciler(mode="log_odds")
    confidence = 0.5
    history = []

    for _ in range(8):
        confidence = reconciler.beta_update(confidence, support=0, contradiction=1)
        history.append(confidence)

    assert history == sorted(history, reverse=True)
    assert history[-1] < 0.01


def test_log_odds_symmetry() -> None:
    reconciler = Reconciler(mode="log_odds")
    confidence = 0.5

    for _ in range(5):
        confidence = reconciler.beta_update(confidence, support=1, contradiction=0)
    for _ in range(5):
        confidence = reconciler.beta_update(confidence, support=0, contradiction=1)

    assert confidence == pytest.approx(0.5, abs=1.0e-9)


def test_legacy_mode_unchanged() -> None:
    reconciler = Reconciler(mode="legacy", prior_strength_penalty=0.05)
    updated = reconciler.beta_update(0.8, support=2, contradiction=3)

    assert updated == pytest.approx(0.8 * (2.0 / 5.0) - 0.05)


def test_forgetting_decay() -> None:
    reconciler = Reconciler(mode="log_odds", gamma=0.5)
    confidence = 0.95
    distances = []

    for _ in range(6):
        confidence = reconciler.beta_update(confidence, support=0, contradiction=0)
        distances.append(abs(confidence - 0.5))

    assert distances == sorted(distances, reverse=True)
    assert distances[-1] < distances[0]


def test_log_odds_clips_boundary_confidence() -> None:
    reconciler = Reconciler(mode="log_odds")

    assert 0.0 < reconciler.beta_update(0.0, support=1, contradiction=0) <= 1.0
    assert 0.0 <= reconciler.beta_update(1.0, support=0, contradiction=1) < 1.0


def test_log_odds_no_evidence_without_forgetting_keeps_confidence() -> None:
    reconciler = Reconciler(mode="log_odds", gamma=0.0)

    assert reconciler.beta_update(0.73, support=0, contradiction=0) == pytest.approx(0.73)


def test_reconciler_archives_low_confidence_nodes(tmp_path) -> None:
    graph_path = tmp_path / "kg.json"
    kg = KnowledgeGraph(json_path=graph_path, auto_load=False, auto_save=True)
    existing = KGNode(
        id="claim_water",
        label="Water Stress",
        content="Water stress is rising.",
        confidence=0.9,
        metadata={"claim_key": "water_stress"},
    )
    kg.add_node(existing)

    session = SessionLog(session_id="s1")
    session.add_kg_delta(
        KGDelta(
            operation="update_node",
            payload={"node": existing.snapshot()},
            support=1,
            contradiction=9,
        )
    )

    result = Reconciler().reconcile(kg, session)
    archived = kg.get_node("claim_water")

    assert archived is not None
    assert archived.status == "archived"
    assert archived.confidence < 0.15
    assert "claim_water" in result.archived_node_ids
    assert graph_path.exists()


def test_reconciler_splits_conflicting_claims_and_exports_graph(tmp_path) -> None:
    kg = KnowledgeGraph(json_path=tmp_path / "kg.json", auto_load=False, auto_save=True)
    base = KGNode(
        id="claim_inflation",
        label="Inflation Outlook",
        content="Inflation will rise.",
        confidence=0.8,
        metadata={"claim_key": "inflation"},
    )
    kg.add_node(base)

    conflicting = KGNode(
        id="claim_inflation_down",
        label="Inflation Outlook",
        content="Inflation will fall.",
        confidence=0.8,
        metadata={"claim_key": "inflation"},
    )
    session = SessionLog(session_id="s2")
    session.add_kg_delta(KGDelta(operation="add_node", payload={"node": conflicting.snapshot()}))

    result = Reconciler().reconcile(kg, session)
    archived = kg.get_node("claim_inflation")
    matches = kg.query(text="inflation")
    dot = kg.export_dot()
    html_path = kg.export_html(tmp_path / "kg.html")
    json_path = kg.export_json(tmp_path / "kg-export.json")

    assert archived is not None
    assert archived.status == "archived"
    assert result.split_nodes["claim_inflation"] == ["claim_inflation__split_1", "claim_inflation_down"]
    assert {"claim_inflation__split_1", "claim_inflation_down"} <= {node.id for node in matches}
    assert "split_into" in dot
    assert html_path.exists()
    assert json_path.exists()


def test_reconciler_archive_triggers_vectorstore_delete(tmp_path) -> None:
    store = SpyVectorStore()
    kg = KnowledgeGraph(json_path=tmp_path / "kg.json", auto_load=False, auto_save=True, vectorstore=store)
    node = KGNode(
        id="claim_archive",
        label="Archive me",
        content="Claim to archive.",
        confidence=0.9,
    )
    kg.add_node(node)

    session = SessionLog(session_id="archive-session")
    session.add_kg_delta(
        KGDelta(
            operation="archive_node",
            target_id=node.id,
            payload={"id": node.id},
        )
    )

    Reconciler().reconcile(kg, session)

    assert store.deleted_ids == [node.id]


def test_reconciler_merge_forces_reembed_when_vectorstore_present(tmp_path) -> None:
    adapter = MappingEmbeddingAdapter(
        {
            "Water stress is rising.": [1.0, 0.0, 0.0],
        }
    )
    store = SpyVectorStore()
    kg = KnowledgeGraph(
        json_path=tmp_path / "kg.json",
        auto_load=False,
        auto_save=True,
        llm_adapter=adapter,
        vectorstore=store,
    )
    existing = KGNode(
        id="claim_water",
        label="Water Stress",
        content="Water stress is rising.",
        confidence=0.7,
        metadata={"claim_key": "water_stress"},
    )
    kg.add_node(existing)

    incoming = KGNode(
        id="claim_water_update",
        label="Water Stress",
        content="Water stress is rising.",
        confidence=0.8,
        metadata={"claim_key": "water_stress", "source": "new_feed"},
    )
    session = SessionLog(session_id="merge-session")
    session.add_kg_delta(KGDelta(operation="add_node", payload={"node": incoming.snapshot()}))

    Reconciler().reconcile(kg, session)

    assert adapter.calls == ["Water stress is rising.", "Water stress is rising."]
    assert len(store.upserted) >= 2
    assert store.upserted[-1][0] == "claim_water"


def test_reconciler_update_self_model_accumulates_mae_and_bias(tmp_path) -> None:
    kg = KnowledgeGraph(json_path=tmp_path / "kg.json", auto_load=False, auto_save=True)
    registry = ForecastRegistry(auto_load=False, auto_save=False)
    reconciler = Reconciler()
    verified_forecasts = []
    specs = [
        ("f1", 0.70, 0.60),
        ("f2", 0.40, 0.55),
        ("f3", 0.90, 0.70),
    ]
    for forecast_id, predicted, actual in specs:
        forecast = Forecast(
            forecast_id=forecast_id,
            domain_id="water",
            outcome_id="cooperation",
            predicted_prob=predicted,
            session_id="s1",
            horizon_steps=3,
            created_at=datetime(2026, 3, 27, tzinfo=timezone.utc),
            created_step=0,
        )
        registry.record(forecast)
        verified_forecasts.append(
            registry.verify(
                forecast_id,
                actual_prob=actual,
                verified_at=datetime(2026, 3, 30, tzinfo=timezone.utc),
            )
        )

    node = None
    for forecast in verified_forecasts:
        node = reconciler.update_self_model(kg, forecast)

    assert node is not None
    assert node.node_type == "self_observation"
    assert node.metadata["n_forecasts"] == 3
    assert node.metadata["mean_abs_error"] == pytest.approx((0.10 + 0.15 + 0.20) / 3.0)
    assert node.metadata["bias"] == pytest.approx((0.10 - 0.15 + 0.20) / 3.0)


def test_reconciler_update_self_model_trims_error_window_to_50(tmp_path) -> None:
    kg = KnowledgeGraph(json_path=tmp_path / "kg.json", auto_load=False, auto_save=True)
    registry = ForecastRegistry(auto_load=False, auto_save=False)
    reconciler = Reconciler()

    for idx in range(55):
        forecast = Forecast(
            forecast_id=f"f{idx}",
            domain_id="water",
            outcome_id="crisis",
            predicted_prob=0.5,
            session_id="s1",
            horizon_steps=3,
            created_at=datetime(2026, 3, 27, tzinfo=timezone.utc),
            created_step=0,
        )
        registry.record(forecast)
        verified = registry.verify(
            forecast.forecast_id,
            actual_prob=max(0.0, 0.5 - 0.01 * (idx % 5)),
            verified_at=datetime(2026, 3, 30, tzinfo=timezone.utc),
        )
        reconciler.update_self_model(kg, verified)

    node = kg.get_node("self:forecast_error:water:crisis")

    assert node is not None
    assert node.metadata["n_forecasts"] == 50
    assert len(json.loads(node.metadata["errors_json"])) == 50
