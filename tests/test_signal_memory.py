"""Tests for cross-session signal memory."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from freeman.agent.signalingestion import (
    ManualSignalSource,
    ShockClassification,
    Signal,
    SignalIngestionEngine,
    SignalMemory,
    SignalTrigger,
)
from freeman_connectors.http import HTTPJSONSignalSource
from freeman_connectors.rss import _normalize_entry_timestamp


def test_signal_ingestion_skips_duplicate_within_one_hour() -> None:
    engine = SignalIngestionEngine()
    memory = SignalMemory()
    first = Signal(
        signal_id="signal:1",
        source_type="manual",
        text="Reservoir stress rising",
        topic="water",
        timestamp=datetime(2026, 3, 27, 12, 0, tzinfo=timezone.utc).isoformat(),
    )
    duplicate = Signal(
        signal_id="signal:1",
        source_type="manual",
        text="Reservoir stress rising",
        topic="water",
        timestamp=datetime(2026, 3, 27, 12, 30, tzinfo=timezone.utc).isoformat(),
    )

    first_triggers = engine.ingest(ManualSignalSource([first]), signal_memory=memory)
    duplicate_triggers = engine.ingest(
        ManualSignalSource([duplicate]),
        signal_memory=memory,
        skip_duplicates_within_hours=1.0,
    )

    assert len(first_triggers) == 1
    assert duplicate_triggers == []


def test_signal_memory_effective_weight_decays_with_time() -> None:
    memory = SignalMemory(decay_halflife_hours=24.0)
    seen_at = datetime(2026, 3, 27, 12, 0, tzinfo=timezone.utc)
    signal = Signal(
        signal_id="signal:2",
        source_type="manual",
        text="New drought alert",
        topic="climate",
        timestamp=seen_at.isoformat(),
    )
    memory.see(signal, "ANALYZE")

    weight_now = memory.effective_weight("signal:2", now=seen_at)
    weight_later = memory.effective_weight("signal:2", now=seen_at + timedelta(hours=24))

    assert weight_now == 1.0
    assert weight_later < weight_now
    assert weight_later == 0.5


def test_signal_memory_snapshot_roundtrip() -> None:
    memory = SignalMemory()
    signal = Signal(
        signal_id="signal:3",
        source_type="manual",
        text="Policy shift",
        topic="policy",
        timestamp=datetime(2026, 3, 27, 12, 0, tzinfo=timezone.utc).isoformat(),
    )
    memory.see(signal, "WATCH")
    snapshot = memory.snapshot()

    restored = SignalMemory()
    restored.load_snapshot(snapshot)

    assert restored.is_duplicate(signal, within_hours=1.0) is True
    assert restored.snapshot()[0]["last_trigger_mode"] == "WATCH"


def test_signal_ingestion_uses_generic_interest_budget_without_dropping_signals() -> None:
    engine = SignalIngestionEngine()
    signals = [
        Signal(signal_id="signal:high", source_type="manual", text="Signal high", topic="general"),
        Signal(signal_id="signal:mid", source_type="manual", text="Signal mid", topic="general"),
        Signal(signal_id="signal:low", source_type="manual", text="Signal low", topic="general"),
    ]

    def classifier(signal: Signal):  # noqa: ANN202
        levels = {
            "signal:high": (0.9, 0.9),
            "signal:mid": (0.7, 0.7),
            "signal:low": (0.55, 0.55),
        }
        severity, semantic_gap = levels[signal.signal_id]
        return {
            "shock_type": "generic",
            "severity": severity,
            "semantic_gap": semantic_gap,
            "rationale": signal.signal_id,
        }

    triggers = engine.ingest(
        ManualSignalSource(signals),
        classifier=classifier,
        analysis_budget=1.0,
        analyze_cost=1.0,
        deep_dive_cost=2.0,
    )

    assert len(triggers) == 3
    assert [trigger.signal_id for trigger in triggers] == ["signal:high", "signal:mid", "signal:low"]
    assert sum(1 for trigger in triggers if trigger.mode == "ANALYZE") == 1
    assert sum(1 for trigger in triggers if trigger.mode == "WATCH") == 2
    assert all(trigger.interest_score >= 0.0 for trigger in triggers)
    assert all(trigger.requested_mode in {"ANALYZE", "DEEP_DIVE"} for trigger in triggers)


def test_signal_ingestion_marks_opposing_same_topic_signals_as_conflicts() -> None:
    engine = SignalIngestionEngine()
    signals = [
        Signal(
            signal_id="signal:stress",
            source_type="manual",
            text="Reservoir stress rising sharply.",
            topic="water",
            entities=["reservoir"],
            sentiment=-0.8,
        ),
        Signal(
            signal_id="signal:recovery",
            source_type="manual",
            text="Reservoir recovery improves allocation outlook.",
            topic="water",
            entities=["reservoir"],
            sentiment=0.7,
        ),
    ]

    triggers = engine.ingest(ManualSignalSource(signals))

    assert {trigger.signal_id for trigger in triggers} == {"signal:stress", "signal:recovery"}
    assert all(trigger.conflict_score > 0.0 for trigger in triggers)
    assert triggers[0].conflicts_with == ["signal:recovery"]
    assert triggers[1].conflicts_with == ["signal:stress"]
    assert all(trigger.conflict_reason == "opposing_sentiment_same_topic_or_entity" for trigger in triggers)


def test_signal_conflict_annotation_can_mark_current_trigger_against_queue() -> None:
    engine = SignalIngestionEngine()
    current = Signal(
        signal_id="signal:current",
        source_type="manual",
        text="Reservoir stress rising.",
        topic="water",
        entities=["reservoir"],
        sentiment=-0.8,
    )
    queued = Signal(
        signal_id="signal:queued",
        source_type="manual",
        text="Reservoir recovery improving.",
        topic="water",
        entities=["reservoir"],
        sentiment=0.8,
    )
    trigger = SignalTrigger(
        signal_id="signal:current",
        mahalanobis_score=0.0,
        classification=ShockClassification("routine", 0.2, 0.1),
        mode="WATCH",
    )

    engine._annotate_signal_conflicts([current, queued], [trigger])

    assert trigger.conflict_score > 0.0
    assert trigger.conflicts_with == ["signal:queued"]


def test_http_json_source_uses_fallback_topic_when_topic_path_is_unset() -> None:
    class _StubHTTPJSONSignalSource(HTTPJSONSignalSource):
        def fetch_payload(self):  # noqa: ANN202
            return {
                "articles": [
                    {
                        "headline": "Factory output rises on stronger export orders.",
                        "published_at": "2026-03-30T12:00:00+00:00",
                        "category": "macro",
                    }
                ]
            }

    source = _StubHTTPJSONSignalSource(
        url="https://example.com/feed",
        item_path="articles",
        field_map={"text": "headline", "timestamp": "published_at"},
        default_topic="world",
    )

    signals = source.fetch()

    assert len(signals) == 1
    assert signals[0].topic == "world"
    assert signals[0].entities == []
    assert signals[0].sentiment == 0.0
    assert signals[0].metadata["category"] == "macro"


def test_http_json_source_skips_service_payload_when_item_path_is_empty() -> None:
    class _EmptyHTTPJSONSignalSource(HTTPJSONSignalSource):
        def fetch_payload(self):  # noqa: ANN202
            return {
                "articles": [],
                "information": {"message": "Plan limit"},
            }

    source = _EmptyHTTPJSONSignalSource(
        url="https://example.com/feed",
        item_path="articles",
        field_map={"text": "headline"},
        default_topic="world",
    )

    assert source.fetch() == []


def test_rss_timestamp_normalizer_converts_rfc822_to_iso() -> None:
    assert _normalize_entry_timestamp({"published": "Sun, 29 Mar 2026 18:42:19 GMT"}) == "2026-03-29T18:42:19+00:00"
