"""Tests for cross-session signal memory."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from freeman.agent.signalingestion import ManualSignalSource, Signal, SignalIngestionEngine, SignalMemory


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
