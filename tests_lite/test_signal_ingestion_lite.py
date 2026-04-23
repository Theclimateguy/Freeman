from freeman.agent.signalingestion import SignalIngestionEngine


def test_signal_engine_classifies_analyze_and_duplicate() -> None:
    engine = SignalIngestionEngine(keywords=["water", "drought"], min_keyword_hits=1)

    first = engine.classify("Severe drought reduces water inflows across the basin.")
    assert first.mode == "ANALYZE"
    assert first.duplicate is False
    assert "drought" in first.keyword_hits

    duplicate = engine.classify(
        "Severe drought reduces water inflows across the basin.",
        processed_hashes=[first.signal_hash],
    )
    assert duplicate.mode == "WATCH"
    assert duplicate.duplicate is True
    assert duplicate.reason == "duplicate_signal"


def test_signal_engine_uses_watch_below_threshold() -> None:
    engine = SignalIngestionEngine(keywords=["inflation"], min_keyword_hits=1)
    result = engine.classify("Reservoir conditions remain stable this week.")
    assert result.mode == "WATCH"
    assert result.reason == "below_keyword_threshold"
