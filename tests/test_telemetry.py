from datetime import datetime, timezone
from concurrent.futures import ThreadPoolExecutor

from veilrouter.telemetry.pricing import estimate_cost
from veilrouter.telemetry.recorder import InMemoryTelemetryRecorder, TelemetryRecord, now_utc


def test_estimate_cost_uses_per_thousand_token_rates_and_unknown_models_are_free():
    pricing = {"model": {"input": 0.01, "output": 0.03}}

    cost = estimate_cost("model", tokens_in=1500, tokens_out=500, pricing=pricing)

    assert cost == 0.03
    assert estimate_cost("unknown", tokens_in=1500, tokens_out=500, pricing=pricing) == 0.0
    assert estimate_cost("model", tokens_in=None, tokens_out=None, pricing=pricing) == 0.0


def test_in_memory_telemetry_report_aggregates_routes_savings_and_redactions():
    recorder = InMemoryTelemetryRecorder()
    timestamp = datetime(2026, 1, 1, tzinfo=timezone.utc)
    recorder.record(
        TelemetryRecord(
            timestamp=timestamp,
            route="local",
            score=1,
            model="local",
            tokens_in=10,
            tokens_out=5,
            latency_ms=1.5,
            pii_detected=False,
            redaction_count=0,
            cost_saved=0.25,
        )
    )
    recorder.record(
        TelemetryRecord(
            timestamp=timestamp,
            route="cloud",
            score=5,
            model="cloud",
            tokens_in=20,
            tokens_out=10,
            latency_ms=2.5,
            pii_detected=True,
            redaction_count=2,
            redaction_categories={"EMAIL": 2},
            cost_estimate=0.5,
        )
    )

    report = recorder.report()

    assert report == {
        "total_calls": 2,
        "local_calls": 1,
        "local_rate": 0.5,
        "total_cost_saved": 0.25,
        "total_redactions": 2,
    }


def test_in_memory_telemetry_recorder_handles_concurrent_writes():
    recorder = InMemoryTelemetryRecorder()
    timestamp = datetime(2026, 1, 1, tzinfo=timezone.utc)

    def record_one(index):
        recorder.record(
            TelemetryRecord(
                timestamp=timestamp,
                route="local" if index % 2 == 0 else "cloud",
                score=index % 6,
                model="model",
                tokens_in=1,
                tokens_out=1,
                latency_ms=1.0,
                pii_detected=False,
                redaction_count=0,
                cost_saved=0.01,
            )
        )

    with ThreadPoolExecutor(max_workers=8) as executor:
        list(executor.map(record_one, range(100)))

    report = recorder.report()

    assert report["total_calls"] == 100
    assert report["local_calls"] == 50
    assert report["total_cost_saved"] == 1.0


def test_empty_telemetry_report_has_zero_rates():
    assert InMemoryTelemetryRecorder().report() == {
        "total_calls": 0,
        "local_calls": 0,
        "local_rate": 0.0,
        "total_cost_saved": 0,
        "total_redactions": 0,
    }


def test_now_utc_returns_timezone_aware_timestamp():
    timestamp = now_utc()

    assert timestamp.tzinfo == timezone.utc
