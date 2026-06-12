import pytest

from benchmarks.benchmark_scorer_models import (
    DEFAULT_DATASET,
    Prediction,
    load_dataset,
    parse_dataset_items,
    rank_summaries,
    route_for_score,
    summarize_predictions,
    validate_dataset_coverage,
)


def test_scoring_eval_dataset_covers_required_cases():
    examples = load_dataset(DEFAULT_DATASET)

    coverage = validate_dataset_coverage(examples)

    assert coverage == {
        "scores_0_to_5": True,
        "at_least_five_per_score": True,
        "pii_and_no_pii": True,
        "long_prompt": True,
        "structured_prompt": True,
        "high_stakes_prompt": True,
        "coding_prompt": True,
        "support_prompt": True,
        "ambiguous_prompt": True,
        "security_prompt": True,
    }
    assert len(examples) >= 36


def test_parse_dataset_items_rejects_invalid_scores():
    data = [{"id": "bad", "prompt": "hello", "expected_score": 6, "contains_pii": False, "tags": []}]

    with pytest.raises(ValueError, match="expected_score must be 0-5"):
        parse_dataset_items(data)


def test_summarize_predictions_reports_accuracy_route_parse_latency_and_speed_metrics():
    predictions = [
        Prediction("easy", expected_score=1, predicted_score=1, latency_ms=10.0),
        Prediction("near", expected_score=3, predicted_score=2, latency_ms=20.0),
        Prediction("wrong-route", expected_score=5, predicted_score=0, latency_ms=30.0, parse_failed=True),
    ]

    summary = summarize_predictions("candidate", predictions, local_score_max=1, setup_latency_ms=1000.0)

    assert summary["exact_accuracy"] == pytest.approx(1 / 3, abs=0.0001)
    assert summary["within_1_accuracy"] == pytest.approx(2 / 3, abs=0.0001)
    assert summary["route_accuracy"] == pytest.approx(2 / 3, abs=0.0001)
    assert summary["parse_failures"] == 1
    assert summary["setup_latency_ms"] == 1000.0
    assert summary["latency_ms"] == {"avg": 20.0, "p50": 20.0, "p95": 30.0, "max": 30.0}
    assert summary["speed_penalty"]["load"] > 0
    assert summary["speed_penalty"]["inference"] > 0


def test_rank_summaries_includes_quality_load_time_and_inference_speed():
    high_quality_slow = {
        "model": "exact-but-slow",
        "rank_score": 0.70,
        "route_accuracy": 0.5,
        "exact_accuracy": 1.0,
        "within_1_accuracy": 1.0,
        "parse_failures": 0,
        "errors": 0,
        "setup_latency_ms": 2000.0,
        "latency_ms": {"p95": 100.0},
    }
    better_route_fast = {
        "model": "better-route-fast",
        "rank_score": 0.80,
        "route_accuracy": 1.0,
        "exact_accuracy": 0.8,
        "within_1_accuracy": 1.0,
        "parse_failures": 0,
        "errors": 0,
        "setup_latency_ms": 1000.0,
        "latency_ms": {"p95": 10.0},
    }

    ranked = rank_summaries([high_quality_slow, better_route_fast])

    assert [summary["model"] for summary in ranked] == ["better-route-fast", "exact-but-slow"]


def test_summarize_predictions_penalizes_slower_load_and_inference_times():
    predictions = [Prediction("same", expected_score=2, predicted_score=2, latency_ms=10.0)]
    fast = summarize_predictions("fast", predictions, setup_latency_ms=10.0)
    slow = summarize_predictions(
        "slow",
        [Prediction("same", expected_score=2, predicted_score=2, latency_ms=60_000.0)],
        setup_latency_ms=180_000.0,
    )

    assert fast["rank_score"] > slow["rank_score"]


@pytest.mark.parametrize(
    ("score", "expected"),
    [
        (0, "local"),
        (1, "local"),
        (2, "cloud"),
        (5, "cloud"),
    ],
)
def test_route_for_score_matches_default_router_threshold(score, expected):
    assert route_for_score(score, local_score_max=1) == expected
