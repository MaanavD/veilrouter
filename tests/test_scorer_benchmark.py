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
        "pii_and_no_pii": True,
        "long_prompt": True,
        "structured_prompt": True,
        "high_stakes_prompt": True,
    }


def test_parse_dataset_items_rejects_invalid_scores():
    data = [{"id": "bad", "prompt": "hello", "expected_score": 6, "contains_pii": False, "tags": []}]

    with pytest.raises(ValueError, match="expected_score must be 0-5"):
        parse_dataset_items(data)


def test_summarize_predictions_reports_accuracy_route_parse_and_latency_metrics():
    predictions = [
        Prediction("easy", expected_score=1, predicted_score=1, latency_ms=10.0),
        Prediction("near", expected_score=3, predicted_score=2, latency_ms=20.0),
        Prediction("wrong-route", expected_score=5, predicted_score=0, latency_ms=30.0, parse_failed=True),
    ]

    summary = summarize_predictions("candidate", predictions, local_score_max=1)

    assert summary["exact_accuracy"] == pytest.approx(1 / 3, abs=0.0001)
    assert summary["within_1_accuracy"] == pytest.approx(2 / 3, abs=0.0001)
    assert summary["route_accuracy"] == pytest.approx(2 / 3, abs=0.0001)
    assert summary["parse_failures"] == 1
    assert summary["latency_ms"] == {"avg": 20.0, "p50": 20.0, "p95": 30.0, "max": 30.0}


def test_rank_summaries_prioritizes_route_accuracy_then_quality_metrics():
    low_route = {
        "model": "exact-but-bad-route",
        "rank_score": 0.80,
        "route_accuracy": 0.5,
        "exact_accuracy": 1.0,
        "within_1_accuracy": 1.0,
        "parse_failures": 0,
        "latency_ms": {"p95": 1.0},
    }
    high_route = {
        "model": "better-route",
        "rank_score": 0.90,
        "route_accuracy": 1.0,
        "exact_accuracy": 0.8,
        "within_1_accuracy": 1.0,
        "parse_failures": 0,
        "latency_ms": {"p95": 10.0},
    }

    ranked = rank_summaries([low_route, high_route])

    assert [summary["model"] for summary in ranked] == ["better-route", "exact-but-bad-route"]


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
