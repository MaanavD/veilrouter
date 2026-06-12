from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from veilroute.errors import ScoreParseError  # noqa: E402
from veilroute.providers.foundry_local import FoundryLocalProvider  # noqa: E402
from veilroute.scoring.llm_scorer import _RUBRIC  # noqa: E402
from veilroute.scoring.parsing import parse_score  # noqa: E402

DEFAULT_DATASET = Path(__file__).with_name("scoring_eval_dataset.json")
MAX_INFERENCE_SPEED_PENALTY = 0.10
MAX_LOAD_SPEED_PENALTY = 0.10
INFERENCE_P95_PENALTY_MS = 60_000.0
LOAD_PENALTY_MS = 180_000.0


@dataclass(frozen=True, slots=True)
class ScorerCandidate:
    score: Callable[[str], tuple[int, bool, str | None]]
    setup_latency_ms: float
    setup_error: str | None = None


@dataclass(frozen=True, slots=True)
class ScoringExample:
    id: str
    prompt: str
    expected_score: int
    contains_pii: bool
    tags: tuple[str, ...]

    @property
    def expected_route(self) -> str:
        return route_for_score(self.expected_score)


@dataclass(frozen=True, slots=True)
class Prediction:
    example_id: str
    expected_score: int
    predicted_score: int
    latency_ms: float
    parse_failed: bool = False
    error: str | None = None


def load_dataset(path: Path = DEFAULT_DATASET) -> list[ScoringExample]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return parse_dataset_items(data)


def parse_dataset_items(data: Any) -> list[ScoringExample]:
    if not isinstance(data, list) or not data:
        raise ValueError("dataset must be a non-empty JSON array")

    examples: list[ScoringExample] = []
    seen_ids: set[str] = set()
    for index, item in enumerate(data):
        if not isinstance(item, dict):
            raise ValueError(f"dataset item {index} must be an object")
        example_id = str(item.get("id", "")).strip()
        if not example_id:
            raise ValueError(f"dataset item {index} is missing id")
        if example_id in seen_ids:
            raise ValueError(f"duplicate dataset id: {example_id}")
        seen_ids.add(example_id)
        prompt = str(item.get("prompt", "")).strip()
        if not prompt:
            raise ValueError(f"{example_id} prompt must be non-empty")
        expected_score = int(item.get("expected_score"))
        if expected_score < 0 or expected_score > 5:
            raise ValueError(f"{example_id} expected_score must be 0-5")
        examples.append(
            ScoringExample(
                id=example_id,
                prompt=prompt,
                expected_score=expected_score,
                contains_pii=bool(item.get("contains_pii", False)),
                tags=tuple(str(tag) for tag in item.get("tags", [])),
            )
        )
    return examples


def validate_dataset_coverage(examples: Iterable[ScoringExample]) -> dict[str, bool]:
    examples = list(examples)
    scores = {example.expected_score for example in examples}
    tags = {tag for example in examples for tag in example.tags}
    counts_by_score = {score: 0 for score in range(6)}
    for example in examples:
        counts_by_score[example.expected_score] += 1
    return {
        "scores_0_to_5": scores == set(range(6)),
        "at_least_five_per_score": all(count >= 5 for count in counts_by_score.values()),
        "pii_and_no_pii": any(example.contains_pii for example in examples)
        and any(not example.contains_pii for example in examples),
        "long_prompt": any("long" in example.tags or len(example.prompt) > 350 for example in examples),
        "structured_prompt": any("structured" in example.tags or "multi-step" in example.tags for example in examples),
        "high_stakes_prompt": "high-stakes" in tags,
        "coding_prompt": "coding" in tags,
        "support_prompt": "support" in tags,
        "ambiguous_prompt": "ambiguous" in tags,
        "security_prompt": "security" in tags,
    }


def route_for_score(score: int, *, local_score_max: int = 1) -> str:
    return "local" if int(score) <= local_score_max else "cloud"


def percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, math.ceil((pct / 100) * len(ordered)) - 1))
    return ordered[index]


def summarize_predictions(
    model: str,
    predictions: list[Prediction],
    *,
    local_score_max: int = 1,
    setup_latency_ms: float = 0.0,
    setup_error: str | None = None,
) -> dict[str, Any]:
    total = len(predictions)
    exact = 0
    within_1 = 0
    route_matches = 0
    parse_failures = 0
    errors = 0
    latencies: list[float] = []

    for prediction in predictions:
        if prediction.error is not None:
            errors += 1
        if prediction.parse_failed:
            parse_failures += 1
        expected_route = route_for_score(prediction.expected_score, local_score_max=local_score_max)
        predicted_route = route_for_score(prediction.predicted_score, local_score_max=local_score_max)
        difference = abs(prediction.predicted_score - prediction.expected_score)
        exact += int(difference == 0)
        within_1 += int(difference <= 1)
        route_matches += int(expected_route == predicted_route)
        latencies.append(prediction.latency_ms)

    exact_accuracy = exact / total if total else 0.0
    within_1_accuracy = within_1 / total if total else 0.0
    route_accuracy = route_matches / total if total else 0.0
    parse_failure_rate = parse_failures / total if total else 0.0
    error_rate = errors / total if total else 0.0
    latency_avg = sum(latencies) / total if total else 0.0
    latency_p95 = percentile(latencies, 95)
    inference_penalty = min(latency_p95 / INFERENCE_P95_PENALTY_MS, MAX_INFERENCE_SPEED_PENALTY)
    load_penalty = min(setup_latency_ms / LOAD_PENALTY_MS, MAX_LOAD_SPEED_PENALTY)
    rank_score = (
        route_accuracy * 0.40
        + exact_accuracy * 0.25
        + within_1_accuracy * 0.20
        - parse_failure_rate * 0.05
        - error_rate * 0.10
        - inference_penalty
        - load_penalty
    )

    return {
        "model": model,
        "examples": total,
        "exact_accuracy": round(exact_accuracy, 4),
        "within_1_accuracy": round(within_1_accuracy, 4),
        "route_accuracy": round(route_accuracy, 4),
        "parse_failures": parse_failures,
        "parse_failure_rate": round(parse_failure_rate, 4),
        "errors": errors,
        "error_rate": round(error_rate, 4),
        "setup_latency_ms": round(setup_latency_ms, 3),
        "setup_error": setup_error,
        "latency_ms": {
            "avg": round(latency_avg, 3),
            "p50": round(percentile(latencies, 50), 3),
            "p95": round(latency_p95, 3),
            "max": round(max(latencies), 3) if latencies else 0.0,
        },
        "speed_penalty": {
            "load": round(load_penalty, 6),
            "inference": round(inference_penalty, 6),
        },
        "rank_score": round(rank_score, 6),
    }


def rank_summaries(summaries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(
        summaries,
        key=lambda item: (
            item["rank_score"],
            item["route_accuracy"],
            item["exact_accuracy"],
            item["within_1_accuracy"],
            -item["parse_failures"],
            -item["errors"],
            -item["setup_latency_ms"],
            -item["latency_ms"]["p95"],
        ),
        reverse=True,
    )


def make_foundry_scorer(model: str, *, temperature: float, max_tokens: int) -> ScorerCandidate:
    started = time.perf_counter()
    provider = FoundryLocalProvider(model=model)

    try:
        provider._ensure_client()
        setup_error = None
    except Exception as exc:
        setup_latency_ms = (time.perf_counter() - started) * 1000
        error = str(exc)

        def raise_setup_error(prompt: str) -> tuple[int, bool, str | None]:
            raise RuntimeError(error)

        return ScorerCandidate(raise_setup_error, setup_latency_ms, error)

    setup_latency_ms = (time.perf_counter() - started) * 1000

    def score(prompt: str) -> tuple[int, bool, str | None]:
        response = provider.complete(
            [
                {"role": "system", "content": _RUBRIC},
                {"role": "user", "content": prompt},
            ],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        try:
            return parse_score(response.text, default=None), False, None
        except ScoreParseError:
            return parse_score(response.text, default=2), True, response.text

    return ScorerCandidate(score, setup_latency_ms, setup_error)


def run_model(
    model: str,
    examples: list[ScoringExample],
    *,
    local_score_max: int,
    temperature: float,
    max_tokens: int,
) -> dict[str, Any]:
    scorer = make_foundry_scorer(model, temperature=temperature, max_tokens=max_tokens)
    predictions: list[Prediction] = []
    for example in examples:
        started = time.perf_counter()
        try:
            predicted_score, parse_failed, raw_error = scorer.score(example.prompt)
            error = None
        except Exception as exc:
            predicted_score = 2
            parse_failed = False
            raw_error = None
            error = str(exc)
        predictions.append(
            Prediction(
                example_id=example.id,
                expected_score=example.expected_score,
                predicted_score=predicted_score,
                latency_ms=(time.perf_counter() - started) * 1000,
                parse_failed=parse_failed,
                error=error or raw_error,
            )
        )

    summary = summarize_predictions(
        model,
        predictions,
        local_score_max=local_score_max,
        setup_latency_ms=scorer.setup_latency_ms,
        setup_error=scorer.setup_error,
    )
    summary["predictions"] = [
        {
            "id": prediction.example_id,
            "expected_score": prediction.expected_score,
            "predicted_score": prediction.predicted_score,
            "expected_route": route_for_score(prediction.expected_score, local_score_max=local_score_max),
            "predicted_route": route_for_score(prediction.predicted_score, local_score_max=local_score_max),
            "latency_ms": round(prediction.latency_ms, 3),
            "parse_failed": prediction.parse_failed,
            "error": prediction.error,
        }
        for prediction in predictions
    ]
    return summary


def print_text(results: dict[str, Any]) -> None:
    print("veilroute Foundry Local scorer benchmark")
    print(f"dataset={results['dataset']} examples={results['examples']} local_score_max={results['local_score_max']}")
    print()
    for index, summary in enumerate(results["ranked"], start=1):
        latency = summary["latency_ms"]
        print(
            f"{index}. {summary['model']} rank={summary['rank_score']} "
            f"route={summary['route_accuracy']:.2%} exact={summary['exact_accuracy']:.2%} "
            f"within1={summary['within_1_accuracy']:.2%} parse_failures={summary['parse_failures']} "
            f"load_ms={summary['setup_latency_ms']} "
            f"inference_ms avg={latency['avg']} p95={latency['p95']}"
        )
        if summary["errors"]:
            print(f"   errors={summary['errors']} (see JSON output for details)")
        if summary["setup_error"]:
            print(f"   setup_error={summary['setup_error']}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark Foundry Local scorer models for veilroute routing scores.")
    parser.add_argument("models", nargs="+", help="Candidate Foundry Local scorer model names.")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET, help="Labeled JSON prompt dataset.")
    parser.add_argument("--limit", type=int, default=None, help="Evaluate only the first N examples.")
    parser.add_argument("--local-score-max", type=int, default=1, help="Scores at or below this route locally.")
    parser.add_argument("--temperature", type=float, default=0.0, help="Scorer sampling temperature.")
    parser.add_argument("--max-tokens", type=int, default=8, help="Maximum scorer output tokens.")
    parser.add_argument("--format", choices=("text", "json"), default="text", help="Output format.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.limit is not None and args.limit < 1:
        raise SystemExit("--limit must be at least 1")

    examples = load_dataset(args.dataset)
    coverage = validate_dataset_coverage(examples)
    if not all(coverage.values()):
        missing = ", ".join(name for name, covered in coverage.items() if not covered)
        raise SystemExit(f"dataset coverage check failed: {missing}")
    if args.limit is not None:
        examples = examples[: args.limit]

    summaries = [
        run_model(
            model,
            examples,
            local_score_max=args.local_score_max,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
        )
        for model in args.models
    ]
    results = {
        "dataset": str(args.dataset),
        "examples": len(examples),
        "local_score_max": args.local_score_max,
        "ranked": rank_summaries(summaries),
    }
    if args.format == "json":
        print(json.dumps(results, indent=2, sort_keys=True))
    else:
        print_text(results)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
