from __future__ import annotations

import argparse
import json
import math
import re
import sys
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from veilrouter import Router, RouterConfig  # noqa: E402
from veilrouter.pii.detector import RegexPiiDetector  # noqa: E402
from veilrouter.providers.base import ChatChunk, ChatResponse, Message  # noqa: E402


PLACEHOLDER_RE = re.compile(r"\[[A-Z][A-Z0-9_]*_[1-9][0-9]*\]")


@dataclass(frozen=True, slots=True)
class Workload:
    name: str
    prompt: str


class HeuristicScorer:
    def score(self, text: str) -> int:
        lowered = text.lower()
        if any(word in lowered for word in ("contract", "incident", "risk", "architecture", "compliance")):
            return 4
        if len(text) > 220 or "summarize" in lowered:
            return 3
        if "explain" in lowered:
            return 2
        return 1


class DeterministicProvider:
    def __init__(self, model: str, *, label: str, delay_ms: float = 0.0) -> None:
        self.model = model
        self.label = label
        self.delay_s = max(0.0, delay_ms) / 1000.0

    def complete(self, messages: list[Message], **opts: Any) -> ChatResponse:
        if self.delay_s:
            time.sleep(self.delay_s)
        text = flatten(messages)
        placeholders = PLACEHOLDER_RE.findall(text)
        suffix = f" Reference {placeholders[0]}." if placeholders else ""
        output = f"{self.label} response for {word_count(text)} input words.{suffix}"
        return ChatResponse(
            text=output,
            model=self.model,
            tokens_in=estimate_tokens(text),
            tokens_out=estimate_tokens(output),
        )

    def stream(self, messages: list[Message], **opts: Any) -> Iterable[ChatChunk]:
        response = self.complete(messages, **opts)
        words = response.text.split(" ")
        for index, word in enumerate(words):
            yield ChatChunk(text=word + ("" if index == len(words) - 1 else " "), model=self.model)
        yield ChatChunk(
            text="",
            model=self.model,
            tokens_in=response.tokens_in,
            tokens_out=response.tokens_out,
        )


def default_workloads() -> list[Workload]:
    return [
        Workload("local_greeting", "Say hello in one short sentence."),
        Workload("local_factual", "Name one thing TLS provides for a web request."),
        Workload(
            "cloud_contract_pii",
            "Summarize the contract risk for Jane Doe at jane.doe@example.com and note next steps.",
        ),
        Workload(
            "cloud_incident_phone",
            "Draft an incident response plan for account 4242-4242-4242-4242. Call +1 (425) 555-0199 if blocked.",
        ),
        Workload(
            "cloud_architecture",
            "Compare two routing architectures for compliance-heavy workloads and recommend a migration plan.",
        ),
    ]


def build_router(delay_ms: float) -> Router:
    config = RouterConfig(
        local_model="fake-local",
        cloud_model="gpt-4o",
        local_score_max=1,
        pii_regex_backstop=True,
    )
    return Router(
        config,
        local_provider=DeterministicProvider("fake-local", label="local", delay_ms=delay_ms),
        cloud_provider=DeterministicProvider("fake-cloud", label="cloud", delay_ms=delay_ms),
        scorer=HeuristicScorer(),
        pii_detector=RegexPiiDetector(),
    )


def run_benchmark(iterations: int, warmup: int, mode: str, delay_ms: float) -> dict[str, Any]:
    router = build_router(delay_ms)
    workloads = default_workloads()
    for _ in range(warmup):
        for workload in workloads:
            run_once(router, workload, mode)

    durations: list[float] = []
    route_counts: Counter[str] = Counter()
    redactions = 0
    cost_saved = 0.0
    tokens_in = 0
    tokens_out = 0

    for _ in range(iterations):
        for workload in workloads:
            started = time.perf_counter()
            response = run_once(router, workload, mode)
            durations.append((time.perf_counter() - started) * 1000)
            route_counts[response["route"]] += 1
            redactions += response["redaction_count"]
            cost_saved += response["cost_saved"]
            tokens_in += response["tokens_in"]
            tokens_out += response["tokens_out"]

    return {
        "mode": mode,
        "iterations": iterations,
        "workloads": len(workloads),
        "calls": len(durations),
        "provider_delay_ms": delay_ms,
        "latency_ms": {
            "avg": round(sum(durations) / len(durations), 3) if durations else 0.0,
            "p50": round(percentile(durations, 50), 3),
            "p95": round(percentile(durations, 95), 3),
            "max": round(max(durations), 3) if durations else 0.0,
        },
        "routes": dict(sorted(route_counts.items())),
        "redactions": redactions,
        "estimated_cost_saved": round(cost_saved, 8),
        "tokens_in": tokens_in,
        "tokens_out": tokens_out,
    }


def run_once(router: Router, workload: Workload, mode: str) -> dict[str, Any]:
    if mode == "stream":
        chunks = list(router.stream(workload.prompt))
        last = chunks[-1] if chunks else None
        record = router.telemetry.records[-1]
        return {
            "route": last.route if last else "none",
            "redaction_count": record.redaction_count,
            "cost_saved": record.cost_saved,
            "tokens_in": record.tokens_in,
            "tokens_out": record.tokens_out,
        }
    response = router.run(workload.prompt)
    return {
        "route": response.route,
        "redaction_count": response.redaction_count,
        "cost_saved": response.cost_saved,
        "tokens_in": response.tokens_in,
        "tokens_out": response.tokens_out,
    }


def flatten(messages: list[Message]) -> str:
    return "\n".join(_walk_strings(messages))


def _walk_strings(value: Any) -> Iterable[str]:
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for child in value.values():
            yield from _walk_strings(child)
    elif isinstance(value, list):
        for child in value:
            yield from _walk_strings(child)


def word_count(text: str) -> int:
    return len([part for part in text.split() if part])


def estimate_tokens(text: str) -> int:
    return max(1, len(text) // 4) if text else 0


def percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, math.ceil((pct / 100) * len(ordered)) - 1))
    return ordered[index]


def print_text(summary: dict[str, Any]) -> None:
    latency = summary["latency_ms"]
    print("veilrouter core benchmark (offline deterministic)")
    print(f"mode={summary['mode']} calls={summary['calls']} workloads={summary['workloads']} iterations={summary['iterations']}")
    print(
        "latency_ms "
        f"avg={latency['avg']} p50={latency['p50']} p95={latency['p95']} max={latency['max']}"
    )
    print(f"routes={summary['routes']} redactions={summary['redactions']} cost_saved={summary['estimated_cost_saved']}")
    print(f"tokens_in={summary['tokens_in']} tokens_out={summary['tokens_out']}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run an offline deterministic benchmark for veilrouter core routing.")
    parser.add_argument("--iterations", type=int, default=50, help="Measured iterations over the built-in workload set.")
    parser.add_argument("--warmup", type=int, default=5, help="Warmup iterations excluded from timings.")
    parser.add_argument("--mode", choices=("run", "stream"), default="run", help="Benchmark Router.run or Router.stream.")
    parser.add_argument("--provider-delay-ms", type=float, default=0.0, help="Optional fake provider delay per call.")
    parser.add_argument("--format", choices=("text", "json"), default="text", help="Output format.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.iterations < 1:
        raise SystemExit("--iterations must be at least 1")
    if args.warmup < 0:
        raise SystemExit("--warmup must be non-negative")
    summary = run_benchmark(args.iterations, args.warmup, args.mode, args.provider_delay_ms)
    if args.format == "json":
        print(json.dumps(summary, indent=2, sort_keys=True))
    else:
        print_text(summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
