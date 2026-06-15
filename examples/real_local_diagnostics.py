from __future__ import annotations

import argparse
import multiprocessing as mp
import time
import traceback
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class StepResult:
    label: str
    elapsed_ms: float
    ok: bool
    value: Any = None
    error: str | None = None


class FixedLocalScorer:
    def score(self, text: str) -> int:
        return 0


class BlockedCloudProvider:
    model = "blocked-cloud"

    def complete(self, messages, **opts):
        raise AssertionError("cloud provider should not be called in the forced-local diagnostic")

    def stream(self, messages, **opts):
        raise AssertionError("cloud provider should not be called in the forced-local diagnostic")


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify actual Foundry Local scorer and local model execution.")
    parser.add_argument("--scorer-model", default="phi-3.5-mini", help="Foundry Local scorer model to load and call.")
    parser.add_argument("--local-model", default="qwen2.5-0.5b", help="Foundry Local local route model to load and call.")
    parser.add_argument("--scorer-max-tokens", type=int, default=256, help="Max scorer output tokens.")
    parser.add_argument("--local-max-tokens", type=int, default=64, help="Max local model output tokens.")
    parser.add_argument("--timeout-seconds", type=int, default=300, help="Timeout per Foundry Local operation.")
    parser.add_argument("--local-prompt", default="Say hello in one short sentence.", help="Prompt for the local model call.")
    parser.add_argument(
        "--scorer-prompt",
        default="Say hello in one short sentence.",
        help="Prompt for the scorer model call.",
    )
    args = parser.parse_args()

    print_section("real Foundry Local diagnostic")
    print("This script uses actual Foundry Local providers in isolated child processes.")
    print("A timeout means Foundry Local model download/load/inference did not finish; it is not using the fake demo path.")
    print(f"timeout per step: {args.timeout_seconds}s")

    steps = [
        (
            "warm scorer provider",
            _warm_provider_child,
            (args.scorer_model,),
        ),
        (
            "warm local provider",
            _warm_provider_child,
            (args.local_model,),
        ),
        (
            "score prompt with real scorer model",
            _score_prompt_child,
            (args.scorer_model, args.scorer_prompt, args.scorer_max_tokens),
        ),
        (
            "run forced-local route through Router",
            _local_route_child,
            (args.local_model, args.local_prompt, args.local_max_tokens),
        ),
    ]

    results: list[StepResult] = []
    for label, target, child_args in steps:
        result = run_step(label, target, child_args, timeout_seconds=args.timeout_seconds)
        results.append(result)
        print_result(result)
        if not result.ok:
            print_section("stopped")
            print("The first failing step is the local Foundry issue to fix before full hybrid routing can be trusted.")
            return 1

    print_section("summary")
    for result in results:
        print(f"{result.label}: {result.elapsed_ms:.2f} ms")

    print_section("local route result")
    local_result = results[-1].value
    print("router route:", local_result["route"])
    print("router model:", local_result["model"])
    print("router text:")
    print(local_result["text"])
    print("tokens:", {"in": local_result["tokens_in"], "out": local_result["tokens_out"]})
    return 0


def run_step(label: str, target, args: tuple[Any, ...], *, timeout_seconds: int) -> StepResult:
    print(f"starting: {label}", flush=True)
    queue: mp.Queue = mp.Queue()
    process = mp.Process(target=target, args=(*args, queue), daemon=True)
    started = time.perf_counter()
    process.start()
    process.join(timeout_seconds)
    elapsed_ms = (time.perf_counter() - started) * 1000
    if process.is_alive():
        process.terminate()
        process.join(10)
        return StepResult(label=label, elapsed_ms=elapsed_ms, ok=False, error=f"timed out after {timeout_seconds}s")
    if queue.empty():
        return StepResult(label=label, elapsed_ms=elapsed_ms, ok=False, error=f"child exited with code {process.exitcode}")
    payload = queue.get()
    if payload["ok"]:
        return StepResult(label=label, elapsed_ms=elapsed_ms, ok=True, value=payload.get("value"))
    return StepResult(label=label, elapsed_ms=elapsed_ms, ok=False, error=payload["error"])


def _warm_provider_child(model: str, queue: mp.Queue) -> None:
    try:
        from veilrouter.providers.foundry_local import FoundryLocalProvider

        provider = FoundryLocalProvider(model=model)
        provider._ensure_client()
        queue.put({"ok": True, "value": {"model": model}})
    except Exception:
        queue.put({"ok": False, "error": traceback.format_exc()})


def _score_prompt_child(model: str, prompt: str, max_tokens: int, queue: mp.Queue) -> None:
    try:
        from veilrouter.providers.foundry_local import FoundryLocalProvider
        from veilrouter.scoring.llm_scorer import LlmDifficultyScorer

        provider = FoundryLocalProvider(model=model)
        scorer = LlmDifficultyScorer(provider, model=model, temperature=0, max_tokens=max_tokens)
        score = scorer.score(prompt)
        queue.put({"ok": True, "value": {"model": model, "score": score}})
    except Exception:
        queue.put({"ok": False, "error": traceback.format_exc()})


def _local_route_child(model: str, prompt: str, max_tokens: int, queue: mp.Queue) -> None:
    try:
        from veilrouter import Router, RouterConfig
        from veilrouter.pii.detector import RegexPiiDetector
        from veilrouter.providers.foundry_local import FoundryLocalProvider

        router = Router(
            RouterConfig(
                local_model=model,
                pii_regex_backstop=False,
            ),
            local_provider=FoundryLocalProvider(model=model),
            cloud_provider=BlockedCloudProvider(),
            scorer=FixedLocalScorer(),
            pii_detector=RegexPiiDetector(),
        )
        response = router.run(prompt, temperature=0, max_tokens=max_tokens)
        queue.put(
            {
                "ok": True,
                "value": {
                    "route": response.route,
                    "model": response.model,
                    "text": response.text,
                    "tokens_in": response.tokens_in,
                    "tokens_out": response.tokens_out,
                },
            }
        )
    except Exception:
        queue.put({"ok": False, "error": traceback.format_exc()})


def print_result(result: StepResult) -> None:
    status = "OK" if result.ok else "FAILED"
    print(f"{status}: {result.label} in {result.elapsed_ms:.2f} ms", flush=True)
    if result.value is not None:
        print(f"value: {result.value}", flush=True)
    if result.error:
        print(result.error, flush=True)


def print_section(title: str) -> None:
    print()
    print("=" * 80)
    print(title)
    print("=" * 80)


if __name__ == "__main__":
    raise SystemExit(main())
