from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Iterable, Iterator

from veilrouter.config import RouterConfig
from veilrouter.errors import LocalContextExceededError, ProviderCallError
from veilrouter.pii.detector import PiiDetector
from veilrouter.pii.redactor import RedactionResult, Redactor
from veilrouter.pii.restorer import StreamRestorer, restore_text
from veilrouter.providers.base import ChatChunk, ChatProvider, ChatResponse, Message
from veilrouter.providers.foundry_local import FoundryLocalProvider
from veilrouter.providers.openai_compatible import OpenAICompatibleProvider
from veilrouter.scoring.llm_scorer import LlmDifficultyScorer
from veilrouter.telemetry.pricing import estimate_cost
from veilrouter.telemetry.recorder import InMemoryTelemetryRecorder, TelemetryRecord, now_utc

logger = logging.getLogger("veilrouter")


@dataclass(frozen=True, slots=True)
class RouterResponse:
    text: str
    route: str
    score: int | None
    model: str | None
    tokens_in: int = 0
    tokens_out: int = 0
    cost_estimate: float = 0.0
    cost_saved: float = 0.0
    pii_detected: bool = False
    redaction_count: int = 0
    redaction_categories: dict[str, int] = field(default_factory=dict)
    raw: Any = None


@dataclass(frozen=True, slots=True)
class RouterChunk:
    text: str
    route: str
    score: int | None
    model: str | None
    tokens_in: int = 0
    tokens_out: int = 0


class Router:
    def __init__(
        self,
        config: RouterConfig | None = None,
        *,
        local_provider: ChatProvider | None = None,
        cloud_provider: ChatProvider | None = None,
        scorer: Any | None = None,
        pii_detector: Any | None = None,
        telemetry: InMemoryTelemetryRecorder | None = None,
    ) -> None:
        self.config = config or RouterConfig()
        if self.config.debug:
            logger.setLevel(logging.DEBUG)
        self.local_provider = local_provider or FoundryLocalProvider(model=self.config.local_model)
        self.cloud_provider = cloud_provider or OpenAICompatibleProvider(
            model=self.config.cloud_model,
            base_url=self.config.cloud_endpoint,
            api_key=self.config.cloud_api_key,
        )
        self.scorer = scorer or LlmDifficultyScorer(
            FoundryLocalProvider(model=self.config.scorer_model or self.config.local_model),
            model=self.config.scorer_model or self.config.local_model,
            temperature=self.config.scorer_temperature,
            max_tokens=self.config.scorer_max_tokens,
        )
        detector = pii_detector
        if detector is None:
            detector = PiiDetector(self.config.pii_model_path, min_score=self.config.pii_min_score)
        self.redactor = Redactor(detector, regex_backstop=self.config.pii_regex_backstop)
        self.telemetry = telemetry or InMemoryTelemetryRecorder()

    def run(self, prompt: str | list[Message], **opts: Any) -> RouterResponse:
        messages = normalize_messages(prompt)
        if is_empty_messages(messages):
            return RouterResponse(text="", route="none", score=None, model=None)

        started = time.perf_counter()
        score = int(self.scorer.score(flatten_messages(messages)))
        route = self._route_for_score(score, messages)
        logger.debug("route=%s score=%s model=%s", route, score, self._model_for_route(route))

        if route == "local":
            provider_response = self.local_provider.complete(messages, **opts)
            response = self._build_response(provider_response, route, score, started, None)
        else:
            response = self._run_cloud(messages, score, started, opts)
        return response

    def stream(self, prompt: str | list[Message], **opts: Any) -> Iterator[RouterChunk]:
        messages = normalize_messages(prompt)
        if is_empty_messages(messages):
            return iter(())
        return self._stream_impl(messages, opts)

    def _stream_impl(self, messages: list[Message], opts: dict[str, Any]) -> Iterator[RouterChunk]:
        started = time.perf_counter()
        score = int(self.scorer.score(flatten_messages(messages)))
        route = self._route_for_score(score, messages)
        tokens_in = estimate_tokens(flatten_messages(messages))
        tokens_out = 0
        final_in: int | None = None
        final_out: int | None = None
        text_parts: list[str] = []
        redaction: RedactionResult | None = None
        restorer: StreamRestorer | None = None
        stream_messages = messages
        provider = self.local_provider
        if route == "cloud":
            redaction = self.redactor.redact_messages(messages)
            stream_messages = redaction.messages
            restorer = StreamRestorer(redaction.placeholder_to_original)
            provider = self.cloud_provider

        for chunk in provider.stream(stream_messages, **opts):
            text = chunk.text
            if restorer is not None:
                text = restorer.feed(text)
            if not text and chunk.tokens_in is None and chunk.tokens_out is None:
                continue
            tokens_out += estimate_tokens(text)
            text_parts.append(text)
            final_in = chunk.tokens_in if chunk.tokens_in is not None else final_in
            final_out = chunk.tokens_out if chunk.tokens_out is not None else final_out
            yield RouterChunk(text=text, route=route, score=score, model=chunk.model or self._model_for_route(route))

        if restorer is not None:
            tail = restorer.finish()
            if tail:
                tokens_out += estimate_tokens(tail)
                text_parts.append(tail)
                yield RouterChunk(text=tail, route=route, score=score, model=self._model_for_route(route))
        self._record(
            route=route,
            score=score,
            model=self._model_for_route(route),
            tokens_in=final_in if final_in is not None else tokens_in,
            tokens_out=final_out if final_out is not None else tokens_out,
            latency_ms=(time.perf_counter() - started) * 1000,
            redaction=redaction,
        )

    def _run_cloud(self, messages: list[Message], score: int, started: float, opts: dict[str, Any]) -> RouterResponse:
        redaction = self.redactor.redact_messages(messages)
        logger.debug(
            "cloud redactions count=%s categories=%s model=%s",
            redaction.redaction_count,
            redaction.categories,
            self.config.cloud_model,
        )
        try:
            provider_response = self.cloud_provider.complete(redaction.messages, **opts)
        except ProviderCallError:
            if not self.config.retry_cloud_failures_locally:
                raise
            provider_response = self.local_provider.complete(messages, **opts)
            return self._build_response(provider_response, "local", score, started, None)
        restored = ChatResponse(
            text=restore_text(provider_response.text, redaction.placeholder_to_original),
            model=provider_response.model,
            tokens_in=provider_response.tokens_in,
            tokens_out=provider_response.tokens_out,
            raw=provider_response.raw,
        )
        return self._build_response(restored, "cloud", score, started, redaction)

    def _route_for_score(self, score: int, messages: list[Message]) -> str:
        if score <= self.config.local_score_max:
            if self.config.max_local_input_tokens is not None:
                tokens = estimate_tokens(flatten_messages(messages))
                if tokens > self.config.max_local_input_tokens:
                    if self.config.route_long_inputs_to_cloud:
                        return "cloud"
                    raise LocalContextExceededError(
                        f"input estimate {tokens} tokens exceeds local context {self.config.max_local_input_tokens}"
                    )
            return "local"
        return "cloud"

    def _build_response(
        self,
        provider_response: ChatResponse,
        route: str,
        score: int,
        started: float,
        redaction: RedactionResult | None,
    ) -> RouterResponse:
        tokens_in = provider_response.tokens_in if provider_response.tokens_in is not None else 0
        tokens_out = provider_response.tokens_out if provider_response.tokens_out is not None else estimate_tokens(provider_response.text)
        if tokens_in == 0:
            tokens_in = estimate_tokens(provider_response.text)
        latency_ms = (time.perf_counter() - started) * 1000
        cost = estimate_cost(self.config.cloud_model, tokens_in, tokens_out, self.config.pricing) if route == "cloud" else 0.0
        saved = estimate_cost(self.config.cloud_model, tokens_in, tokens_out, self.config.pricing) if route == "local" else 0.0
        self._record(
            route=route,
            score=score,
            model=provider_response.model or self._model_for_route(route),
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            latency_ms=latency_ms,
            redaction=redaction,
            cost_estimate=cost,
            cost_saved=saved,
        )
        return RouterResponse(
            text=provider_response.text,
            route=route,
            score=score,
            model=provider_response.model or self._model_for_route(route),
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            cost_estimate=cost,
            cost_saved=saved,
            pii_detected=bool(redaction and redaction.redaction_count),
            redaction_count=redaction.redaction_count if redaction else 0,
            redaction_categories=redaction.categories if redaction else {},
            raw=provider_response.raw,
        )

    def _record(
        self,
        *,
        route: str,
        score: int | None,
        model: str | None,
        tokens_in: int,
        tokens_out: int,
        latency_ms: float,
        redaction: RedactionResult | None,
        cost_estimate: float | None = None,
        cost_saved: float | None = None,
    ) -> None:
        cost = estimate_cost(self.config.cloud_model, tokens_in, tokens_out, self.config.pricing) if cost_estimate is None and route == "cloud" else (cost_estimate or 0.0)
        saved = estimate_cost(self.config.cloud_model, tokens_in, tokens_out, self.config.pricing) if cost_saved is None and route == "local" else (cost_saved or 0.0)
        record = TelemetryRecord(
            timestamp=now_utc(),
            route=route,
            score=score,
            model=model,
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            latency_ms=latency_ms,
            pii_detected=bool(redaction and redaction.redaction_count),
            redaction_count=redaction.redaction_count if redaction else 0,
            redaction_categories=redaction.categories if redaction else {},
            cost_estimate=cost,
            cost_saved=saved,
        )
        self.telemetry.record(record)
        if self.config.telemetry_sink is not None:
            self.config.telemetry_sink(record)

    def _model_for_route(self, route: str) -> str | None:
        if route == "local":
            return getattr(self.local_provider, "model", self.config.local_model)
        if route == "cloud":
            return getattr(self.cloud_provider, "model", self.config.cloud_model)
        return None


class AsyncRouter(Router):
    async def run(self, prompt: str | list[Message], **opts: Any) -> RouterResponse:  # type: ignore[override]
        messages = normalize_messages(prompt)
        if is_empty_messages(messages):
            return RouterResponse(text="", route="none", score=None, model=None)
        started = time.perf_counter()
        score = await self._score_async(flatten_messages(messages))
        route = self._route_for_score(score, messages)
        if route == "local":
            provider_response = await _acomplete(self.local_provider, messages, opts)
            return self._build_response(provider_response, route, score, started, None)
        redaction = self.redactor.redact_messages(messages)
        try:
            provider_response = await _acomplete(self.cloud_provider, redaction.messages, opts)
        except ProviderCallError:
            if not self.config.retry_cloud_failures_locally:
                raise
            provider_response = await _acomplete(self.local_provider, messages, opts)
            return self._build_response(provider_response, "local", score, started, None)
        restored = ChatResponse(
            text=restore_text(provider_response.text, redaction.placeholder_to_original),
            model=provider_response.model,
            tokens_in=provider_response.tokens_in,
            tokens_out=provider_response.tokens_out,
            raw=provider_response.raw,
        )
        return self._build_response(restored, route, score, started, redaction)

    async def stream(self, prompt: str | list[Message], **opts: Any) -> AsyncIterator[RouterChunk]:  # type: ignore[override]
        messages = normalize_messages(prompt)
        if is_empty_messages(messages):
            return
        started = time.perf_counter()
        score = await self._score_async(flatten_messages(messages))
        route = self._route_for_score(score, messages)
        redaction: RedactionResult | None = None
        restorer: StreamRestorer | None = None
        provider = self.local_provider
        stream_messages = messages
        if route == "cloud":
            redaction = self.redactor.redact_messages(messages)
            stream_messages = redaction.messages
            restorer = StreamRestorer(redaction.placeholder_to_original)
            provider = self.cloud_provider

        tokens_in = estimate_tokens(flatten_messages(messages))
        tokens_out = 0
        final_in: int | None = None
        final_out: int | None = None
        async for chunk in _astream(provider, stream_messages, opts):
            text = restorer.feed(chunk.text) if restorer else chunk.text
            tokens_out += estimate_tokens(text)
            final_in = chunk.tokens_in if chunk.tokens_in is not None else final_in
            final_out = chunk.tokens_out if chunk.tokens_out is not None else final_out
            if text:
                yield RouterChunk(text=text, route=route, score=score, model=chunk.model or self._model_for_route(route))
        if restorer is not None:
            tail = restorer.finish()
            if tail:
                tokens_out += estimate_tokens(tail)
                yield RouterChunk(text=tail, route=route, score=score, model=self._model_for_route(route))
        self._record(
            route=route,
            score=score,
            model=self._model_for_route(route),
            tokens_in=final_in if final_in is not None else tokens_in,
            tokens_out=final_out if final_out is not None else tokens_out,
            latency_ms=(time.perf_counter() - started) * 1000,
            redaction=redaction,
        )

    async def _score_async(self, text: str) -> int:
        if hasattr(self.scorer, "ascore"):
            return int(await self.scorer.ascore(text))
        if hasattr(self.scorer, "score") and asyncio.iscoroutinefunction(self.scorer.score):
            return int(await self.scorer.score(text))
        return await asyncio.to_thread(self.scorer.score, text)


def run(prompt: str | list[Message], config: RouterConfig | None = None, **opts: Any) -> RouterResponse:
    return Router(config).run(prompt, **opts)


def normalize_messages(prompt: str | list[Message]) -> list[Message]:
    if isinstance(prompt, str):
        return [{"role": "user", "content": prompt}]
    if not isinstance(prompt, list):
        raise TypeError("prompt must be a string or a list of chat messages")
    normalized: list[Message] = []
    for item in prompt:
        if not isinstance(item, dict):
            raise TypeError("each chat message must be a dict")
        if "role" not in item:
            raise ValueError("each chat message must include role")
        normalized.append(dict(item))
    return normalized


def flatten_messages(messages: list[Message]) -> str:
    parts: list[str] = []
    for message in messages:
        if isinstance(message, dict) and "content" in message:
            for value in _walk_strings(message["content"]):
                parts.append(value)
    return "\n".join(parts)


def is_empty_messages(messages: list[Message]) -> bool:
    return not flatten_messages(messages).strip()


def estimate_tokens(text: str) -> int:
    return max(1, len(text) // 4) if text else 0


def _walk_strings(value: Any) -> Iterable[str]:
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for child in value.values():
            yield from _walk_strings(child)
    elif isinstance(value, list):
        for child in value:
            yield from _walk_strings(child)


async def _acomplete(provider: Any, messages: list[Message], opts: dict[str, Any]) -> ChatResponse:
    if hasattr(provider, "acomplete"):
        return await provider.acomplete(messages, **opts)
    return await asyncio.to_thread(provider.complete, messages, **opts)


async def _astream(provider: Any, messages: list[Message], opts: dict[str, Any]) -> AsyncIterator[ChatChunk]:
    if hasattr(provider, "astream"):
        async for chunk in provider.astream(messages, **opts):
            yield chunk
        return
    chunks = await asyncio.to_thread(lambda: list(provider.stream(messages, **opts)))
    for chunk in chunks:
        yield chunk
