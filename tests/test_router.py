import asyncio

import pytest

from veilroute.config import RouterConfig
from veilroute.errors import LocalContextExceededError, ProviderCallError
from veilroute.pii.detector import Span
from veilroute.providers.base import ChatChunk, ChatResponse
from veilroute.router import AsyncRouter, Router, flatten_messages, normalize_messages
from veilroute.telemetry.recorder import InMemoryTelemetryRecorder


class StaticScorer:
    def __init__(self, score: int) -> None:
        self.value = score
        self.inputs = []

    def score(self, text: str) -> int:
        self.inputs.append(text)
        return self.value

    async def ascore(self, text: str) -> int:
        self.inputs.append(text)
        return self.value


class EmailDetector:
    def detect(self, text: str):
        target = "ada@example.com"
        start = text.find(target)
        if start < 0:
            return []
        return [Span("contact.email", start, start + len(target), target, 1.0)]


class FakeProvider:
    def __init__(self, model: str, text: str = "ok", *, fail: bool = False, chunks=None) -> None:
        self.model = model
        self.text = text
        self.fail = fail
        self.chunks = chunks or []
        self.complete_calls = []
        self.stream_calls = []

    def complete(self, messages, **opts):
        self.complete_calls.append({"messages": messages, "opts": opts})
        if self.fail:
            raise ProviderCallError("provider failed")
        return ChatResponse(text=self.text, model=self.model, tokens_in=12, tokens_out=4, raw={"ok": True})

    def stream(self, messages, **opts):
        self.stream_calls.append({"messages": messages, "opts": opts})
        if self.fail:
            raise ProviderCallError("provider failed")
        yield from self.chunks

    async def acomplete(self, messages, **opts):
        return self.complete(messages, **opts)

    async def astream(self, messages, **opts):
        for chunk in self.stream(messages, **opts):
            yield chunk


def test_normalize_and_flatten_messages_support_strings_and_nested_content():
    messages = normalize_messages("hello")

    assert messages == [{"role": "user", "content": "hello"}]
    assert flatten_messages([{"role": "user", "content": [{"text": "one"}, {"text": "two"}]}]) == "one\ntwo"


@pytest.mark.parametrize("prompt", [123, [{"content": "missing role"}], ["not a dict"]])
def test_normalize_messages_rejects_invalid_prompts(prompt):
    with pytest.raises((TypeError, ValueError)):
        normalize_messages(prompt)


def test_router_returns_empty_response_without_calling_dependencies_for_blank_prompt():
    telemetry = InMemoryTelemetryRecorder()
    router = Router(
        RouterConfig(pii_regex_backstop=False),
        local_provider=FakeProvider("local"),
        cloud_provider=FakeProvider("cloud"),
        scorer=StaticScorer(1),
        pii_detector=EmailDetector(),
        telemetry=telemetry,
    )

    response = router.run("   ")

    assert response.route == "none"
    assert response.text == ""
    assert telemetry.records == []


def test_router_routes_low_scores_to_local_provider_and_records_savings():
    local = FakeProvider("local", text="local answer")
    cloud = FakeProvider("cloud", text="cloud answer")
    telemetry = InMemoryTelemetryRecorder()
    router = Router(
        RouterConfig(local_score_max=2, cloud_model="gpt-4o", pii_regex_backstop=False),
        local_provider=local,
        cloud_provider=cloud,
        scorer=StaticScorer(1),
        pii_detector=EmailDetector(),
        telemetry=telemetry,
    )

    response = router.run("hello", temperature=0)

    assert response.text == "local answer"
    assert response.route == "local"
    assert response.cost_estimate == 0.0
    assert response.cost_saved > 0.0
    assert len(local.complete_calls) == 1
    assert cloud.complete_calls == []
    assert telemetry.records[0].route == "local"


def test_router_routes_high_scores_to_cloud_with_redaction_and_restoration():
    local = FakeProvider("local", text="local answer")
    cloud = FakeProvider("cloud", text="Sent to [EMAIL_1]")
    sink = []
    router = Router(
        RouterConfig(local_score_max=1, cloud_model="gpt-4o", pii_regex_backstop=False, telemetry_sink=sink.append),
        local_provider=local,
        cloud_provider=cloud,
        scorer=StaticScorer(5),
        pii_detector=EmailDetector(),
    )

    response = router.run("Please email ada@example.com")

    assert response.text == "Sent to ada@example.com"
    assert response.route == "cloud"
    assert response.pii_detected is True
    assert response.redaction_count == 1
    assert response.redaction_categories == {"EMAIL": 1}
    assert cloud.complete_calls[0]["messages"][0]["content"] == "Please email [EMAIL_1]"
    assert "ada@example.com" not in cloud.complete_calls[0]["messages"][0]["content"]
    assert sink[0].pii_detected is True


def test_router_can_retry_cloud_failures_locally_with_original_messages():
    local = FakeProvider("local", text="local fallback")
    cloud = FakeProvider("cloud", fail=True)
    router = Router(
        RouterConfig(local_score_max=1, retry_cloud_failures_locally=True, pii_regex_backstop=False),
        local_provider=local,
        cloud_provider=cloud,
        scorer=StaticScorer(5),
        pii_detector=EmailDetector(),
    )

    response = router.run("Please email ada@example.com")

    assert response.text == "local fallback"
    assert response.route == "local"
    assert local.complete_calls[0]["messages"][0]["content"] == "Please email ada@example.com"


def test_router_routes_long_local_inputs_to_cloud_when_configured():
    local = FakeProvider("local", text="local answer")
    cloud = FakeProvider("cloud", text="cloud answer")
    router = Router(
        RouterConfig(
            local_score_max=5,
            max_local_input_tokens=1,
            route_long_inputs_to_cloud=True,
            pii_regex_backstop=False,
        ),
        local_provider=local,
        cloud_provider=cloud,
        scorer=StaticScorer(0),
        pii_detector=EmailDetector(),
    )

    response = router.run("this prompt is definitely longer than one token")

    assert response.route == "cloud"
    assert local.complete_calls == []
    assert len(cloud.complete_calls) == 1


def test_router_raises_when_long_local_inputs_cannot_be_routed_to_cloud():
    router = Router(
        RouterConfig(
            local_score_max=5,
            max_local_input_tokens=1,
            route_long_inputs_to_cloud=False,
            pii_regex_backstop=False,
        ),
        local_provider=FakeProvider("local"),
        cloud_provider=FakeProvider("cloud"),
        scorer=StaticScorer(0),
        pii_detector=EmailDetector(),
    )

    with pytest.raises(LocalContextExceededError):
        router.run("this prompt is definitely longer than one token")


def test_router_stream_restores_cloud_chunks_and_records_usage():
    cloud = FakeProvider(
        "cloud",
        chunks=[
            ChatChunk("Hello ", model="cloud"),
            ChatChunk("[EMAIL", model="cloud"),
            ChatChunk("_1]", model="cloud", tokens_in=10, tokens_out=2),
        ],
    )
    telemetry = InMemoryTelemetryRecorder()
    router = Router(
        RouterConfig(local_score_max=1, pii_regex_backstop=False),
        local_provider=FakeProvider("local"),
        cloud_provider=cloud,
        scorer=StaticScorer(5),
        pii_detector=EmailDetector(),
        telemetry=telemetry,
    )

    chunks = list(router.stream("Email ada@example.com"))

    assert [chunk.text for chunk in chunks] == ["Hello ", "ada@example.com"]
    assert cloud.stream_calls[0]["messages"][0]["content"] == "Email [EMAIL_1]"
    assert telemetry.records[0].tokens_in == 10
    assert telemetry.records[0].tokens_out == 2


def test_async_router_run_routes_and_restores_cloud_responses():
    async def exercise():
        router = AsyncRouter(
            RouterConfig(local_score_max=1, pii_regex_backstop=False),
            local_provider=FakeProvider("local"),
            cloud_provider=FakeProvider("cloud", text="Async [EMAIL_1]"),
            scorer=StaticScorer(5),
            pii_detector=EmailDetector(),
        )

        return await router.run("Email ada@example.com")

    response = asyncio.run(exercise())

    assert response.text == "Async ada@example.com"
    assert response.route == "cloud"


def test_async_router_stream_restores_cloud_chunks():
    async def exercise():
        router = AsyncRouter(
            RouterConfig(local_score_max=1, pii_regex_backstop=False),
            local_provider=FakeProvider("local"),
            cloud_provider=FakeProvider("cloud", chunks=[ChatChunk("[EMAIL"), ChatChunk("_1]")]),
            scorer=StaticScorer(5),
            pii_detector=EmailDetector(),
        )

        return [chunk async for chunk in router.stream("Email ada@example.com")]

    chunks = asyncio.run(exercise())

    assert [chunk.text for chunk in chunks] == ["ada@example.com"]
