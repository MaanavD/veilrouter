import asyncio
from types import SimpleNamespace

import pytest

from veilroute.errors import ConfigurationError, ProviderCallError, ProviderSetupError
from veilroute.providers.foundry_local import FoundryLocalProvider
from veilroute.providers.openai_compatible import OpenAICompatibleProvider


class RecordingCreate:
    def __init__(self, response):
        self.response = response
        self.calls = []

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        if isinstance(self.response, Exception):
            raise self.response
        return self.response


def _openai_response(text="hello", model="returned-model", prompt_tokens=7, completion_tokens=3):
    return SimpleNamespace(
        model=model,
        usage=SimpleNamespace(prompt_tokens=prompt_tokens, completion_tokens=completion_tokens),
        choices=[SimpleNamespace(message=SimpleNamespace(content=text))],
    )


def _stream_chunk(text="", model="returned-model", prompt_tokens=None, completion_tokens=None):
    usage = None
    if prompt_tokens is not None or completion_tokens is not None:
        usage = SimpleNamespace(prompt_tokens=prompt_tokens, completion_tokens=completion_tokens)
    choices = [SimpleNamespace(delta=SimpleNamespace(content=text))] if text else []
    return SimpleNamespace(model=model, usage=usage, choices=choices)


def test_openai_compatible_complete_maps_sdk_response_without_network():
    create = RecordingCreate(_openai_response(text="cloud answer"))
    provider = OpenAICompatibleProvider(model="gpt-test", base_url="https://example.invalid/v1", api_key="key")
    provider._client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))

    response = provider.complete([{"role": "user", "content": "hello"}], temperature=0)

    assert response.text == "cloud answer"
    assert response.model == "returned-model"
    assert response.tokens_in == 7
    assert response.tokens_out == 3
    assert create.calls[0]["model"] == "gpt-test"
    assert create.calls[0]["temperature"] == 0


def test_openai_compatible_stream_requests_usage_and_maps_chunks_without_network():
    create = RecordingCreate(
        [
            _stream_chunk("Hel"),
            _stream_chunk("lo"),
            _stream_chunk("", prompt_tokens=5, completion_tokens=2),
        ]
    )
    provider = OpenAICompatibleProvider(model="gpt-test", base_url=None, api_key="key")
    provider._client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))

    chunks = list(provider.stream([{"role": "user", "content": "hello"}]))

    assert [chunk.text for chunk in chunks] == ["Hel", "lo", ""]
    assert chunks[-1].tokens_in == 5
    assert chunks[-1].tokens_out == 2
    assert create.calls[0]["stream"] is True
    assert create.calls[0]["stream_options"] == {"include_usage": True}


def test_openai_compatible_requires_api_key_before_creating_client():
    provider = OpenAICompatibleProvider(model="gpt-test", base_url=None, api_key=None)

    with pytest.raises(ConfigurationError):
        provider._sync_client()


def test_openai_compatible_wraps_sdk_failures():
    create = RecordingCreate(RuntimeError("provider down"))
    provider = OpenAICompatibleProvider(model="gpt-test", base_url=None, api_key="key")
    provider._client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))

    with pytest.raises(ProviderCallError, match="cloud provider call failed"):
        provider.complete([{"role": "user", "content": "hello"}])


def test_openai_compatible_async_complete_maps_sdk_response_without_network():
    class AsyncCreate:
        def __init__(self):
            self.calls = []

        async def __call__(self, **kwargs):
            self.calls.append(kwargs)
            return _openai_response(text="async cloud")

    async def exercise():
        create = AsyncCreate()
        provider = OpenAICompatibleProvider(model="gpt-test", base_url=None, api_key="key")
        provider._async_client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))

        response = await provider.acomplete([{"role": "user", "content": "hello"}])

        return response, create.calls

    response, calls = asyncio.run(exercise())

    assert response.text == "async cloud"
    assert calls[0]["model"] == "gpt-test"


def test_foundry_local_complete_maps_openai_shaped_client_without_sdk():
    create = RecordingCreate(_openai_response(text="local answer", model="local-returned"))
    provider = FoundryLocalProvider(model="local-test")
    provider._client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))

    response = provider.complete([{"role": "user", "content": "hello"}], top_p=0.9)

    assert response.text == "local answer"
    assert response.model == "local-returned"
    assert response.tokens_in == 7
    assert response.tokens_out == 3
    assert create.calls[0]["model"] == "local-test"
    assert create.calls[0]["top_p"] == 0.9


def test_foundry_local_stream_uses_streaming_client_when_available():
    create = RecordingCreate([_stream_chunk("A"), _stream_chunk("B", prompt_tokens=4, completion_tokens=2)])
    provider = FoundryLocalProvider(model="local-test")
    provider._client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))

    chunks = list(provider.stream([{"role": "user", "content": "hello"}]))

    assert [chunk.text for chunk in chunks] == ["A", "B"]
    assert chunks[-1].tokens_in == 4
    assert chunks[-1].tokens_out == 2
    assert create.calls[0]["stream"] is True


def test_foundry_local_falls_back_to_complete_for_simple_clients():
    class SimpleClient:
        def complete(self, **kwargs):
            return SimpleNamespace(text=f"completed by {kwargs['model']}")

    provider = FoundryLocalProvider(model="local-test")
    provider._client = SimpleClient()

    chunks = list(provider.stream([{"role": "user", "content": "hello"}]))

    assert [chunk.text for chunk in chunks] == ["completed by local-test"]
    assert chunks[0].model == "local-test"


def test_foundry_local_reports_unsupported_client_shape():
    provider = FoundryLocalProvider(model="local-test")
    provider._client = object()

    with pytest.raises(ProviderSetupError, match="unsupported Foundry Local chat client shape"):
        provider.complete([{"role": "user", "content": "hello"}])
