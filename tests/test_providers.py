import asyncio
import importlib
import sys
from types import SimpleNamespace

import pytest

from veilrouter.errors import ConfigurationError, ProviderCallError, ProviderSetupError
import veilrouter.providers.foundry_local as foundry_local_module
from veilrouter.providers.foundry_local import FoundryLocalProvider
from veilrouter.providers.openai_compatible import OpenAICompatibleProvider


class RecordingCreate:
    def __init__(self, response):
        self.response = response
        self.calls = []

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        if isinstance(self.response, Exception):
            raise self.response
        return self.response


def _openai_response(text="hello", model="returned-model", prompt_tokens=7, completion_tokens=3, tool_calls=None):
    return SimpleNamespace(
        model=model,
        usage=SimpleNamespace(prompt_tokens=prompt_tokens, completion_tokens=completion_tokens),
        choices=[SimpleNamespace(message=SimpleNamespace(content=text, tool_calls=tool_calls))],
    )


def _stream_chunk(text="", model="returned-model", prompt_tokens=None, completion_tokens=None):
    usage = None
    if prompt_tokens is not None or completion_tokens is not None:
        usage = SimpleNamespace(prompt_tokens=prompt_tokens, completion_tokens=completion_tokens)
    choices = [SimpleNamespace(delta=SimpleNamespace(content=text))] if text else []
    return SimpleNamespace(model=model, usage=usage, choices=choices)


def test_openai_compatible_complete_maps_sdk_response_without_network():
    tool_calls = [{"id": "call_1"}]
    create = RecordingCreate(_openai_response(text="cloud answer", tool_calls=tool_calls))
    provider = OpenAICompatibleProvider(model="gpt-test", base_url="https://example.invalid/v1", api_key="key")
    provider._client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))

    response = provider.complete([{"role": "user", "content": "hello"}], temperature=0)

    assert response.text == "cloud answer"
    assert response.model == "returned-model"
    assert response.tokens_in == 7
    assert response.tokens_out == 3
    assert response.tool_calls == tool_calls
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


@pytest.mark.parametrize(
    ("sdk_response", "expected_text"),
    [
        ("plain sdk answer", "plain sdk answer"),
        (SimpleNamespace(text="text sdk answer"), "text sdk answer"),
        (SimpleNamespace(content="content sdk answer"), "content sdk answer"),
        (SimpleNamespace(message=SimpleNamespace(content="message sdk answer")), "message sdk answer"),
        (
            SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        message=SimpleNamespace(content="choice sdk answer"),
                    )
                ]
            ),
            "choice sdk answer",
        ),
    ],
)
def test_foundry_local_complete_maps_foundry_sdk_chat_client_shapes(sdk_response, expected_text):
    class SdkChatClient:
        def __init__(self):
            self.settings = SimpleNamespace(temperature=None, max_tokens=None)
            self.calls = []

        def complete_chat(self, messages):
            self.calls.append(messages)
            return sdk_response

    client = SdkChatClient()
    provider = FoundryLocalProvider(model="local-test")
    provider._client = client

    response = provider.complete([{"role": "user", "content": "hello"}], temperature=0, max_tokens=8)

    assert response.text == expected_text
    assert client.calls == [[{"role": "user", "content": "hello"}]]
    assert client.settings.temperature == 0
    assert client.settings.max_tokens == 8


def test_foundry_local_uses_installed_foundry_local_sdk_when_legacy_import_missing(monkeypatch):
    class Configuration:
        def __init__(self, *, app_name):
            self.app_name = app_name

    class FakeChatClient:
        def __init__(self):
            self.settings = SimpleNamespace(temperature=None, max_tokens=None)
            self.calls = []

        def complete_chat(self, messages):
            self.calls.append(messages)
            return SimpleNamespace(message=SimpleNamespace(content="sdk answer"))

    class FakeModel:
        def __init__(self):
            self.downloads = 0
            self.loads = 0
            self.client = FakeChatClient()

        def download(self):
            self.downloads += 1

        def load(self):
            self.loads += 1

        def get_chat_client(self):
            return self.client

    class FakeCatalog:
        def __init__(self, model):
            self.model = model
            self.requested = []

        def get_model(self, name):
            self.requested.append(name)
            return self.model

    model = FakeModel()
    manager = SimpleNamespace(catalog=FakeCatalog(model))

    class FakeFoundryLocalManager:
        instance = manager
        initialized_with = []

        @classmethod
        def initialize(cls, config):
            cls.initialized_with.append(config)

    monkeypatch.setitem(
        sys.modules,
        "foundry_local_sdk",
        SimpleNamespace(Configuration=Configuration, FoundryLocalManager=FakeFoundryLocalManager),
    )
    monkeypatch.setitem(sys.modules, "foundry_local", None)

    provider = FoundryLocalProvider(model="qwen3-0.6b")

    assert model.downloads == 0
    assert model.loads == 0

    response = provider.complete([{"role": "user", "content": "hello"}], temperature=0, max_tokens=8)

    assert response.text == "sdk answer"
    assert FakeFoundryLocalManager.initialized_with[0].app_name == "veilrouter"
    assert manager.catalog.requested == ["qwen3-0.6b"]
    assert model.downloads == 1
    assert model.loads == 1
    assert model.client.settings.temperature == 0
    assert model.client.settings.max_tokens == 8


def test_foundry_local_sdk_reuses_existing_singleton(monkeypatch):
    class Configuration:
        def __init__(self, *, app_name):
            self.app_name = app_name

    class FakeChatClient:
        def __init__(self, text):
            self.text = text
            self.settings = SimpleNamespace()

        def complete_chat(self, messages):
            return self.text

    class FakeModel:
        def __init__(self, name):
            self.name = name

        def download(self):
            pass

        def load(self):
            pass

        def get_chat_client(self):
            return FakeChatClient(f"answer from {self.name}")

    class FakeCatalog:
        def get_model(self, name):
            return FakeModel(name)

    manager = SimpleNamespace(catalog=FakeCatalog())

    class FakeFoundryLocalManager:
        instance = manager
        initialize_calls = 0

        @classmethod
        def initialize(cls, config):
            cls.initialize_calls += 1
            if cls.initialize_calls > 1:
                raise RuntimeError("FoundryLocalManager is a singleton and has already been initialized.")

    monkeypatch.setitem(
        sys.modules,
        "foundry_local_sdk",
        SimpleNamespace(Configuration=Configuration, FoundryLocalManager=FakeFoundryLocalManager),
    )
    monkeypatch.setitem(sys.modules, "foundry_local", None)

    first = FoundryLocalProvider(model="model-a").complete([{"role": "user", "content": "hello"}])
    second = FoundryLocalProvider(model="model-b").complete([{"role": "user", "content": "hello"}])

    assert first.text == "answer from model-a"
    assert second.text == "answer from model-b"


def test_foundry_local_preserves_legacy_foundry_local_fallback(monkeypatch):
    class LegacyChatClient:
        def __init__(self):
            self.calls = []

        def complete(self, **kwargs):
            self.calls.append(kwargs)
            return SimpleNamespace(text=f"legacy answer from {kwargs['model']}")

    class LegacyManager:
        instances = []

        def __init__(self, *args):
            self.args = args
            self.downloaded = []
            self.loaded = []
            self.client = LegacyChatClient()
            LegacyManager.instances.append(self)

        def download_model(self, model):
            self.downloaded.append(model)

        def load_model(self, model):
            self.loaded.append(model)

        def get_chat_client(self, *args):
            self.chat_client_args = args
            return self.client

    original_import_module = importlib.import_module

    def fake_import_module(name, package=None):
        if name == "foundry_local_sdk":
            raise ImportError("not installed")
        if name == "foundry_local":
            return SimpleNamespace(FoundryLocalManager=LegacyManager)
        return original_import_module(name, package)

    monkeypatch.setattr(foundry_local_module.importlib, "import_module", fake_import_module)

    provider = FoundryLocalProvider(model="legacy-local")
    response = provider.complete([{"role": "user", "content": "hello"}], top_p=0.9)

    manager = LegacyManager.instances[0]
    assert response.text == "legacy answer from legacy-local"
    assert manager.args == ("legacy-local",)
    assert manager.downloaded == ["legacy-local"]
    assert manager.loaded == ["legacy-local"]
    assert manager.chat_client_args == ("legacy-local",)
    assert manager.client.calls[0]["top_p"] == 0.9


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
