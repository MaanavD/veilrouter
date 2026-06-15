from __future__ import annotations

import asyncio
import importlib
import threading
from typing import Any, AsyncIterator, Iterator

from veilrouter.errors import ProviderCallError, ProviderSetupError
from veilrouter.providers.base import ChatChunk, ChatResponse, Message


class FoundryLocalProvider:
    def __init__(self, *, model: str) -> None:
        self.model = model
        self._manager: Any | None = None
        self._model_handle: Any | None = None
        self._client: Any | None = None
        self._sdk_name: str | None = None
        self._client_lock = threading.Lock()

    def _ensure_client(self) -> Any:
        if self._client is not None:
            return self._client
        with self._client_lock:
            if self._client is not None:
                return self._client
            try:
                self._manager = self._initialize_manager()
                self._load_model_if_supported()
                self._client = self._get_chat_client()
                return self._client
            except ProviderSetupError:
                raise
            except Exception as exc:
                raise ProviderSetupError(f"could not initialize Foundry Local chat client: {exc}") from exc

    def _initialize_manager(self) -> Any:
        try:
            return self._initialize_foundry_local_sdk()
        except ImportError:
            pass
        try:
            return self._initialize_legacy_foundry_local()
        except ImportError as legacy_exc:
            raise ProviderSetupError(
                "Foundry Local SDK is required for local routing. Install veilrouter[foundry]."
            ) from legacy_exc

    def _initialize_foundry_local_sdk(self) -> Any:
        module = importlib.import_module("foundry_local_sdk")
        configuration = getattr(module, "Configuration", None)
        manager_cls = getattr(module, "FoundryLocalManager", None)
        if configuration is None or manager_cls is None:
            raise ProviderSetupError("foundry_local_sdk does not expose Configuration and FoundryLocalManager")

        initialize = getattr(manager_cls, "initialize", None)
        if initialize is not None:
            try:
                initialize(configuration(app_name="veilrouter"))
            except Exception as exc:
                if "singleton" not in str(exc).lower() and "already been initialized" not in str(exc).lower():
                    raise

        manager = getattr(manager_cls, "instance", None)
        if callable(manager):
            manager = manager()
        if manager is None:
            manager = manager_cls()
        self._sdk_name = "foundry_local_sdk"
        return manager

    def _initialize_legacy_foundry_local(self) -> Any:
        module = importlib.import_module("foundry_local")
        manager_cls = getattr(module, "FoundryLocalManager")
        errors: list[Exception] = []
        for args in ((self.model,), ()):
            try:
                manager = manager_cls(*args)
                self._sdk_name = "foundry_local"
                return manager
            except Exception as exc:
                errors.append(exc)
        raise ProviderSetupError(f"could not initialize FoundryLocalManager: {errors[-1]}") from errors[-1]

    def _load_model_if_supported(self) -> None:
        manager = self._manager
        catalog = getattr(manager, "catalog", None)
        model_info = None
        if catalog is not None and hasattr(catalog, "get_model"):
            model_info = catalog.get_model(self.model)
        self._model_handle = model_info

        if model_info is not None and self._sdk_name == "foundry_local_sdk":
            for name in ("download", "download_model"):
                method = getattr(model_info, name, None)
                if method is not None:
                    method()
                    break
            for name in ("load", "load_model"):
                method = getattr(model_info, name, None)
                if method is not None:
                    method()
                    break
            return

        for name in ("download_model", "download"):
            method = getattr(manager, name, None)
            if method is not None:
                method(model_info or self.model)
                break
        for name in ("load_model", "load"):
            method = getattr(manager, name, None)
            if method is not None:
                method(model_info or self.model)
                break

    def _get_chat_client(self) -> Any:
        manager = self._manager
        model_handle = self._model_handle
        if model_handle is not None:
            method = getattr(model_handle, "get_chat_client", None)
            if method is not None:
                return method()
        method = getattr(manager, "get_chat_client", None)
        if method is None:
            raise ProviderSetupError("FoundryLocalManager does not expose get_chat_client()")
        for args in ((self.model,), ()):
            try:
                return method(*args)
            except TypeError:
                continue
        return method()

    def complete(self, messages: list[Message], **opts: Any) -> ChatResponse:
        client = self._ensure_client()
        try:
            if hasattr(client, "chat") and hasattr(client.chat, "completions"):
                response = client.chat.completions.create(model=self.model, messages=messages, **opts)
                usage = getattr(response, "usage", None)
                message = response.choices[0].message
                return ChatResponse(
                    text=message.content or "",
                    model=getattr(response, "model", self.model),
                    tokens_in=getattr(usage, "prompt_tokens", None),
                    tokens_out=getattr(usage, "completion_tokens", None),
                    tool_calls=getattr(message, "tool_calls", None),
                    raw=response,
                )
            if hasattr(client, "complete_chat"):
                _apply_client_settings(client, opts)
                response = client.complete_chat(messages)
                text = _extract_response_text(response)
                return ChatResponse(text=text, model=self.model, tool_calls=_extract_tool_calls(response), raw=response)
            if hasattr(client, "complete"):
                response = client.complete(messages=messages, model=self.model, **opts)
                text = _extract_response_text(response)
                return ChatResponse(text=text, model=self.model, raw=response)
        except Exception as exc:
            raise ProviderCallError(f"Foundry Local call failed: {exc}") from exc
        raise ProviderSetupError("unsupported Foundry Local chat client shape")

    def stream(self, messages: list[Message], **opts: Any) -> Iterator[ChatChunk]:
        client = self._ensure_client()
        try:
            if hasattr(client, "chat") and hasattr(client.chat, "completions"):
                stream = client.chat.completions.create(model=self.model, messages=messages, stream=True, **opts)
                for chunk in stream:
                    usage = getattr(chunk, "usage", None)
                    delta = chunk.choices[0].delta.content if getattr(chunk, "choices", None) else None
                    yield ChatChunk(
                        text=delta or "",
                        model=getattr(chunk, "model", self.model),
                        tokens_in=getattr(usage, "prompt_tokens", None),
                        tokens_out=getattr(usage, "completion_tokens", None),
                        raw=chunk,
                    )
                return
        except Exception as exc:
            raise ProviderCallError(f"Foundry Local stream failed: {exc}") from exc
        yield ChatChunk(text=self.complete(messages, **opts).text, model=self.model)

    async def acomplete(self, messages: list[Message], **opts: Any) -> ChatResponse:
        return await asyncio.to_thread(self.complete, messages, **opts)

    async def astream(self, messages: list[Message], **opts: Any) -> AsyncIterator[ChatChunk]:
        for chunk in await asyncio.to_thread(lambda: list(self.stream(messages, **opts))):
            yield chunk


def _apply_client_settings(client: Any, opts: dict[str, Any]) -> None:
    settings = getattr(client, "settings", None)
    if settings is None:
        return
    for key, value in opts.items():
        if hasattr(settings, key):
            setattr(settings, key, value)


def _extract_response_text(response: Any) -> str:
    if isinstance(response, str):
        return response
    if isinstance(response, dict):
        choices = response.get("choices")
        if choices:
            text = _extract_response_text(choices[0])
            if text:
                return text
        for key in ("text", "content", "message", "output_text"):
            value = response.get(key)
            text = _extract_response_text(value) if value is not None and key == "message" else value
            if isinstance(text, str):
                return text
        return ""
    choices = getattr(response, "choices", None)
    if choices:
        text = _extract_response_text(choices[0])
        if text:
            return text
    for attr in ("text", "content", "output_text"):
        value = getattr(response, attr, None)
        if isinstance(value, str):
            return value
    message = getattr(response, "message", None)
    if message is not None:
        text = _extract_response_text(message)
        if text:
            return text
    delta = getattr(response, "delta", None)
    if delta is not None:
        return _extract_response_text(delta)
    return ""


def _extract_tool_calls(response: Any) -> Any:
    if isinstance(response, dict):
        tool_calls = response.get("tool_calls")
        if tool_calls is not None:
            return tool_calls
        choices = response.get("choices")
        if choices:
            return _extract_tool_calls(choices[0])
        message = response.get("message")
        if message is not None:
            return _extract_tool_calls(message)
        return None
    tool_calls = getattr(response, "tool_calls", None)
    if tool_calls is not None:
        return tool_calls
    choices = getattr(response, "choices", None)
    if choices:
        return _extract_tool_calls(choices[0])
    message = getattr(response, "message", None)
    if message is not None:
        return _extract_tool_calls(message)
    return None
