from __future__ import annotations

import asyncio
from typing import Any, AsyncIterator, Iterator

from veilroute.errors import ProviderCallError, ProviderSetupError
from veilroute.providers.base import ChatChunk, ChatResponse, Message


class FoundryLocalProvider:
    def __init__(self, *, model: str) -> None:
        self.model = model
        self._manager: Any | None = None
        self._client: Any | None = None

    def _ensure_client(self) -> Any:
        if self._client is not None:
            return self._client
        try:
            from foundry_local import FoundryLocalManager
        except ImportError as exc:
            raise ProviderSetupError(
                "Foundry Local SDK is required for local routing. Install veilroute[foundry]."
            ) from exc

        errors: list[Exception] = []
        for args in ((self.model,), ()):
            try:
                self._manager = FoundryLocalManager(*args)
                break
            except Exception as exc:
                errors.append(exc)
        if self._manager is None:
            raise ProviderSetupError(f"could not initialize FoundryLocalManager: {errors[-1]}") from errors[-1]

        try:
            self._load_model_if_supported()
            self._client = self._get_chat_client()
            return self._client
        except Exception as exc:
            raise ProviderSetupError(f"could not initialize Foundry Local chat client: {exc}") from exc

    def _load_model_if_supported(self) -> None:
        manager = self._manager
        catalog = getattr(manager, "catalog", None)
        model_info = None
        if catalog is not None and hasattr(catalog, "get_model"):
            model_info = catalog.get_model(self.model)
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
                return ChatResponse(
                    text=response.choices[0].message.content or "",
                    model=getattr(response, "model", self.model),
                    tokens_in=getattr(usage, "prompt_tokens", None),
                    tokens_out=getattr(usage, "completion_tokens", None),
                    raw=response,
                )
            if hasattr(client, "complete"):
                response = client.complete(messages=messages, model=self.model, **opts)
                text = getattr(response, "text", response if isinstance(response, str) else "")
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
