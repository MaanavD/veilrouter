from __future__ import annotations

from typing import Any, AsyncIterator, Iterator

from veilroute.errors import ConfigurationError, ProviderCallError, ProviderSetupError
from veilroute.providers.base import ChatChunk, ChatResponse, Message


class OpenAICompatibleProvider:
    def __init__(self, *, model: str, base_url: str | None, api_key: str | None) -> None:
        self.model = model
        self.base_url = base_url
        self.api_key = api_key
        self._client: Any | None = None
        self._async_client: Any | None = None

    def _sync_client(self) -> Any:
        if not self.api_key:
            raise ConfigurationError("cloud_api_key is required for cloud routing")
        if self._client is None:
            try:
                from openai import OpenAI
            except ImportError as exc:
                raise ProviderSetupError("install openai to use OpenAICompatibleProvider") from exc
            self._client = OpenAI(api_key=self.api_key, base_url=self.base_url)
        return self._client

    def _aclient(self) -> Any:
        if not self.api_key:
            raise ConfigurationError("cloud_api_key is required for cloud routing")
        if self._async_client is None:
            try:
                from openai import AsyncOpenAI
            except ImportError as exc:
                raise ProviderSetupError("install openai to use OpenAICompatibleProvider") from exc
            self._async_client = AsyncOpenAI(api_key=self.api_key, base_url=self.base_url)
        return self._async_client

    def complete(self, messages: list[Message], **opts: Any) -> ChatResponse:
        try:
            response = self._sync_client().chat.completions.create(
                model=self.model,
                messages=messages,
                **opts,
            )
        except Exception as exc:  # SDK errors vary by provider implementation.
            raise ProviderCallError(f"cloud provider call failed: {exc}") from exc
        usage = getattr(response, "usage", None)
        text = response.choices[0].message.content or ""
        return ChatResponse(
            text=text,
            model=getattr(response, "model", self.model),
            tokens_in=getattr(usage, "prompt_tokens", None),
            tokens_out=getattr(usage, "completion_tokens", None),
            raw=response,
        )

    def stream(self, messages: list[Message], **opts: Any) -> Iterator[ChatChunk]:
        opts = {**opts, "stream": True}
        opts.setdefault("stream_options", {"include_usage": True})
        try:
            stream = self._sync_client().chat.completions.create(
                model=self.model,
                messages=messages,
                **opts,
            )
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
        except Exception as exc:
            raise ProviderCallError(f"cloud provider stream failed: {exc}") from exc

    async def acomplete(self, messages: list[Message], **opts: Any) -> ChatResponse:
        try:
            response = await self._aclient().chat.completions.create(
                model=self.model,
                messages=messages,
                **opts,
            )
        except Exception as exc:
            raise ProviderCallError(f"cloud provider call failed: {exc}") from exc
        usage = getattr(response, "usage", None)
        text = response.choices[0].message.content or ""
        return ChatResponse(
            text=text,
            model=getattr(response, "model", self.model),
            tokens_in=getattr(usage, "prompt_tokens", None),
            tokens_out=getattr(usage, "completion_tokens", None),
            raw=response,
        )

    async def astream(self, messages: list[Message], **opts: Any) -> AsyncIterator[ChatChunk]:
        opts = {**opts, "stream": True}
        opts.setdefault("stream_options", {"include_usage": True})
        try:
            stream = await self._aclient().chat.completions.create(
                model=self.model,
                messages=messages,
                **opts,
            )
            async for chunk in stream:
                usage = getattr(chunk, "usage", None)
                delta = chunk.choices[0].delta.content if getattr(chunk, "choices", None) else None
                yield ChatChunk(
                    text=delta or "",
                    model=getattr(chunk, "model", self.model),
                    tokens_in=getattr(usage, "prompt_tokens", None),
                    tokens_out=getattr(usage, "completion_tokens", None),
                    raw=chunk,
                )
        except Exception as exc:
            raise ProviderCallError(f"cloud provider stream failed: {exc}") from exc
