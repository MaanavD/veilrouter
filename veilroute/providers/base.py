from __future__ import annotations

from dataclasses import dataclass
from typing import Any, AsyncIterator, Iterator, Protocol

Message = dict[str, Any]


@dataclass(frozen=True, slots=True)
class ChatResponse:
    text: str
    model: str | None = None
    tokens_in: int | None = None
    tokens_out: int | None = None
    raw: Any = None


@dataclass(frozen=True, slots=True)
class ChatChunk:
    text: str
    model: str | None = None
    tokens_in: int | None = None
    tokens_out: int | None = None
    raw: Any = None


class ChatProvider(Protocol):
    model: str

    def complete(self, messages: list[Message], **opts: Any) -> ChatResponse:
        ...

    def stream(self, messages: list[Message], **opts: Any) -> Iterator[ChatChunk]:
        ...


class AsyncChatProvider(Protocol):
    model: str

    async def acomplete(self, messages: list[Message], **opts: Any) -> ChatResponse:
        ...

    def astream(self, messages: list[Message], **opts: Any) -> AsyncIterator[ChatChunk]:
        ...
