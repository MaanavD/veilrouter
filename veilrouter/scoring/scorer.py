from __future__ import annotations

from typing import Protocol


class DifficultyScorer(Protocol):
    def score(self, text: str) -> int:
        ...


class AsyncDifficultyScorer(Protocol):
    async def score(self, text: str) -> int:
        ...
