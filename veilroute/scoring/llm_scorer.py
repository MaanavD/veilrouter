from __future__ import annotations

from veilroute.providers.base import AsyncChatProvider, ChatProvider
from veilroute.scoring.parsing import parse_score

_RUBRIC = """You score prompt difficulty for model routing.
Return only JSON: {"score": n} where n is an integer 0-5.
0 trivial/greeting; 1 simple factual; 2 moderate reasoning; 3 multi-step or structured;
4 complex reasoning or long-context; 5 expert, high-stakes, or agentic."""


class LlmDifficultyScorer:
    def __init__(
        self,
        provider: ChatProvider | AsyncChatProvider,
        *,
        model: str,
        temperature: float = 0.0,
        max_tokens: int = 8,
    ) -> None:
        self.provider = provider
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

    def score(self, text: str) -> int:
        response = self.provider.complete(
            [
                {"role": "system", "content": _RUBRIC},
                {"role": "user", "content": text},
            ],
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        return parse_score(response.text, default=2)

    async def ascore(self, text: str) -> int:
        if not hasattr(self.provider, "acomplete"):
            return self.score(text)
        response = await self.provider.acomplete(
            [
                {"role": "system", "content": _RUBRIC},
                {"role": "user", "content": text},
            ],
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        return parse_score(response.text, default=2)
