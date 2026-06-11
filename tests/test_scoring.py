import asyncio

import pytest

from veilroute.errors import ScoreParseError
from veilroute.providers.base import ChatResponse
from veilroute.scoring.llm_scorer import LlmDifficultyScorer
from veilroute.scoring.parsing import parse_score


@pytest.mark.parametrize(
    ("output", "expected"),
    [
        ("4", 4),
        ('{"score": 3}', 3),
        ('{"score": 9}', 5),
        ("difficulty score: 1 because it is factual", 1),
        ("", 2),
        ("not a score", 2),
    ],
)
def test_parse_score_accepts_common_model_outputs_and_safe_defaults(output, expected):
    assert parse_score(output) == expected


@pytest.mark.parametrize("output", ["", "not parseable", '{"score": "hard"}'])
def test_parse_score_raises_when_no_default_is_allowed(output):
    with pytest.raises(ScoreParseError):
        parse_score(output, default=None)


class RecordingProvider:
    def __init__(self, text: str) -> None:
        self.text = text
        self.calls = []

    def complete(self, messages, **opts):
        self.calls.append((messages, opts))
        return ChatResponse(text=self.text, model="scorer-model")


def test_llm_difficulty_scorer_scores_with_provider_response_and_options():
    provider = RecordingProvider('{"score": 4}')
    scorer = LlmDifficultyScorer(provider, model="scorer-model", temperature=0.2, max_tokens=3)

    score = scorer.score("Explain a migration plan")

    assert score == 4
    assert provider.calls[0][0][-1] == {"role": "user", "content": "Explain a migration plan"}
    assert provider.calls[0][1] == {"temperature": 0.2, "max_tokens": 3}


class AsyncRecordingProvider:
    def __init__(self, text: str) -> None:
        self.text = text
        self.calls = []

    async def acomplete(self, messages, **opts):
        self.calls.append((messages, opts))
        return ChatResponse(text=self.text, model="scorer-model")


def test_llm_difficulty_scorer_supports_async_providers():
    async def exercise():
        provider = AsyncRecordingProvider("5")
        scorer = LlmDifficultyScorer(provider, model="scorer-model")

        score = await scorer.ascore("Design a distributed system")

        return score, provider.calls

    score, calls = asyncio.run(exercise())

    assert score == 5
    assert calls[0][0][-1]["content"] == "Design a distributed system"
