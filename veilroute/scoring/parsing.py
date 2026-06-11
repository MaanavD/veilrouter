from __future__ import annotations

import json
import re
from typing import Any

from veilroute.errors import ScoreParseError

_SCORE_RE = re.compile(r"\b([0-5])\b")


def parse_score(output: str, *, default: int | None = 2) -> int:
    """Parse a 0-5 difficulty score, defaulting cloud-safe on malformed output."""
    text = output.strip()
    if not text:
        if default is None:
            raise ScoreParseError("empty score output")
        return _clamp(default)

    try:
        parsed: Any = json.loads(text)
    except json.JSONDecodeError:
        parsed = None

    if isinstance(parsed, dict) and "score" in parsed:
        try:
            return _clamp(int(parsed["score"]))
        except (TypeError, ValueError):
            pass
    elif isinstance(parsed, int):
        return _clamp(parsed)

    match = _SCORE_RE.search(text)
    if match:
        return _clamp(int(match.group(1)))
    if default is None:
        raise ScoreParseError(f"could not parse score from {output!r}")
    return _clamp(default)


def _clamp(value: int) -> int:
    return max(0, min(5, int(value)))
