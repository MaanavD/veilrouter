from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any

from veilroute.pii.detector import RegexPiiDetector, Span
from veilroute.pii.placeholders import PlaceholderFactory, category_label, is_placeholder


@dataclass(slots=True)
class RedactionResult:
    messages: list[dict[str, Any]]
    placeholder_to_original: dict[str, str]
    original_to_placeholder: dict[str, str]
    redaction_count: int
    categories: dict[str, int] = field(default_factory=dict)


class Redactor:
    def __init__(self, detector: Any | None = None, *, regex_backstop: bool = True) -> None:
        self.detector = detector
        self.regex_backstop = regex_backstop
        self.regex_detector = RegexPiiDetector()

    def redact_messages(self, messages: list[dict[str, Any]]) -> RedactionResult:
        copied = copy.deepcopy(messages)
        factory = PlaceholderFactory()
        placeholder_to_original: dict[str, str] = {}
        original_to_placeholder: dict[str, str] = {}
        categories: dict[str, int] = {}

        for message in copied:
            self._redact_value(message, factory, placeholder_to_original, original_to_placeholder, categories)

        return RedactionResult(
            messages=copied,
            placeholder_to_original=placeholder_to_original,
            original_to_placeholder=original_to_placeholder,
            redaction_count=len(original_to_placeholder),
            categories=categories,
        )

    def _redact_value(
        self,
        value: Any,
        factory: PlaceholderFactory,
        placeholder_to_original: dict[str, str],
        original_to_placeholder: dict[str, str],
        categories: dict[str, int],
    ) -> Any:
        if isinstance(value, dict):
            for key, child in list(value.items()):
                value[key] = self._redact_value(child, factory, placeholder_to_original, original_to_placeholder, categories)
            return value
        if isinstance(value, list):
            for idx, child in enumerate(value):
                value[idx] = self._redact_value(child, factory, placeholder_to_original, original_to_placeholder, categories)
            return value
        if isinstance(value, str):
            return self._redact_text(value, factory, placeholder_to_original, original_to_placeholder, categories)
        return value

    def _redact_text(
        self,
        text: str,
        factory: PlaceholderFactory,
        placeholder_to_original: dict[str, str],
        original_to_placeholder: dict[str, str],
        categories: dict[str, int],
    ) -> str:
        spans = self._detect(text)
        if not spans:
            return text
        assigned: list[tuple[Span, str]] = []
        for span in spans:
            original = text[span.start : span.end]
            if not original or is_placeholder(original):
                continue
            placeholder = original_to_placeholder.get(original)
            if placeholder is None:
                placeholder = factory.create(span.category)
                original_to_placeholder[original] = placeholder
                placeholder_to_original[placeholder] = original
                label = category_label(span.category)
                categories[label] = categories.get(label, 0) + 1
            assigned.append((span, placeholder))
        redacted = text
        for span, placeholder in sorted(assigned, key=lambda item: item[0].start, reverse=True):
            redacted = redacted[: span.start] + placeholder + redacted[span.end :]
        return redacted

    def _detect(self, text: str) -> list[Span]:
        spans: list[Span] = []
        if self.detector is not None:
            spans.extend(self.detector.detect(text))
        if self.regex_backstop:
            spans.extend(self.regex_detector.detect(text))
        return _merge_spans(spans)


def _merge_spans(spans: list[Span]) -> list[Span]:
    ordered = sorted(spans, key=lambda s: (s.start, -(s.end - s.start), -s.score))
    merged: list[Span] = []
    for span in ordered:
        if span.end <= span.start:
            continue
        if any(not (span.end <= kept.start or span.start >= kept.end) for kept in merged):
            continue
        merged.append(span)
    return sorted(merged, key=lambda s: (s.start, s.end))
