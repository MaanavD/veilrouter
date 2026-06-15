from __future__ import annotations

import re
from collections import defaultdict

PLACEHOLDER_RE = re.compile(r"\[(?P<label>[A-Z][A-Z0-9]*(?:_[A-Z0-9]+)*)_(?P<index>[1-9][0-9]*)\]")


def is_placeholder(text: str) -> bool:
    return PLACEHOLDER_RE.fullmatch(text) is not None


def category_label(category: str) -> str:
    normalized = category.strip().upper().replace(".", "_").replace("-", "_").replace(" ", "_")
    aliases = {
        "CONTACT_EMAIL": "EMAIL",
        "EMAIL_ADDRESS": "EMAIL",
        "IDENTITY_SSN": "SSN",
        "FINANCIAL_CREDIT_CARD": "CREDIT_CARD",
        "IDENTITY_PERSON_NAME": "PERSON_NAME",
        "CONTACT_PHONE": "PHONE",
        "PHONE_NUMBER": "PHONE",
    }
    return aliases.get(normalized, normalized or "PII")


class PlaceholderFactory:
    def __init__(self) -> None:
        self._counters: defaultdict[str, int] = defaultdict(int)

    def create(self, category: str) -> str:
        label = category_label(category)
        self._counters[label] += 1
        return f"[{label}_{self._counters[label]}]"
