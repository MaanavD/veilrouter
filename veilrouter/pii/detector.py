from __future__ import annotations

import re
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np


@dataclass(frozen=True, slots=True)
class Span:
    category: str
    start: int
    end: int
    text: str
    score: float = 1.0


class RegexPiiDetector:
    EMAIL_RE = re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.IGNORECASE)
    SSN_RE = re.compile(r"\b\d{3}-\d{2}-\d{4}\b")
    IBAN_RE = re.compile(r"\b[A-Z]{2}\d{2}[A-Z0-9]{11,30}\b", re.IGNORECASE)
    PHONE_RE = re.compile(r"(?<!\w)(?:\+?\d[\d .()\-]{7,}\d)(?!\w)")
    CARD_RE = re.compile(r"(?<!\d)(?:\d[ -]?){13,19}(?!\d)")
    PERSON_NAME_RE = re.compile(
        r"\b(?:to|for|from|with|customer|client|patient|employee|contact|recipient|user)\s+"
        r"(?P<name>[A-Z][a-z]+(?:[-'][A-Z][a-z]+)?(?:\s+[A-Z][a-z]+(?:[-'][A-Z][a-z]+)?){1,3})"
        r"(?=\s+(?:at|via|on|about|and|with|cannot|can|is|was|will)\b|[,.;]|$)"
    )
    PERSON_NAME_BEFORE_EMAIL_RE = re.compile(
        r"\b(?P<name>[A-Z][a-z]+(?:[-'][A-Z][a-z]+)?(?:\s+[A-Z][a-z]+(?:[-'][A-Z][a-z]+)?){1,3})"
        r"\s+at\s+[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b",
    )

    def detect(self, text: str) -> list[Span]:
        spans: list[Span] = []
        spans.extend(_spans_from_regex("contact.email", self.EMAIL_RE, text))
        spans.extend(_spans_from_regex("identity.ssn", self.SSN_RE, text))
        spans.extend(_spans_from_regex("financial.iban", self.IBAN_RE, text))
        for match in self.CARD_RE.finditer(text):
            candidate = match.group(0)
            digits = re.sub(r"\D", "", candidate)
            if 13 <= len(digits) <= 19 and _luhn_valid(digits):
                spans.append(Span("financial.credit_card", match.start(), match.end(), candidate, 1.0))
        for match in self.PHONE_RE.finditer(text):
            candidate = match.group(0)
            digits = re.sub(r"\D", "", candidate)
            if 13 <= len(digits) <= 19:
                continue
            spans.append(Span("contact.phone", match.start(), match.end(), candidate, 1.0))
        spans.extend(_person_name_spans(self.PERSON_NAME_RE, text))
        spans.extend(_person_name_spans(self.PERSON_NAME_BEFORE_EMAIL_RE, text))
        return _dedupe_spans(spans)


class PiiDetector:
    """Reusable ONNX + Viterbi PII detector for the LFM2.5 classifier head."""

    def __init__(
        self,
        model_path: str | Path,
        *,
        providers: list[str] | None = None,
        min_score: float = 0.0,
        max_chars: int = 2000,
        overlap_chars: int = 200,
    ) -> None:
        self.repo_dir, self.model_file = _resolve_model_paths(Path(model_path))
        self.providers = providers or ["CPUExecutionProvider"]
        self.min_score = min_score
        self.max_chars = max_chars
        self.overlap_chars = overlap_chars
        self._lock = threading.Lock()
        self._session: Any | None = None
        self._tokenizer: Any | None = None
        self._id2label: tuple[str, ...] | None = None
        self._transition: np.ndarray | None = None
        self._start_mask: np.ndarray | None = None
        self._end_mask: np.ndarray | None = None

    def detect(self, text: str) -> list[Span]:
        if not text:
            return []
        spans: list[Span] = []
        for base, chunk in _iter_chunks(text, self.max_chars, self.overlap_chars):
            for span in self._detect_chunk(chunk):
                rebased = Span(span.category, span.start + base, span.end + base, text[span.start + base : span.end + base], span.score)
                spans.append(rebased)
        return _dedupe_spans(spans)

    def _ensure_loaded(self) -> None:
        if self._session is not None:
            return
        with self._lock:
            if self._session is not None:
                return
            try:
                import onnxruntime as ort
                from tokenizers import Tokenizer
            except ImportError as exc:
                raise RuntimeError("onnxruntime and tokenizers are required for PiiDetector") from exc
            self._tokenizer = Tokenizer.from_file(str(self.repo_dir / "tokenizer.json"))
            self._id2label = _load_schema(self.repo_dir)
            self._transition = _build_transition_matrix(self._id2label)
            self._start_mask, self._end_mask = _start_end_masks(self._id2label)
            self._session = ort.InferenceSession(str(self.model_file), providers=self.providers)

    def _detect_chunk(self, text: str) -> list[Span]:
        self._ensure_loaded()
        tokenizer = self._tokenizer
        session = self._session
        id2label = self._id2label
        assert tokenizer is not None and session is not None and id2label is not None
        assert self._transition is not None and self._start_mask is not None and self._end_mask is not None

        encoding = tokenizer.encode(text)
        token_ids = np.asarray([encoding.ids], dtype=np.int64)
        attn_mask = np.asarray([encoding.attention_mask], dtype=np.int64)
        offsets = np.asarray(encoding.offsets, dtype=np.int64)
        feeds = _build_feeds(session, token_ids, attn_mask)
        outputs = session.run(None, feeds)
        logits = np.asarray(_select_logits(outputs, len(id2label))[0], dtype=np.float32)
        if logits.shape[0] != offsets.shape[0]:
            raise ValueError(f"Token/logit length mismatch: logits={logits.shape[0]} offsets={offsets.shape[0]}")
        path = _viterbi(logits, self._transition, self._start_mask, self._end_mask)
        spans = _path_to_spans(path, offsets, text, logits, id2label)
        return [s for s in (_tighten_span(text, span) for span in spans) if s.score >= self.min_score and s.end > s.start]


def _resolve_model_paths(path: Path) -> tuple[Path, Path]:
    path = path.expanduser().resolve()
    if path.is_dir():
        repo = path
        for candidate in (repo / "onnx" / "model_q4.onnx", repo / "onnx" / "model_fp16.onnx", repo / "onnx" / "model_fp32.onnx"):
            if candidate.exists():
                return repo, candidate
        raise FileNotFoundError(f"No ONNX model found under {repo}")
    repo = path.parent.parent if path.parent.name == "onnx" else path.parent
    return repo, path


def _spans_from_regex(category: str, regex: re.Pattern[str], text: str) -> list[Span]:
    return [Span(category, m.start(), m.end(), m.group(0), 1.0) for m in regex.finditer(text)]


def _person_name_spans(regex: re.Pattern[str], text: str) -> list[Span]:
    spans: list[Span] = []
    for match in regex.finditer(text):
        group = match.group("name")
        start, end = match.span("name")
        if _looks_like_person_name(group):
            spans.append(Span("identity.person_name", start, end, group, 1.0))
    return spans


def _looks_like_person_name(value: str) -> bool:
    parts = value.split()
    if len(parts) < 2 or len(parts) > 4:
        return False
    blocked = {"API", "HTTP", "HTTPS", "OAuth", "CPU", "JSON", "SQL", "OpenAI", "OpenRouter", "Foundry", "Local"}
    return all(part not in blocked and part[:1].isupper() for part in parts)


def _luhn_valid(digits: str) -> bool:
    total = 0
    parity = len(digits) % 2
    for idx, char in enumerate(digits):
        value = int(char)
        if idx % 2 == parity:
            value *= 2
            if value > 9:
                value -= 9
        total += value
    return total % 10 == 0


def _iter_chunks(text: str, max_chars: int, overlap: int) -> Iterable[tuple[int, str]]:
    if len(text) <= max_chars:
        yield 0, text
        return
    start = 0
    step = max(1, max_chars - overlap)
    while start < len(text):
        end = min(len(text), start + max_chars)
        yield start, text[start:end]
        if end == len(text):
            break
        start += step


def _dedupe_spans(spans: list[Span]) -> list[Span]:
    ordered = sorted(spans, key=lambda s: (s.start, -(s.end - s.start), -s.score))
    result: list[Span] = []
    for span in ordered:
        if span.end <= span.start:
            continue
        if any(not (span.end <= kept.start or span.start >= kept.end) for kept in result):
            continue
        result.append(span)
    return sorted(result, key=lambda s: (s.start, s.end))


def _tighten_span(text: str, span: Span) -> Span:
    start, end = span.start, span.end
    while start < end and text[start].isspace():
        start += 1
    while end > start and text[end - 1].isspace():
        end -= 1
    if any(key in span.category for key in ("email", "iban", "phone", "ssn", "credit_card")):
        while start > 0 and _pii_adjacent_char(text[start - 1]):
            start -= 1
        while end < len(text) and _pii_adjacent_char(text[end]):
            end += 1
    return Span(span.category, start, end, text[start:end], span.score)


def _pii_adjacent_char(char: str) -> bool:
    return char.isalnum() or char in "._%+-@() "


def _load_schema(repo_dir: Path) -> tuple[str, ...]:
    import json

    schema = json.loads((repo_dir / "label_schema.json").read_text())
    num_labels = int(schema["num_labels"])
    id2label = tuple(schema["id2label"][str(i)] for i in range(num_labels))
    if id2label[0] != "O":
        raise ValueError(f"expected label 0 to be 'O', got {id2label[0]!r}")
    return id2label


def _tag(label: str) -> str:
    return "O" if label == "O" else label[0]


def _category(label: str) -> str:
    return "" if label == "O" else label[2:]


def _build_transition_matrix(id2label: tuple[str, ...]) -> np.ndarray:
    num = len(id2label)
    trans = np.full((num, num), -1e9, dtype=np.float32)
    for prev in range(num):
        prev_tag = _tag(id2label[prev])
        prev_cat = _category(id2label[prev])
        for curr in range(num):
            curr_tag = _tag(id2label[curr])
            curr_cat = _category(id2label[curr])
            if prev_tag == "O" and (curr_tag == "O" or curr_tag in ("B", "S")):
                trans[prev, curr] = 0.0
            elif prev_tag in ("B", "I") and curr_cat == prev_cat and curr_tag in ("I", "E"):
                trans[prev, curr] = 0.0
            elif prev_tag in ("E", "S") and (curr_tag == "O" or curr_tag in ("B", "S")):
                trans[prev, curr] = 0.0
    return trans


def _start_end_masks(id2label: tuple[str, ...]) -> tuple[np.ndarray, np.ndarray]:
    start = np.zeros(len(id2label), dtype=bool)
    end = np.zeros(len(id2label), dtype=bool)
    for idx, label in enumerate(id2label):
        tag = _tag(label)
        start[idx] = tag in ("O", "B", "S")
        end[idx] = tag in ("O", "E", "S")
    return start, end


def _build_feeds(session: Any, input_ids: np.ndarray, attention_mask: np.ndarray) -> dict[str, np.ndarray]:
    feeds: dict[str, np.ndarray] = {}
    for inp in session.get_inputs():
        if inp.name == "input_ids":
            feeds[inp.name] = input_ids
        elif inp.name == "attention_mask":
            feeds[inp.name] = attention_mask
        elif inp.name == "token_type_ids":
            feeds[inp.name] = np.zeros_like(input_ids, dtype=np.int64)
        else:
            raise ValueError(f"Unsupported input tensor {inp.name!r}")
    return feeds


def _select_logits(outputs: list[np.ndarray], num_labels: int) -> np.ndarray:
    for arr in outputs:
        if arr.ndim == 3 and arr.shape[-1] == num_labels:
            return arr
    for arr in outputs:
        if arr.ndim == 3:
            return arr
    raise ValueError("Could not find a 3D logits tensor in ONNX outputs.")


def _viterbi(logits: np.ndarray, transition: np.ndarray, start_mask: np.ndarray, end_mask: np.ndarray) -> list[int]:
    length, num_labels = logits.shape
    if length == 0:
        return []
    dp = np.full((length, num_labels), -1e9, dtype=np.float32)
    back = np.zeros((length, num_labels), dtype=np.int32)
    dp[0] = np.where(start_mask, logits[0], -1e9)
    for t in range(1, length):
        scores = dp[t - 1][:, None] + transition
        best_prev = np.argmax(scores, axis=0)
        dp[t] = scores[best_prev, np.arange(num_labels)] + logits[t]
        back[t] = best_prev
    final = np.where(end_mask, dp[length - 1], -1e9)
    last = int(np.argmax(final))
    path = [0] * length
    path[-1] = last
    for t in range(length - 1, 0, -1):
        path[t - 1] = int(back[t, path[t]])
    return path


def _path_to_spans(path: list[int], offsets: np.ndarray, text: str, logits: np.ndarray, id2label: tuple[str, ...]) -> list[Span]:
    spans: list[Span] = []
    cur_cat = ""
    cur_start = -1
    cur_tokens: list[int] = []

    def emit(end_char: int) -> None:
        nonlocal cur_cat, cur_start, cur_tokens
        if cur_start < 0 or end_char <= cur_start:
            cur_cat, cur_start, cur_tokens = "", -1, []
            return
        probs = [_softmax_prob(logits[idx], path[idx]) for idx in cur_tokens]
        score = float(sum(probs) / len(probs)) if probs else 1.0
        spans.append(Span(cur_cat, cur_start, end_char, text[cur_start:end_char], score))
        cur_cat, cur_start, cur_tokens = "", -1, []

    for idx, label_idx in enumerate(path):
        tag = _tag(id2label[label_idx])
        cat = _category(id2label[label_idx])
        start, end = int(offsets[idx][0]), int(offsets[idx][1])
        if end <= start:
            if cur_start >= 0 and cur_tokens:
                emit(int(offsets[cur_tokens[-1]][1]))
            continue
        if tag == "O":
            if cur_start >= 0:
                emit(start)
        elif tag == "S":
            if cur_start >= 0:
                emit(start)
            cur_cat, cur_start, cur_tokens = cat, start, [idx]
            emit(end)
        elif tag == "B":
            if cur_start >= 0:
                emit(start)
            cur_cat, cur_start, cur_tokens = cat, start, [idx]
        elif tag == "I" and cur_start >= 0 and cat == cur_cat:
            cur_tokens.append(idx)
        elif tag == "E" and cur_start >= 0 and cat == cur_cat:
            cur_tokens.append(idx)
            emit(end)
    if cur_start >= 0 and cur_tokens:
        emit(int(offsets[cur_tokens[-1]][1]))
    return spans


def _softmax_prob(row: np.ndarray, chosen: int) -> float:
    max_value = float(np.max(row))
    exp = np.exp(row - max_value)
    total = float(np.sum(exp))
    return float(exp[chosen] / total) if total > 0 else 1.0
