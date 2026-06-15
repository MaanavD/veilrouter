from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from threading import RLock
from typing import Any


@dataclass(frozen=True, slots=True)
class TelemetryRecord:
    timestamp: datetime
    route: str
    score: int | None
    model: str | None
    tokens_in: int
    tokens_out: int
    latency_ms: float
    pii_detected: bool
    redaction_count: int
    redaction_categories: dict[str, int] = field(default_factory=dict)
    cost_estimate: float = 0.0
    cost_saved: float = 0.0


class InMemoryTelemetryRecorder:
    def __init__(self) -> None:
        self.records: list[TelemetryRecord] = []
        self._lock = RLock()

    def record(self, record: TelemetryRecord) -> None:
        with self._lock:
            self.records.append(record)

    def report(self) -> dict[str, Any]:
        with self._lock:
            records = list(self.records)
        total = len(records)
        local = sum(1 for record in records if record.route == "local")
        return {
            "total_calls": total,
            "local_calls": local,
            "local_rate": (local / total) if total else 0.0,
            "total_cost_saved": sum(record.cost_saved for record in records),
            "total_redactions": sum(record.redaction_count for record in records),
        }


def now_utc() -> datetime:
    return datetime.now(timezone.utc)
