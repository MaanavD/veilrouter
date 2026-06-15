from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from veilroute.telemetry.pricing import DEFAULT_PRICING


DEFAULT_SCORER_MODEL = "phi-3.5-mini"
DEFAULT_SCORER_MAX_TOKENS = 256
TelemetrySink = Callable[[Any], None]


@dataclass(frozen=True, slots=True)
class ModelSpec:
    name: str
    max_context_tokens: int | None = None


@dataclass(slots=True)
class RouterConfig:
    local_model: str = "qwen2.5-0.5b"
    scorer_model: str | None = DEFAULT_SCORER_MODEL
    cloud_endpoint: str | None = None
    cloud_api_key: str | None = None
    cloud_model: str = "gpt-4o"
    local_score_max: int = 1
    pii_model_path: str | Path = "./LFM2.5-350M-Classifier-PII-Demo"
    pii_min_score: float = 0.0
    pii_regex_backstop: bool = True
    retry_cloud_failures_locally: bool = False
    scorer_temperature: float = 0.0
    scorer_max_tokens: int = DEFAULT_SCORER_MAX_TOKENS
    pricing: dict[str, dict[str, float]] = field(default_factory=lambda: dict(DEFAULT_PRICING))
    telemetry_sink: TelemetrySink | None = None
    debug: bool = False
    max_local_input_tokens: int | None = None
    route_long_inputs_to_cloud: bool = True

    def __post_init__(self) -> None:
        if self.scorer_model is None:
            self.scorer_model = self.local_model
        if self.cloud_api_key is None:
            self.cloud_api_key = os.getenv("VEILROUTE_CLOUD_API_KEY")
        if self.cloud_endpoint is None:
            self.cloud_endpoint = os.getenv("VEILROUTE_CLOUD_ENDPOINT")
        self.local_score_max = max(0, min(5, int(self.local_score_max)))
        self.pii_model_path = Path(self.pii_model_path)

    def __repr__(self) -> str:
        values = {
            "local_model": self.local_model,
            "scorer_model": self.scorer_model,
            "cloud_endpoint": self.cloud_endpoint,
            "cloud_api_key": "***" if self.cloud_api_key else None,
            "cloud_model": self.cloud_model,
            "local_score_max": self.local_score_max,
            "pii_model_path": str(self.pii_model_path),
            "pii_min_score": self.pii_min_score,
            "pii_regex_backstop": self.pii_regex_backstop,
            "retry_cloud_failures_locally": self.retry_cloud_failures_locally,
            "debug": self.debug,
        }
        args = ", ".join(f"{key}={value!r}" for key, value in values.items())
        return f"RouterConfig({args})"
