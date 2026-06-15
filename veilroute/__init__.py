"""Public API for veilroute."""

from veilroute.config import DEFAULT_SCORER_MAX_TOKENS, DEFAULT_SCORER_MODEL, ModelSpec, RouterConfig
from veilroute.router import AsyncRouter, Router, RouterChunk, RouterResponse, run

__version__ = "0.1.0"

__all__ = [
    "AsyncRouter",
    "DEFAULT_SCORER_MAX_TOKENS",
    "DEFAULT_SCORER_MODEL",
    "ModelSpec",
    "Router",
    "RouterChunk",
    "RouterConfig",
    "RouterResponse",
    "__version__",
    "run",
]
