from __future__ import annotations

DEFAULT_PRICING: dict[str, dict[str, float]] = {
    "gpt-4o": {"input": 0.005, "output": 0.015},
    "gpt-4o-mini": {"input": 0.00015, "output": 0.0006},
}


def estimate_cost(
    model: str,
    tokens_in: int | None,
    tokens_out: int | None,
    pricing: dict[str, dict[str, float]] | None = None,
) -> float:
    table = pricing or DEFAULT_PRICING
    rates = table.get(model)
    if rates is None:
        return 0.0
    return ((tokens_in or 0) / 1000.0 * rates.get("input", 0.0)) + (
        (tokens_out or 0) / 1000.0 * rates.get("output", 0.0)
    )
