from pathlib import Path

import pytest

from veilroute.config import DEFAULT_SCORER_MAX_TOKENS, DEFAULT_SCORER_MODEL, ModelSpec, RouterConfig
from veilroute.errors import (
    ConfigurationError,
    LocalContextExceededError,
    ProviderCallError,
    ProviderSetupError,
    ScoreParseError,
    VeilrouteError,
)


def test_router_config_reads_environment_clamps_score_and_redacts_secret(monkeypatch):
    monkeypatch.setenv("VEILROUTE_CLOUD_API_KEY", "secret-key")
    monkeypatch.setenv("VEILROUTE_CLOUD_ENDPOINT", "https://example.invalid/v1")

    config = RouterConfig(local_model="local-model", local_score_max=99, pii_model_path="models/pii")

    assert config.scorer_model == DEFAULT_SCORER_MODEL
    assert config.scorer_max_tokens == DEFAULT_SCORER_MAX_TOKENS
    assert config.cloud_api_key == "secret-key"
    assert config.cloud_endpoint == "https://example.invalid/v1"
    assert config.local_score_max == 5
    assert isinstance(config.pii_model_path, Path)
    assert "secret-key" not in repr(config)
    assert "cloud_api_key='***'" in repr(config)


def test_router_config_can_still_use_local_model_as_scorer_when_explicitly_requested(monkeypatch):
    monkeypatch.delenv("VEILROUTE_CLOUD_API_KEY", raising=False)
    monkeypatch.delenv("VEILROUTE_CLOUD_ENDPOINT", raising=False)

    config = RouterConfig(local_model="local-model", scorer_model=None)

    assert config.scorer_model == "local-model"


@pytest.mark.parametrize(
    ("value", "expected"),
    [(-1, 0), (0, 0), (3, 3), (6, 5), ("4", 4)],
)
def test_router_config_normalizes_local_score_bounds(monkeypatch, value, expected):
    monkeypatch.delenv("VEILROUTE_CLOUD_API_KEY", raising=False)
    monkeypatch.delenv("VEILROUTE_CLOUD_ENDPOINT", raising=False)

    config = RouterConfig(local_score_max=value)

    assert config.local_score_max == expected


def test_model_spec_preserves_public_fields():
    spec = ModelSpec(name="tiny-local", max_context_tokens=2048)

    assert spec.name == "tiny-local"
    assert spec.max_context_tokens == 2048


@pytest.mark.parametrize(
    "error_type",
    [ConfigurationError, ProviderSetupError, ProviderCallError, ScoreParseError, LocalContextExceededError],
)
def test_package_errors_share_common_base_type(error_type):
    assert issubclass(error_type, VeilrouteError)
