class VeilrouteError(Exception):
    """Base exception for veilroute failures."""


class ConfigurationError(VeilrouteError):
    """Raised when router configuration is incomplete or invalid."""


class ProviderSetupError(VeilrouteError):
    """Raised when a configured provider cannot be initialized."""


class ProviderCallError(VeilrouteError):
    """Raised when a provider call fails."""


class ScoreParseError(VeilrouteError):
    """Raised when a difficulty score cannot be parsed and no default is allowed."""


class LocalContextExceededError(VeilrouteError):
    """Raised when input is too large for the configured local model context."""
