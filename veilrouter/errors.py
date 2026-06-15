class VeilrouterError(Exception):
    """Base exception for veilrouter failures."""


class ConfigurationError(VeilrouterError):
    """Raised when router configuration is incomplete or invalid."""


class ProviderSetupError(VeilrouterError):
    """Raised when a configured provider cannot be initialized."""


class ProviderCallError(VeilrouterError):
    """Raised when a provider call fails."""


class ScoreParseError(VeilrouterError):
    """Raised when a difficulty score cannot be parsed and no default is allowed."""


class LocalContextExceededError(VeilrouterError):
    """Raised when input is too large for the configured local model context."""
