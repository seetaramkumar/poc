"""
trading_data/exceptions.py
--------------------------
Custom exception hierarchy for the data sourcing layer.
All exceptions inherit from DataLayerError for easy top-level catching.
"""


class DataLayerError(Exception):
    """Base exception for all data sourcing errors."""


class ProviderError(DataLayerError):
    """Raised when a provider fails to fetch data."""

    def __init__(self, provider: str, symbol: str, reason: str):
        self.provider = provider
        self.symbol = symbol
        self.reason = reason
        super().__init__(f"[{provider}] Failed to fetch '{symbol}': {reason}")


class SymbolNotFoundError(DataLayerError):
    """Raised when a symbol cannot be resolved or returns no data."""

    def __init__(self, symbol: str, provider: str = ""):
        prefix = f"[{provider}] " if provider else ""
        super().__init__(f"{prefix}Symbol '{symbol}' returned no data.")
        self.symbol = symbol
        self.provider = provider


class CacheError(DataLayerError):
    """Raised when reading from or writing to the local cache fails."""


class ConfigurationError(DataLayerError):
    """Raised when a provider is misconfigured (missing credentials, etc.)."""

    def __init__(self, provider: str, detail: str):
        super().__init__(f"[{provider}] Configuration error: {detail}")
        self.provider = provider
