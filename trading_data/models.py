"""
trading_data/models.py
----------------------
Shared data models, type aliases, and constants used across the data layer.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime
from enum import Enum
from typing import Optional

import pandas as pd


# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------
DateLike = str | date | datetime          # accepted date input forms
OHLCVFrame = pd.DataFrame                 # DataFrame with normalised columns


# ---------------------------------------------------------------------------
# Canonical column schema
# ---------------------------------------------------------------------------
OHLCV_COLUMNS: list[str] = ["open", "high", "low", "close", "volume"]

# Column dtype mapping applied after normalisation
OHLCV_DTYPES: dict[str, str] = {
    "open":   "float64",
    "high":   "float64",
    "low":    "float64",
    "close":  "float64",
    "volume": "int64",
}


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------
class AssetClass(str, Enum):
    """Broad asset class used for symbol routing and display."""
    INDEX  = "index"
    EQUITY = "equity"
    ETF    = "etf"
    CRYPTO = "crypto"
    FX     = "fx"


class Provider(str, Enum):
    """Supported data provider identifiers."""
    YAHOO   = "yahoo"
    ZERODHA = "zerodha"
    POLYGON = "polygon"
    IBKR    = "ibkr"


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------
@dataclass
class SymbolInfo:
    """
    Canonical descriptor for a tradeable symbol.

    Attributes
    ----------
    canonical :
        Internally consistent identifier (e.g. ``"NIFTY50"``).
    provider_symbol :
        Exchange-specific ticker used by the active provider (e.g. ``"^NSEI"``).
    asset_class :
        Broad classification of the instrument.
    exchange :
        Primary exchange (e.g. ``"NSE"``, ``"NASDAQ"``).
    currency :
        Settlement currency (e.g. ``"INR"``, ``"USD"``).
    description :
        Human-readable name (e.g. ``"Nifty 50 Index"``).
    """
    canonical:       str
    provider_symbol: str
    asset_class:     AssetClass = AssetClass.EQUITY
    exchange:        str        = ""
    currency:        str        = "USD"
    description:     str        = ""


@dataclass
class FetchResult:
    """
    Return type from a single provider fetch.

    Attributes
    ----------
    symbol :
        The symbol that was fetched.
    provider :
        Name of the provider that returned the data.
    data :
        Normalised OHLCV DataFrame (empty if fetch failed).
    success :
        ``True`` if data was retrieved without error.
    error :
        Error message when ``success`` is ``False``.
    from_cache :
        ``True`` if data came from the local parquet cache.
    rows :
        Number of rows returned (convenience accessor).
    """
    symbol:     str
    provider:   str
    data:       OHLCVFrame = field(default_factory=pd.DataFrame)
    success:    bool       = True
    error:      str        = ""
    from_cache: bool       = False

    @property
    def rows(self) -> int:
        return len(self.data)

    def __repr__(self) -> str:
        status = "OK" if self.success else f"ERR({self.error})"
        source = "cache" if self.from_cache else "live"
        return (
            f"FetchResult(symbol={self.symbol!r}, provider={self.provider!r}, "
            f"rows={self.rows}, source={source}, status={status})"
        )


@dataclass
class DataManagerConfig:
    """
    Top-level configuration for :class:`~trading_data.manager.DataManager`.

    Attributes
    ----------
    default_provider :
        Which provider to use when none is specified at call time.
    cache_enabled :
        Toggle the local parquet cache on/off.
    cache_dir :
        Directory where parquet files are stored.
    cache_max_age_days :
        Cached files older than this are considered stale.
    retry_attempts :
        Number of provider call attempts before raising.
    retry_backoff_seconds :
        Initial wait time between retries (doubles each attempt).
    """
    default_provider:      Provider = Provider.YAHOO
    cache_enabled:         bool     = True
    cache_dir:             str      = ".cache/ohlcv"
    cache_max_age_days:    int      = 1
    retry_attempts:        int      = 3
    retry_backoff_seconds: float    = 1.0
