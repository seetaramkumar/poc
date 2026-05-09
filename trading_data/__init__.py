"""
trading_data
============
Modular data sourcing layer for algorithmic trading.

Quick start
-----------
>>> from trading_data import DataManager
>>> manager = DataManager()
>>> df = manager.get_daily_data("NIFTY50", start="2020-01-01", end="2025-01-01")

Public surface
--------------
:class:`DataManager`             — primary orchestrator
:class:`DataManagerConfig`       — runtime configuration
:class:`BaseDataProvider`        — base class for custom providers
:class:`YahooFinanceProvider`    — active yfinance implementation
:class:`SymbolMapper`            — canonical → provider symbol translation
:class:`ParquetCache`            — local parquet cache
:mod:`trading_data.models`       — shared types (FetchResult, SymbolInfo …)
:mod:`trading_data.exceptions`   — exception hierarchy
"""

from trading_data.exceptions import (
    CacheError,
    ConfigurationError,
    DataLayerError,
    ProviderError,
    SymbolNotFoundError,
)
from trading_data.manager import DataManager
from trading_data.models import (
    AssetClass,
    DataManagerConfig,
    FetchResult,
    OHLCVFrame,
    Provider,
    SymbolInfo,
)
from trading_data.providers.base import BaseDataProvider
from trading_data.providers.yahoo import YahooFinanceProvider
from trading_data.symbols.mapper import SymbolMapper
from trading_data.cache.parquet_cache import ParquetCache

__all__ = [
    # Manager
    "DataManager",
    "DataManagerConfig",
    # Providers
    "BaseDataProvider",
    "YahooFinanceProvider",
    # Symbol mapping
    "SymbolMapper",
    # Cache
    "ParquetCache",
    # Models
    "AssetClass",
    "FetchResult",
    "OHLCVFrame",
    "Provider",
    "SymbolInfo",
    # Exceptions
    "DataLayerError",
    "ProviderError",
    "SymbolNotFoundError",
    "CacheError",
    "ConfigurationError",
]

__version__ = "1.0.0"
__author__  = "Algorithmic Trading Platform"
