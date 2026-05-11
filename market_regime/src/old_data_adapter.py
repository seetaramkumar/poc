"""
data_adapter.py — Bridge between trading_data and MarketRegimeEngine
=====================================================================
This is the ONLY module in the regime engine that knows about
trading_data.  Everything else in src/ stays isolated from the
data sourcing layer.

Why a separate adapter?
-----------------------
* Keeps the regime engine testable without a live network connection.
* If the data provider changes (Yahoo → Zerodha), only this file changes.
* Validates the DataFrame contract before handing it to the engine,
  giving a clear error message instead of a cryptic downstream failure.

Typical usage
-------------
    from src.data_adapter import DataAdapter

    adapter = DataAdapter()
    df = adapter.fetch("NIFTY50", start="2020-01-01", end="2024-12-31")

    from src import MarketRegimeEngine
    engine = MarketRegimeEngine()
    result = engine.analyze(df)
"""

from __future__ import annotations

import logging
from datetime import date, timedelta
from pathlib import Path
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)

# Minimum bars needed for EMA-200 + ADX/ATR warmup
_MIN_BARS_REQUIRED = 220


class DataAdapter:
    """
    Wraps ``trading_data.DataManager`` and returns OHLCV DataFrames
    that are guaranteed compatible with ``MarketRegimeEngine``.

    Parameters
    ----------
    cache_dir :
        Directory for the local parquet cache.
        Defaults to ``.cache/ohlcv`` relative to the project root.
    cache_max_age_days :
        How many days before a cached file is considered stale.
    """

    def __init__(
        self,
        cache_dir: str = ".cache/ohlcv",
        cache_max_age_days: int = 1,
    ) -> None:
        # Import here so the rest of the engine never has a hard
        # dependency on trading_data at module-load time.
        try:
            from trading_data import DataManager, DataManagerConfig
            self._manager = DataManager(
                config=DataManagerConfig(
                    cache_enabled=True,
                    cache_dir=cache_dir,
                    cache_max_age_days=cache_max_age_days,
                    retry_attempts=3,
                    retry_backoff_seconds=1.0,
                )
            )
            logger.info("DataAdapter initialised with DataManager (cache=%s).", cache_dir)
        except ImportError as exc:
            raise ImportError(
                "trading_data package not found. "
                "Make sure the 'trading_data/' folder is present in the project root."
            ) from exc

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fetch(
        self,
        symbol: str,
        start: str,
        end: str,
        refresh: bool = False,
    ) -> pd.DataFrame:
        """
        Fetch daily OHLCV data and validate it for use with the regime engine.

        Parameters
        ----------
        symbol :
            Canonical name (``"NIFTY50"``, ``"SP500"``) or raw provider
            ticker (``"^NSEI"``, ``"AAPL"``).
        start :
            Inclusive start date as ``"YYYY-MM-DD"``.
        end :
            Inclusive end date as ``"YYYY-MM-DD"``.
        refresh :
            Force a live fetch even if a valid cache entry exists.

        Returns
        -------
        pd.DataFrame
            Columns : open, high, low, close, volume
            Index   : DatetimeIndex named "date", tz-naive, ascending.

        Raises
        ------
        ValueError
            If the returned DataFrame is empty or has too few bars.
        RuntimeError
            If the data provider returns an incompatible schema.
        """
        logger.info("Fetching %s  %s → %s", symbol, start, end)

        df = self._manager.get_daily_data(
            symbol=symbol,
            start=start,
            end=end,
            refresh=refresh,
        )

        self._validate(df, symbol)
        logger.info("Fetched %d bars for '%s'.", len(df), symbol)
        return df

    def fetch_latest(self, symbol: str) -> pd.DataFrame:
        """
        Return the most recent available trading bar for *symbol*.

        Useful for live / end-of-day classification without specifying
        a full date range.

        Parameters
        ----------
        symbol :
            Canonical name or raw ticker.

        Returns
        -------
        pd.DataFrame
            Single-row OHLCV frame.
        """
        return self._manager.fetch_latest(symbol)

    def list_symbols(self) -> list[str]:
        """Return all canonical symbols registered in the data layer."""
        return self._manager.list_symbols()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _validate(df: pd.DataFrame, symbol: str) -> None:
        """
        Assert that df meets the contract expected by MarketRegimeEngine.

        Raises ValueError with a helpful message on any violation.
        """
        if df is None or df.empty:
            raise ValueError(f"No data returned for symbol '{symbol}'.")

        required = {"open", "high", "low", "close", "volume"}
        missing = required - set(df.columns)
        if missing:
            raise RuntimeError(
                f"DataFrame for '{symbol}' is missing columns: {missing}. "
                "Check the data provider's normalise() output."
            )

        if not isinstance(df.index, pd.DatetimeIndex):
            raise RuntimeError(
                f"DataFrame for '{symbol}' must have a DatetimeIndex. "
                f"Got: {type(df.index).__name__}"
            )

        if len(df) < _MIN_BARS_REQUIRED:
            logger.warning(
                "'%s' has only %d bars (minimum recommended: %d). "
                "Early bars will return UNCERTAIN due to EMA-200 warmup.",
                symbol, len(df), _MIN_BARS_REQUIRED,
            )
