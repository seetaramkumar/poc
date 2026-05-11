"""
trading_data/providers/base.py
------------------------------
Abstract base class that every data provider must implement.

Design contract
---------------
* ``fetch_ohlcv()``  — core method; all others are built on top.
* ``fetch_latest()`` — returns the most recent available trading day.
* ``fetch_multiple_symbols()`` — batch fetch with per-symbol error isolation.
* ``normalise()``    — utility that every provider **must** call before
  returning data so the output schema is guaranteed consistent.
* ``name`` property  — identifies the provider in logs and error messages.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from datetime import date, datetime, timedelta
from typing import Optional

import pandas as pd

from trading_data.exceptions import ProviderError, SymbolNotFoundError
from trading_data.models import (
    OHLCV_COLUMNS,
    OHLCV_DTYPES,
    DateLike,
    FetchResult,
    OHLCVFrame,
)

logger = logging.getLogger(__name__)


class BaseDataProvider(ABC):
    """
    Abstract data provider.

    Subclasses implement :meth:`fetch_ohlcv` and optionally override
    :meth:`validate_connection`. Everything else (normalisation, batch
    fetching, latest-bar extraction) is provided here.

    Parameters
    ----------
    timeout :
        HTTP / socket timeout in seconds used by concrete implementations.
    """

    def __init__(self, timeout: int = 30):
        self.timeout = timeout
        self._log = logging.getLogger(f"{__name__}.{self.name}")

    # ------------------------------------------------------------------
    # Identity
    # ------------------------------------------------------------------

    @property
    @abstractmethod
    def name(self) -> str:
        """Short identifier string, e.g. ``'yahoo'``, ``'polygon'``."""

    # ------------------------------------------------------------------
    # Core abstract method
    # ------------------------------------------------------------------

    @abstractmethod
    def fetch_ohlcv(
        self,
        symbol: str,
        start: DateLike,
        end: DateLike,
        interval: str = "1d",
    ) -> OHLCVFrame:
        """
        Fetch OHLCV bars for *symbol* over [*start*, *end*].

        Parameters
        ----------
        symbol :
            Provider-specific ticker (already resolved by the mapper).
        start :
            Inclusive start date (``"YYYY-MM-DD"``, ``date``, or ``datetime``).
        end :
            Inclusive end date.
        interval :
            Bar interval.  Only ``"1d"`` (daily) is mandatory; providers may
            support finer granularity (``"1h"``, ``"5m"`` …).

        Returns
        -------
        pd.DataFrame
            Normalised OHLCV frame with a ``DatetimeIndex``.
            Guaranteed columns: ``open, high, low, close, volume``.

        Raises
        ------
        ProviderError
            Any network / parsing failure.
        SymbolNotFoundError
            Symbol resolved but no data returned.
        """

    # ------------------------------------------------------------------
    # Convenience methods (concrete – built on fetch_ohlcv)
    # ------------------------------------------------------------------

    def fetch_latest(self, symbol: str, lookback_days: int = 5) -> OHLCVFrame:
        """
        Return the most recent available trading bar for *symbol*.

        Uses a rolling *lookback_days* window to account for weekends and
        market holidays.

        Parameters
        ----------
        symbol :
            Provider-specific ticker.
        lookback_days :
            How many calendar days back to look.

        Returns
        -------
        pd.DataFrame
            Single-row OHLCV frame, or empty frame if nothing found.
        """
        end   = date.today()
        start = end - timedelta(days=lookback_days)
        df    = self.fetch_ohlcv(symbol, start=start, end=end)
        if df.empty:
            return df
        return df.iloc[[-1]]   # last row, keep DataFrame shape

    def fetch_multiple_symbols(
        self,
        symbols: list[str],
        start: DateLike,
        end: DateLike,
        interval: str = "1d",
    ) -> dict[str, FetchResult]:
        """
        Batch-fetch OHLCV data for multiple symbols.

        Each symbol is fetched independently; failures are captured in the
        returned :class:`~trading_data.models.FetchResult` rather than
        propagating as exceptions, so one bad ticker never aborts the batch.

        Parameters
        ----------
        symbols :
            List of provider-specific tickers.
        start, end :
            Date range (inclusive).
        interval :
            Bar interval.

        Returns
        -------
        dict[str, FetchResult]
            Keyed by symbol; values carry data *and* error information.
        """
        results: dict[str, FetchResult] = {}
        for sym in symbols:
            try:
                df = self.fetch_ohlcv(sym, start=start, end=end, interval=interval)
                results[sym] = FetchResult(
                    symbol=sym, provider=self.name, data=df, success=True
                )
                self._log.debug("Fetched %d rows for '%s'.", len(df), sym)
            except (ProviderError, SymbolNotFoundError) as exc:
                self._log.warning("Skipping '%s': %s", sym, exc)
                results[sym] = FetchResult(
                    symbol=sym, provider=self.name,
                    data=pd.DataFrame(), success=False, error=str(exc),
                )
            except Exception as exc:          # noqa: BLE001
                self._log.error("Unexpected error for '%s': %s", sym, exc)
                results[sym] = FetchResult(
                    symbol=sym, provider=self.name,
                    data=pd.DataFrame(), success=False, error=str(exc),
                )
        return results

    # ------------------------------------------------------------------
    # Optional hook
    # ------------------------------------------------------------------

    def validate_connection(self) -> bool:
        """
        Check that the provider is reachable and credentials are valid.

        Default implementation always returns ``True``.  Override in
        providers that require authentication (Zerodha, IBKR, Polygon).

        Returns
        -------
        bool
            ``True`` if the provider is ready to accept requests.
        """
        return True

    # ------------------------------------------------------------------
    # Normalisation (called inside every fetch_ohlcv implementation)
    # ------------------------------------------------------------------

    @staticmethod
    def normalise(df: pd.DataFrame, symbol: str, provider: str) -> OHLCVFrame:
        """
        Transform a raw provider DataFrame into the canonical OHLCV schema.

        Steps
        -----
        1. Lower-case all column names.
        2. Keep only ``open, high, low, close, volume``; drop everything else.
        3. Enforce numeric dtypes.
        4. Drop rows where all OHLC values are NaN (e.g. non-trading days).
        5. Sort the index ascending and name it ``date``.
        6. Ensure the index is a ``DatetimeIndex`` (tz-naive, UTC-normalised).

        Parameters
        ----------
        df :
            Raw DataFrame from the provider.
        symbol :
            Ticker name (used in error messages only).
        provider :
            Provider name (used in error messages only).

        Returns
        -------
        pd.DataFrame
            Normalised OHLCV frame.

        Raises
        ------
        SymbolNotFoundError
            If *df* is empty before or after normalisation.
        """
        if df is None or df.empty:
            raise SymbolNotFoundError(symbol, provider)

        df = df.copy()

        # 1. Flatten MultiIndex columns FIRST (yfinance >= 0.2 returns MultiIndex
        #    even for a single ticker, e.g. ('Close', '^NSEI') -> 'close').
        #    Must happen before any .lower() call — tuples have no .lower().
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [c[0].lower().strip() for c in df.columns]
        else:
            df.columns = [c.lower().strip() for c in df.columns]

        # 2. Select only canonical columns that are present
        available = [c for c in OHLCV_COLUMNS if c in df.columns]
        missing   = [c for c in OHLCV_COLUMNS if c not in df.columns]
        if missing:
            logger.warning(
                "[%s] '%s' missing columns: %s", provider, symbol, missing
            )
        df = df[available].copy()

        # 3. Enforce dtypes
        for col, dtype in OHLCV_DTYPES.items():
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
                if dtype == "int64":
                    df[col] = df[col].fillna(0).astype("int64")
                else:
                    df[col] = df[col].astype("float64")

        # 4. Drop rows with all OHLC values NaN
        ohlc = [c for c in ["open", "high", "low", "close"] if c in df.columns]
        df.dropna(subset=ohlc, how="all", inplace=True)

        # 5 & 6. DatetimeIndex
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)
        # Strip timezone info (tz-naive storage)
        if df.index.tz is not None:
            df.index = df.index.tz_localize(None)
        df.index.name = "date"
        df.sort_index(inplace=True)

        if df.empty:
            raise SymbolNotFoundError(symbol, provider)

        return df

    # ------------------------------------------------------------------
    # Date helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _to_date_str(d: DateLike) -> str:
        """Convert *d* to an ISO-8601 string (``YYYY-MM-DD``)."""
        if isinstance(d, str):
            return d
        if isinstance(d, (date, datetime)):
            return d.strftime("%Y-%m-%d")
        raise TypeError(f"Unsupported date type: {type(d)}")

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name!r})"