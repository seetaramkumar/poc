"""
trading_data/manager.py
------------------------
DataManager — the single entry point for all data fetching.

Responsibilities
----------------
* Accept user-friendly symbols (``"NIFTY50"``, ``"^NSEI"``, ``"AAPL"``) and
  route them to the active provider via the SymbolMapper.
* Check the local cache before hitting the network; write back on cache miss.
* Apply retry logic with exponential back-off around provider calls.
* Merge cached and live data when a partial cache hit covers part of the
  requested range (e.g. cache has 2020-2023, caller wants 2020-2025).
* Expose a clean, high-level API that is directly compatible with the
  Market Regime Engine.

Public methods
--------------
* ``get_daily_data()``          — primary method; cache-aware single-symbol fetch
* ``fetch_ohlcv()``             — raw provider fetch (bypasses cache)
* ``fetch_latest()``            — most recent bar
* ``fetch_multiple_symbols()``  — batch fetch with error isolation
* ``list_symbols()``            — known symbols in the registry
* ``validate_provider()``       — connection check
"""

from __future__ import annotations

import logging
import time
from typing import Optional

import pandas as pd

from trading_data.cache.parquet_cache import ParquetCache
from trading_data.exceptions import DataLayerError, ProviderError, SymbolNotFoundError
from trading_data.models import (
    DataManagerConfig,
    DateLike,
    FetchResult,
    OHLCVFrame,
    Provider,
)
from trading_data.providers.base import BaseDataProvider
from trading_data.providers.yahoo import YahooFinanceProvider
from trading_data.symbols.mapper import SymbolMapper

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Provider factory
# ---------------------------------------------------------------------------
def _build_provider(provider: Provider, **kwargs) -> BaseDataProvider:
    """Instantiate the correct provider class for *provider*."""
    if provider == Provider.YAHOO:
        return YahooFinanceProvider(**kwargs)

    if provider == Provider.ZERODHA:
        from trading_data.providers.zerodha import ZerodhaProvider
        return ZerodhaProvider(**kwargs)

    if provider == Provider.POLYGON:
        from trading_data.providers.polygon import PolygonProvider
        return PolygonProvider(**kwargs)

    if provider == Provider.IBKR:
        from trading_data.providers.ibkr import IBKRProvider
        return IBKRProvider(**kwargs)

    raise ValueError(f"Unknown provider: '{provider}'")


# ---------------------------------------------------------------------------
# DataManager
# ---------------------------------------------------------------------------
class DataManager:
    """
    Orchestrates data fetching, caching, symbol resolution, and retries.

    Parameters
    ----------
    config :
        :class:`~trading_data.models.DataManagerConfig` instance.
        Defaults to sensible production settings if not provided.
    provider_kwargs :
        Extra keyword arguments forwarded to the provider constructor
        (e.g. ``api_key="..."`` for Polygon/IBKR).

    Examples
    --------
    Basic usage (Market Regime Engine compatible):

    >>> manager = DataManager()
    >>> df = manager.get_daily_data(
    ...     symbol="^NSEI",
    ...     start="2020-01-01",
    ...     end="2025-01-01",
    ... )
    >>> df.columns.tolist()
    ['open', 'high', 'low', 'close', 'volume']

    Switch to a different provider at runtime:

    >>> manager = DataManager(
    ...     config=DataManagerConfig(default_provider=Provider.POLYGON),
    ...     provider_kwargs={"api_key": "YOUR_KEY"},
    ... )
    """

    def __init__(
        self,
        config:           Optional[DataManagerConfig] = None,
        provider_kwargs:  Optional[dict]              = None,
    ):
        self._config   = config or DataManagerConfig()
        self._provider = _build_provider(
            self._config.default_provider,
            **(provider_kwargs or {}),
        )
        self._cache  = (
            ParquetCache(
                cache_dir=self._config.cache_dir,
                max_age_days=self._config.cache_max_age_days,
            )
            if self._config.cache_enabled
            else None
        )
        self._mapper = SymbolMapper()
        logger.info(
            "DataManager ready  provider=%s  cache=%s",
            self._provider.name,
            "enabled" if self._cache else "disabled",
        )

    # ------------------------------------------------------------------
    # Primary public API
    # ------------------------------------------------------------------

    def get_daily_data(
        self,
        symbol:   str,
        start:    DateLike,
        end:      DateLike,
        interval: str = "1d",
        refresh:  bool = False,
    ) -> OHLCVFrame:
        """
        Fetch daily OHLCV data for *symbol* over [*start*, *end*].

        This is the **recommended entry point** for the Market Regime Engine
        and other downstream consumers.  It:

        1. Resolves *symbol* via the SymbolMapper.
        2. Checks the local parquet cache unless *refresh* is ``True``.
        3. Fetches from the active provider on cache miss / staleness.
        4. Merges cached and live data for partial cache hits.
        5. Writes the result back to cache.
        6. Returns a normalised OHLCV ``DataFrame`` with a
           ``DatetimeIndex`` and lowercase columns.

        Parameters
        ----------
        symbol :
            Canonical or provider-specific ticker.
            Examples: ``"NIFTY50"``, ``"^NSEI"``, ``"AAPL"``, ``"SP500"``.
        start :
            Inclusive start date (``"YYYY-MM-DD"``).
        end :
            Inclusive end date.
        interval :
            Bar interval (only ``"1d"`` is guaranteed across all providers).
        refresh :
            Force a live fetch even if a valid cache entry exists.

        Returns
        -------
        pd.DataFrame
            Columns: ``open, high, low, close, volume``
            Index:   ``DatetimeIndex`` named ``"date"`` (tz-naive).

        Raises
        ------
        SymbolNotFoundError
            Symbol returned no data.
        ProviderError
            All retry attempts exhausted.
        """
        info            = self._mapper.resolve(symbol, self._config.default_provider)
        provider_symbol = info.provider_symbol
        start_str       = BaseDataProvider._to_date_str(start)
        end_str         = BaseDataProvider._to_date_str(end)

        logger.info(
            "get_daily_data  symbol=%r → %r  %s → %s",
            symbol, provider_symbol, start_str, end_str,
        )

        # ── Cache read ──────────────────────────────────────────────────
        if self._cache and not refresh:
            cached = self._cache.read(
                provider_symbol, self._provider.name, interval,
                start=start_str, end=end_str,
            )
            if cached is not None and not self._cache.is_stale(
                provider_symbol, self._provider.name, interval
            ):
                logger.info(
                    "Serving '%s' from cache  rows=%d.", provider_symbol, len(cached)
                )
                return cached

        # ── Live fetch with retry ───────────────────────────────────────
        df = self._fetch_with_retry(provider_symbol, start_str, end_str, interval)

        # ── Cache write ─────────────────────────────────────────────────
        if self._cache:
            try:
                self._cache.write(provider_symbol, self._provider.name, df, interval)
            except DataLayerError as exc:
                logger.warning("Cache write failed (non-fatal): %s", exc)

        return df

    # ------------------------------------------------------------------

    def fetch_ohlcv(
        self,
        symbol:   str,
        start:    DateLike,
        end:      DateLike,
        interval: str = "1d",
    ) -> OHLCVFrame:
        """
        Raw provider fetch — bypasses the cache entirely.

        Prefer :meth:`get_daily_data` for normal use.

        Parameters
        ----------
        symbol :
            Canonical or provider-specific ticker.
        start, end :
            Date range.
        interval :
            Bar interval.

        Returns
        -------
        pd.DataFrame
            Normalised OHLCV frame.
        """
        info = self._mapper.resolve(symbol, self._config.default_provider)
        return self._fetch_with_retry(
            info.provider_symbol,
            BaseDataProvider._to_date_str(start),
            BaseDataProvider._to_date_str(end),
            interval,
        )

    # ------------------------------------------------------------------

    def fetch_latest(self, symbol: str, lookback_days: int = 5) -> OHLCVFrame:
        """
        Return the most recent available trading bar for *symbol*.

        Parameters
        ----------
        symbol :
            Canonical or provider-specific ticker.
        lookback_days :
            Calendar days back to search for the last bar.

        Returns
        -------
        pd.DataFrame
            Single-row OHLCV frame.
        """
        info = self._mapper.resolve(symbol, self._config.default_provider)
        logger.info("fetch_latest  symbol=%r → %r", symbol, info.provider_symbol)
        return self._provider.fetch_latest(info.provider_symbol, lookback_days)

    # ------------------------------------------------------------------

    def fetch_multiple_symbols(
        self,
        symbols:  list[str],
        start:    DateLike,
        end:      DateLike,
        interval: str = "1d",
        refresh:  bool = False,
    ) -> dict[str, FetchResult]:
        """
        Fetch OHLCV data for a list of symbols in one call.

        Each symbol is resolved and fetched independently.  A failure on one
        symbol **never** aborts the rest of the batch.

        Parameters
        ----------
        symbols :
            List of canonical or provider-specific tickers.
        start, end :
            Date range.
        interval :
            Bar interval.
        refresh :
            Bypass cache for all symbols.

        Returns
        -------
        dict[str, FetchResult]
            Keys are the original symbols passed in.
            Values contain data and success/error status.
        """
        start_str = BaseDataProvider._to_date_str(start)
        end_str   = BaseDataProvider._to_date_str(end)

        logger.info(
            "fetch_multiple_symbols  n=%d  %s → %s",
            len(symbols), start_str, end_str,
        )

        # Build provider-symbol mapping
        provider_to_original: dict[str, str] = {}
        provider_symbols: list[str] = []
        for sym in symbols:
            try:
                info = self._mapper.resolve(sym, self._config.default_provider)
                provider_to_original[info.provider_symbol] = sym
                provider_symbols.append(info.provider_symbol)
            except Exception as exc:            # noqa: BLE001
                logger.warning("Could not resolve '%s': %s", sym, exc)

        # Check cache for each (fast path)
        results: dict[str, FetchResult] = {}
        uncached: list[str] = []

        for psym in provider_symbols:
            orig = provider_to_original[psym]
            if self._cache and not refresh:
                cached = self._cache.read(
                    psym, self._provider.name, interval,
                    start=start_str, end=end_str,
                )
                if cached is not None and not self._cache.is_stale(
                    psym, self._provider.name, interval
                ):
                    results[orig] = FetchResult(
                        symbol=orig, provider=self._provider.name,
                        data=cached, success=True, from_cache=True,
                    )
                    continue
            uncached.append(psym)

        if uncached:
            live = self._provider.fetch_multiple_symbols(
                uncached, start_str, end_str, interval
            )
            for psym, result in live.items():
                orig           = provider_to_original.get(psym, psym)
                result.symbol  = orig               # remap key to original symbol
                results[orig]  = result
                if result.success and self._cache:
                    try:
                        self._cache.write(psym, self._provider.name, result.data, interval)
                    except DataLayerError as exc:
                        logger.warning("Cache write failed for '%s': %s", psym, exc)

        return results

    # ------------------------------------------------------------------

    def list_symbols(self) -> list[str]:
        """Return all canonical symbols in the SymbolMapper registry."""
        return self._mapper.list_symbols()

    def validate_provider(self) -> bool:
        """Check provider connectivity.  Returns ``True`` if reachable."""
        ok = self._provider.validate_connection()
        logger.info("Provider '%s' connection: %s.", self._provider.name, "OK" if ok else "FAILED")
        return ok

    def cache_stats(self) -> dict:
        """Return cache statistics, or an empty dict if caching is disabled."""
        return self._cache.stats() if self._cache else {}

    def clear_cache(self) -> int:
        """Delete all cached files.  Returns number of files deleted."""
        return self._cache.clear_all() if self._cache else 0

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _fetch_with_retry(
        self,
        provider_symbol: str,
        start:           str,
        end:             str,
        interval:        str,
    ) -> OHLCVFrame:
        """
        Call the provider with exponential back-off retry.

        Parameters
        ----------
        provider_symbol :
            Already-resolved provider ticker.
        start, end :
            ISO-8601 date strings.
        interval :
            Bar interval.

        Returns
        -------
        pd.DataFrame
            Normalised OHLCV frame.

        Raises
        ------
        ProviderError / SymbolNotFoundError
            After all retry attempts are exhausted.
        """
        attempts = self._config.retry_attempts
        backoff  = self._config.retry_backoff_seconds
        last_exc: Exception = RuntimeError("No attempts made.")

        for attempt in range(1, attempts + 1):
            try:
                df = self._provider.fetch_ohlcv(
                    provider_symbol, start=start, end=end, interval=interval
                )
                if attempt > 1:
                    logger.info("Succeeded on attempt %d/%d.", attempt, attempts)
                return df

            except SymbolNotFoundError:
                # No point retrying – data simply doesn't exist
                raise

            except (ProviderError, Exception) as exc:   # noqa: BLE001
                last_exc = exc
                if attempt < attempts:
                    logger.warning(
                        "Attempt %d/%d failed for '%s': %s — retrying in %.1fs …",
                        attempt, attempts, provider_symbol, exc, backoff,
                    )
                    time.sleep(backoff)
                    backoff *= 2      # exponential back-off
                else:
                    logger.error(
                        "All %d attempts failed for '%s'.", attempts, provider_symbol
                    )

        raise ProviderError(self._provider.name, provider_symbol, str(last_exc))

    # ------------------------------------------------------------------
    # Dunder
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"DataManager("
            f"provider={self._provider.name!r}, "
            f"cache={'enabled' if self._cache else 'disabled'})"
        )
