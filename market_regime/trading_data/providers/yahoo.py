"""
trading_data/providers/yahoo.py
--------------------------------
Yahoo Finance provider powered by ``yfinance``.

Key characteristics
-------------------
* Handles both individual and batch downloads via ``yf.download()``.
* Covers global indices (``^NSEI``, ``^GSPC``), US equities, and NSE/BSE
  stocks with the ``.NS`` / ``.BO`` suffixes.
* Auto-detects yfinance column schema (flat vs MultiIndex).
* Implements the retry / back-off protocol via the base class machinery.
"""

from __future__ import annotations

import logging
from typing import Optional

import pandas as pd
import yfinance as yf

from trading_data.exceptions import ProviderError, SymbolNotFoundError
from trading_data.models import DateLike, OHLCVFrame
from trading_data.providers.base import BaseDataProvider

logger = logging.getLogger(__name__)


class YahooFinanceProvider(BaseDataProvider):
    """
    Data provider backed by `yfinance <https://github.com/ranaroussi/yfinance>`_.

    Parameters
    ----------
    auto_adjust :
        When ``True`` (default) yfinance returns split- and dividend-adjusted
        prices.  Set to ``False`` to get raw prices.
    timeout :
        Socket timeout for HTTP requests (seconds).
    progress :
        Show yfinance download progress bar.  Disable in production.

    Examples
    --------
    >>> from trading_data.providers.yahoo import YahooFinanceProvider
    >>> p = YahooFinanceProvider()
    >>> df = p.fetch_ohlcv("^NSEI", "2023-01-01", "2024-01-01")
    >>> df.columns.tolist()
    ['open', 'high', 'low', 'close', 'volume']
    """

    def __init__(
        self,
        auto_adjust: bool = True,
        timeout:     int  = 30,
        progress:    bool = False,
    ):
        super().__init__(timeout=timeout)
        self._auto_adjust = auto_adjust
        self._progress    = progress

    # ------------------------------------------------------------------
    # Identity
    # ------------------------------------------------------------------

    @property
    def name(self) -> str:
        return "yahoo"

    # ------------------------------------------------------------------
    # Core fetch
    # ------------------------------------------------------------------

    def fetch_ohlcv(
        self,
        symbol:   str,
        start:    DateLike,
        end:      DateLike,
        interval: str = "1d",
    ) -> OHLCVFrame:
        """
        Download OHLCV data from Yahoo Finance.

        Parameters
        ----------
        symbol :
            Yahoo Finance ticker (e.g. ``"^NSEI"``, ``"RELIANCE.NS"``).
        start :
            Inclusive start date.
        end :
            Inclusive end date.
        interval :
            Bar interval (``"1d"``, ``"1wk"``, ``"1mo"`` etc.).

        Returns
        -------
        pd.DataFrame
            Normalised OHLCV frame indexed by date.

        Raises
        ------
        ProviderError
            Network failure or yfinance exception.
        SymbolNotFoundError
            Ticker is valid but returned no rows.
        """
        start_str = self._to_date_str(start)
        end_str   = self._to_date_str(end)

        self._log.debug(
            "Downloading '%s'  %s → %s  interval=%s",
            symbol, start_str, end_str, interval,
        )

        try:
            raw: pd.DataFrame = yf.download(
                tickers=symbol,
                start=start_str,
                end=end_str,
                interval=interval,
                auto_adjust=self._auto_adjust,
                progress=self._progress,
                threads=False,           # single symbol – threading adds overhead
                multi_level_index=False, # yfinance >= 1.0: flat columns for single ticker
            )
        except Exception as exc:      # noqa: BLE001
            raise ProviderError(self.name, symbol, str(exc)) from exc

        if raw is None or raw.empty:
            raise SymbolNotFoundError(symbol, self.name)

        # yfinance ≥ 0.2.x returns a MultiIndex when downloading multiple
        # tickers; for a single ticker it returns a flat column frame.
        # normalise() handles both cases.
        return self.normalise(raw, symbol=symbol, provider=self.name)

    # ------------------------------------------------------------------
    # Optimised batch download
    # ------------------------------------------------------------------

    def fetch_multiple_symbols(
        self,
        symbols:  list[str],
        start:    DateLike,
        end:      DateLike,
        interval: str = "1d",
    ) -> dict[str, "FetchResult"]:  # noqa: F821  (forward ref ok at runtime)
        """
        Download multiple tickers in a single yfinance batch call.

        Overrides the base class implementation to exploit ``yf.download``'s
        native multi-ticker support, which is significantly faster than
        sequential calls for large symbol lists.

        Parameters
        ----------
        symbols :
            List of Yahoo Finance tickers.
        start, end :
            Date range.
        interval :
            Bar interval.

        Returns
        -------
        dict[str, FetchResult]
            One entry per symbol; errors are isolated per symbol.
        """
        from trading_data.models import FetchResult  # local import to avoid cycle

        if len(symbols) == 1:
            return super().fetch_multiple_symbols(symbols, start, end, interval)

        start_str = self._to_date_str(start)
        end_str   = self._to_date_str(end)

        self._log.info(
            "Batch-downloading %d symbols  %s → %s",
            len(symbols), start_str, end_str,
        )

        results: dict[str, FetchResult] = {}

        try:
            raw: pd.DataFrame = yf.download(
                tickers=symbols,
                start=start_str,
                end=end_str,
                interval=interval,
                auto_adjust=self._auto_adjust,
                progress=self._progress,
                group_by="ticker",
                threads=True,
            )
        except Exception as exc:          # noqa: BLE001
            # Fall back to sequential on batch failure
            self._log.warning(
                "Batch download failed (%s); falling back to sequential.", exc
            )
            return super().fetch_multiple_symbols(symbols, start, end, interval)

        for sym in symbols:
            try:
                if isinstance(raw.columns, pd.MultiIndex):
                    sym_df = raw[sym].copy() if sym in raw.columns.get_level_values(0) else pd.DataFrame()
                else:
                    sym_df = raw.copy()

                df = self.normalise(sym_df, symbol=sym, provider=self.name)
                results[sym] = FetchResult(
                    symbol=sym, provider=self.name, data=df, success=True
                )
            except SymbolNotFoundError as exc:
                results[sym] = FetchResult(
                    symbol=sym, provider=self.name,
                    data=pd.DataFrame(), success=False, error=str(exc),
                )
            except Exception as exc:      # noqa: BLE001
                results[sym] = FetchResult(
                    symbol=sym, provider=self.name,
                    data=pd.DataFrame(), success=False, error=str(exc),
                )

        return results

    # ------------------------------------------------------------------
    # Connection check
    # ------------------------------------------------------------------

    def validate_connection(self) -> bool:
        """
        Probe Yahoo Finance with a lightweight ticker info call.

        Returns
        -------
        bool
            ``True`` if Yahoo Finance is reachable.
        """
        try:
            ticker = yf.Ticker("AAPL")
            info   = ticker.fast_info          # lightweight endpoint
            return bool(info)
        except Exception:                      # noqa: BLE001
            return False