"""
stock_regime/filters/price_filter.py
======================================
Filters stocks based on absolute price level.

Why price matters
-----------------
Sub-penny and very low-priced stocks (penny stocks on NSE: < ₹10–50;
on NYSE: < $1–2) have wide bid-ask spreads, poor order execution, and
regime signals that are frequently noise rather than signal.  They also
distort relative strength calculations (a move from ₹5 to ₹6 is +20%
but represents ₹1 of absolute value).

Circuit-breaker detection
-------------------------
Indian markets apply a 20% daily circuit breaker.  A stock hitting the
circuit breaker on 3+ days in the past 90 days is exhibiting extreme
volatility or manipulation.  Its regime classification will be unreliable.
We flag (but do not hard-reject by default) these stocks.
"""

from __future__ import annotations

import logging
from typing import Optional

import pandas as pd

from .models import FILTER_PRICE, FilterReason

logger = logging.getLogger(__name__)

# Circuit breaker thresholds by exchange.
# India: 20% upper/lower circuit; NYSE: no per-stock circuit (index-level only)
_CIRCUIT_BREAKER_PCT = {
    "NSE":  0.199,   # flag if |return| > 19.9% (just below the 20% limit)
    "NYSE": 0.50,    # flag if |return| > 50% (split-unadjusted spike)
    "NASDAQ": 0.50,
}
_DEFAULT_CIRCUIT_PCT = 0.50


class PriceFilter:
    """
    Validates that a stock's current price is within acceptable bounds.

    Parameters
    ----------
    min_price : float
        Reject stocks whose latest close is below this value.
        Currency matches the exchange (INR for NSE, USD for NYSE).
    max_price : float or None
        Optional upper price cap.  Rarely needed.
    exchange : str
        Exchange identifier used to select circuit-breaker thresholds.
        One of: ``"NSE"``, ``"NYSE"``, ``"NASDAQ"``.
    circuit_lookback_days : int
        How many calendar days to look back for circuit-breaker events.
    max_circuit_days : int
        Maximum number of circuit-breaker-level moves before flagging.
    circuit_is_fatal : bool
        When True, circuit-breaker detection causes hard rejection.
        When False (default), it logs a warning but allows the stock.
    """

    def __init__(
        self,
        min_price:            float = 10.0,
        max_price:            Optional[float] = None,
        exchange:             str = "NSE",
        circuit_lookback_days:int = 90,
        max_circuit_days:     int = 3,
        circuit_is_fatal:     bool = False,
    ) -> None:
        self.min_price             = min_price
        self.max_price             = max_price
        self.exchange              = exchange.upper()
        self.circuit_lookback_days = circuit_lookback_days
        self.max_circuit_days      = max_circuit_days
        self.circuit_is_fatal      = circuit_is_fatal
        self._circuit_pct          = _CIRCUIT_BREAKER_PCT.get(
            self.exchange, _DEFAULT_CIRCUIT_PCT
        )

    def check(self, symbol: str, df: pd.DataFrame) -> list[FilterReason]:
        """
        Run all price checks on *df*.

        Returns
        -------
        list[FilterReason]
            Empty = passed. Non-empty = failed one or more checks.
        """
        if df.empty:
            return []

        reasons: list[FilterReason] = []
        latest_close = float(df["close"].iloc[-1])

        reasons += self._check_min_price(symbol, latest_close)
        if self.max_price is not None:
            reasons += self._check_max_price(symbol, latest_close)
        reasons += self._check_circuit_breakers(symbol, df)
        return reasons

    # ──────────────────────────────────────────────────────────────────────────
    #  Individual checks
    # ──────────────────────────────────────────────────────────────────────────

    def _check_min_price(self, symbol: str, close: float) -> list[FilterReason]:
        if close < self.min_price:
            logger.debug(
                "%s rejected [price/min_price]: close=%.2f < threshold=%.2f",
                symbol, close, self.min_price,
            )
            return [FilterReason(
                filter_name = FILTER_PRICE,
                check       = "min_price",
                reason      = (
                    f"Close {close:.2f} below minimum {self.min_price:.2f} "
                    f"— likely a penny stock or very illiquid name"
                ),
                metric      = close,
                threshold   = self.min_price,
            )]
        return []

    def _check_max_price(self, symbol: str, close: float) -> list[FilterReason]:
        if self.max_price and close > self.max_price:
            return [FilterReason(
                filter_name = FILTER_PRICE,
                check       = "max_price",
                reason      = f"Close {close:.2f} above maximum {self.max_price:.2f}",
                metric      = close,
                threshold   = self.max_price,
            )]
        return []

    def _check_circuit_breakers(
        self, symbol: str, df: pd.DataFrame
    ) -> list[FilterReason]:
        """
        Count days in the recent lookback window where the daily return
        exceeded the circuit-breaker threshold.

        For NSE stocks this catches stocks hitting the 20% upper/lower circuit.
        For NYSE/NASDAQ it catches split-unadjusted price spikes.
        """
        cutoff = df.index[-1] - pd.Timedelta(days=self.circuit_lookback_days)
        recent = df[df.index >= cutoff]["close"]

        if len(recent) < 2:
            return []

        abs_returns   = recent.pct_change().abs().dropna()
        circuit_days  = int((abs_returns > self._circuit_pct).sum())

        if circuit_days > self.max_circuit_days:
            msg = (
                f"{circuit_days} circuit-breaker events (|return| > "
                f"{self._circuit_pct*100:.0f}%) in last "
                f"{self.circuit_lookback_days} days "
                f"(threshold: {self.max_circuit_days})"
            )
            if self.circuit_is_fatal:
                logger.debug("%s rejected [price/circuit_breaker]: %s", symbol, msg)
                return [FilterReason(
                    filter_name = FILTER_PRICE,
                    check       = "circuit_breaker",
                    reason      = msg,
                    metric      = float(circuit_days),
                    threshold   = float(self.max_circuit_days),
                )]
            else:
                # Warning only — log but do not reject
                logger.warning(
                    "%s [price/circuit_breaker] WARNING: %s", symbol, msg
                )

        return []