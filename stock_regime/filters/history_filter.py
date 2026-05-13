"""
stock_regime/filters/history_filter.py
========================================
Validates that a stock has sufficient OHLCV history for reliable indicator
computation.

Checks (in order)
-----------------
1. min_bars         — fewer than N rows → EMA-200 will be NaN on most bars
2. max_gap_days     — large gaps in recent history → unreliable rolling indicators
3. max_stale_days   — last bar too old → stock is suspended or delisted

These are the cheapest checks to compute (no arithmetic beyond counting),
so they run first to short-circuit the more expensive price and liquidity checks.
"""

from __future__ import annotations

import logging
from datetime import date, timedelta

import numpy as np
import pandas as pd

from .models import FILTER_HISTORY, FilterReason

logger = logging.getLogger(__name__)


class HistoryFilter:
    """
    Validates OHLCV history depth and continuity.

    Parameters
    ----------
    min_bars : int
        Minimum number of OHLCV rows required.
        Recommended: 220 (200 for EMA-200 + 20 warmup).
    max_gap_days : int
        Maximum consecutive missing trading days allowed in the last
        ``gap_lookback_days`` calendar days.  A gap larger than this
        suggests a suspension or data feed issue.
    max_stale_days : int
        Maximum number of calendar days between the last bar and today.
        Stocks with stale data may be suspended.
    gap_lookback_days : int
        How many calendar days to look back when checking for gaps.
        Default 90 (≈ one quarter).
    """

    def __init__(
        self,
        min_bars:          int = 220,
        max_gap_days:      int = 5,
        max_stale_days:    int = 3,
        gap_lookback_days: int = 90,
    ) -> None:
        self.min_bars          = min_bars
        self.max_gap_days      = max_gap_days
        self.max_stale_days    = max_stale_days
        self.gap_lookback_days = gap_lookback_days

    def check(self, symbol: str, df: pd.DataFrame) -> list[FilterReason]:
        """
        Run all history checks on *df*.

        Returns
        -------
        list[FilterReason]
            Empty list = symbol passed all checks.
            Non-empty list = at least one check failed.
        """
        reasons: list[FilterReason] = []
        reasons += self._check_min_bars(symbol, df)
        reasons += self._check_staleness(symbol, df)
        reasons += self._check_gaps(symbol, df)
        return reasons

    # ──────────────────────────────────────────────────────────────────────────
    #  Individual checks
    # ──────────────────────────────────────────────────────────────────────────

    def _check_min_bars(self, symbol: str, df: pd.DataFrame) -> list[FilterReason]:
        n = len(df)
        if n < self.min_bars:
            logger.debug(
                "%s rejected [history/min_bars]: %d rows < %d required",
                symbol, n, self.min_bars,
            )
            return [FilterReason(
                filter_name = FILTER_HISTORY,
                check       = "min_bars",
                reason      = f"Only {n} bars; need {self.min_bars} for EMA-200 warmup",
                metric      = float(n),
                threshold   = float(self.min_bars),
            )]
        return []

    def _check_staleness(self, symbol: str, df: pd.DataFrame) -> list[FilterReason]:
        """Reject if the most recent bar is older than max_stale_days."""
        if df.empty:
            return []
        last_bar_date = df.index[-1].date()
        today         = date.today()
        age_days      = (today - last_bar_date).days

        # Adjust for weekends: a bar on Friday is 3 days old on Monday, which is fine.
        # We use calendar days and set max_stale_days >= 3 by default.
        if age_days > self.max_stale_days:
            logger.debug(
                "%s rejected [history/stale]: last bar %s is %d days old",
                symbol, last_bar_date, age_days,
            )
            return [FilterReason(
                filter_name = FILTER_HISTORY,
                check       = "stale_data",
                reason      = (
                    f"Last bar {last_bar_date} is {age_days} calendar days old "
                    f"(threshold: {self.max_stale_days}). Stock may be suspended."
                ),
                metric      = float(age_days),
                threshold   = float(self.max_stale_days),
            )]
        return []

    def _check_gaps(self, symbol: str, df: pd.DataFrame) -> list[FilterReason]:
        """
        Check for unexpectedly large gaps between trading days in recent history.

        We look at the last gap_lookback_days calendar days and find the maximum
        consecutive-day gap between actual trading bars.  Weekends produce a
        natural gap of 3 days (Friday → Monday).  We flag gaps larger than
        max_gap_days, which catches suspensions, circuit breakers, and data feeds
        that missed a day.
        """
        if df.empty or len(df) < 2:
            return []

        cutoff   = df.index[-1] - pd.Timedelta(days=self.gap_lookback_days)
        recent   = df[df.index >= cutoff]

        if len(recent) < 2:
            return []

        # Differences between consecutive bar dates in calendar days
        diffs = recent.index.to_series().diff().dt.days.dropna()

        # Weekends contribute 3-day gaps (Fri → Mon); holidays up to 4.
        # We flag anything beyond max_gap_days (default 5).
        max_gap = diffs.max()

        if max_gap > self.max_gap_days:
            gap_idx = diffs.idxmax()
            logger.debug(
                "%s rejected [history/gap]: %d-day gap ending %s (threshold: %d)",
                symbol, int(max_gap), gap_idx.date(), self.max_gap_days,
            )
            return [FilterReason(
                filter_name = FILTER_HISTORY,
                check       = "data_gap",
                reason      = (
                    f"Gap of {int(max_gap)} calendar days found ending {gap_idx.date()} "
                    f"in the last {self.gap_lookback_days} days "
                    f"(threshold: {self.max_gap_days})"
                ),
                metric      = float(max_gap),
                threshold   = float(self.max_gap_days),
            )]
        return []