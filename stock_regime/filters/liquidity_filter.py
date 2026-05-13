"""
stock_regime/filters/liquidity_filter.py
==========================================
Filters stocks that are too illiquid to trade reliably.

Three checks are applied:

1. Average Daily Value (ADV)
   ADV = mean(close × volume) over adv_period trading days.
   A stock below min_adv cannot be entered/exited at scale without
   significant market impact.  This is the primary filter.

2. Zero-volume day ratio
   A stock with many zero-volume days has structural liquidity problems
   (no market maker, circuit-halted, effectively suspended).

3. Volume consistency (coefficient of variation)
   Very erratic volume (CV > threshold) suggests block trades, options
   expiry distortions, or data errors rather than genuine trading interest.
   This is a softer signal — flagged as a warning, not a hard rejection.

ADV currency
------------
NSE stocks: ADV is computed in INR (close × volume where close is in ₹)
NYSE/NASDAQ: ADV in USD

The thresholds are configured separately per exchange in config.yaml.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd

from .models import FILTER_LIQUIDITY, FilterReason

logger = logging.getLogger(__name__)


class LiquidityFilter:
    """
    Validates that a stock has sufficient liquidity for reliable classification
    and practical tradability.

    Parameters
    ----------
    min_adv :
        Minimum average daily value traded.  In INR for NSE (specify in Crore:
        1 Crore = 10,000,000 INR), in USD for NYSE.
    adv_in_crore :
        When True, min_adv is expressed in Crore INR.  Set to False for USD.
    adv_period :
        Number of most-recent trading days to use for ADV calculation.
    max_zero_volume_ratio :
        Maximum fraction of bars (over adv_period) that may have zero volume.
    max_volume_cv :
        Maximum coefficient of variation (std/mean) for volume.
        Values > 3.0 suggest pathological volume distribution.
        Set to None to disable this check.
    zero_volume_is_fatal :
        When True, zero-volume ratio breach causes hard rejection (default True).
    volume_cv_is_fatal :
        When True, CV breach causes hard rejection (default False — warning only).
    """

    def __init__(
        self,
        min_adv:               float = 50.0,
        adv_in_crore:          bool  = True,
        adv_period:            int   = 20,
        max_zero_volume_ratio: float = 0.05,
        max_volume_cv:         Optional[float] = 5.0,
        zero_volume_is_fatal:  bool  = True,
        volume_cv_is_fatal:    bool  = False,
    ) -> None:
        self.min_adv                = min_adv
        self.adv_in_crore           = adv_in_crore
        self.adv_period             = adv_period
        self.max_zero_volume_ratio  = max_zero_volume_ratio
        self.max_volume_cv          = max_volume_cv
        self.zero_volume_is_fatal   = zero_volume_is_fatal
        self.volume_cv_is_fatal     = volume_cv_is_fatal

        # Precompute the ADV threshold in raw currency units for comparison
        self._min_adv_raw = min_adv * 1e7 if adv_in_crore else min_adv * 1e6

        label = f"₹{min_adv}Cr" if adv_in_crore else f"${min_adv}M"
        logger.debug("LiquidityFilter: min_adv=%s  adv_period=%dd", label, adv_period)

    def check(self, symbol: str, df: pd.DataFrame) -> list[FilterReason]:
        """
        Run all liquidity checks on *df*.

        Returns
        -------
        list[FilterReason]
            Empty = passed. Non-empty = failed at least one hard check.
        """
        if df.empty or "volume" not in df.columns or "close" not in df.columns:
            return []

        recent = df.tail(self.adv_period)
        if len(recent) < max(self.adv_period // 2, 5):
            # Not enough data for a meaningful ADV calculation
            return []

        reasons: list[FilterReason] = []
        reasons += self._check_adv(symbol, recent)
        reasons += self._check_zero_volume(symbol, recent)
        reasons += self._check_volume_cv(symbol, recent)
        return reasons

    # ──────────────────────────────────────────────────────────────────────────
    #  Individual checks
    # ──────────────────────────────────────────────────────────────────────────

    def _check_adv(self, symbol: str, recent: pd.DataFrame) -> list[FilterReason]:
        """
        Average Daily Value = mean(close × volume) over adv_period days.
        Excludes zero-volume days from the average (they would deflate ADV
        for stocks with genuine volume on trading days).
        """
        daily_value = recent["close"] * recent["volume"]
        # Exclude zero-volume days from the mean
        nonzero     = daily_value[daily_value > 0]

        if nonzero.empty:
            adv_raw = 0.0
        else:
            adv_raw = float(nonzero.mean())

        # Convert for display
        adv_display = adv_raw / 1e7 if self.adv_in_crore else adv_raw / 1e6
        unit        = "Cr" if self.adv_in_crore else "M"

        if adv_raw < self._min_adv_raw:
            logger.debug(
                "%s rejected [liquidity/min_adv]: ADV=%.2f%s < %.2f%s",
                symbol, adv_display, unit, self.min_adv, unit,
            )
            return [FilterReason(
                filter_name = FILTER_LIQUIDITY,
                check       = "min_adv",
                reason      = (
                    f"ADV={adv_display:.2f}{unit} below minimum "
                    f"{self.min_adv:.2f}{unit} — stock is too illiquid"
                ),
                metric      = adv_display,
                threshold   = self.min_adv,
            )]
        return []

    def _check_zero_volume(self, symbol: str, recent: pd.DataFrame) -> list[FilterReason]:
        """
        Fraction of bars with zero volume.
        A high ratio indicates a structurally illiquid or suspended stock.
        """
        n_bars       = len(recent)
        n_zero       = int((recent["volume"] == 0).sum())
        zero_ratio   = n_zero / n_bars

        if zero_ratio > self.max_zero_volume_ratio:
            msg = (
                f"{n_zero}/{n_bars} bars ({zero_ratio*100:.1f}%) have zero volume "
                f"(threshold: {self.max_zero_volume_ratio*100:.1f}%)"
            )
            if self.zero_volume_is_fatal:
                logger.debug("%s rejected [liquidity/zero_volume]: %s", symbol, msg)
                return [FilterReason(
                    filter_name = FILTER_LIQUIDITY,
                    check       = "zero_volume_ratio",
                    reason      = msg,
                    metric      = zero_ratio,
                    threshold   = self.max_zero_volume_ratio,
                )]
            else:
                logger.warning("%s [liquidity/zero_volume] WARNING: %s", symbol, msg)

        return []

    def _check_volume_cv(self, symbol: str, recent: pd.DataFrame) -> list[FilterReason]:
        """
        Coefficient of variation (std/mean) for volume.
        Very high CV suggests block trades, data errors, or pathological volume.
        This is a soft check by default.
        """
        if self.max_volume_cv is None:
            return []

        vol  = recent["volume"].replace(0, pd.NA).dropna()
        if len(vol) < 5:
            return []

        mean_vol = float(vol.mean())
        std_vol  = float(vol.std())
        cv       = std_vol / mean_vol if mean_vol > 0 else 0.0

        if cv > self.max_volume_cv:
            msg = (
                f"Volume CV={cv:.2f} exceeds threshold {self.max_volume_cv:.2f} "
                f"— erratic volume pattern"
            )
            if self.volume_cv_is_fatal:
                logger.debug("%s rejected [liquidity/volume_cv]: %s", symbol, msg)
                return [FilterReason(
                    filter_name = FILTER_LIQUIDITY,
                    check       = "volume_cv",
                    reason      = msg,
                    metric      = cv,
                    threshold   = self.max_volume_cv,
                )]
            else:
                logger.warning("%s [liquidity/volume_cv] WARNING: %s", symbol, msg)

        return []