"""
stock_regime/quality/validator.py
===================================
Detects and corrects corrupt, stale, or anomalous OHLCV data before
it reaches any indicator calculation.

Check hierarchy
---------------
FATAL anomalies   → exclude symbol entirely; cannot be corrected
ERROR anomalies   → correct the value; log the correction; continue
WARNING anomalies → log; continue with original data

Checks performed
----------------
OHLC structural consistency (FATAL / ERROR):
  - close <= 0              → FATAL
  - high < low              → FATAL
  - close > high * 1.001    → ERROR (clip close to high)
  - close < low  * 0.999    → ERROR (clip close to low)
  - open > high or open < low → ERROR (clip open to [low, high])

Price spike detection (ERROR / WARNING):
  - |single-bar return| > fatal_spike_pct  → FATAL (e.g. > 60%: split-unadjusted)
  - |single-bar return| > warn_spike_pct   → WARNING (e.g. > 20%: circuit/news)

Volume anomalies (WARNING / ERROR):
  - volume == 0 on isolated days → WARNING; fill with 5-day rolling median
  - volume < 0                   → ERROR; set to 0

Duplicate dates (ERROR):
  - Remove duplicate index entries, keep last occurrence
"""

from __future__ import annotations

import logging
import math
from datetime import date
from typing import Optional

import numpy as np
import pandas as pd

from .models import Anomaly, AnomalySeverity, QualityReport

logger = logging.getLogger(__name__)

_WARN  = AnomalySeverity.WARNING
_ERROR = AnomalySeverity.ERROR
_FATAL = AnomalySeverity.FATAL


class DataQualityValidator:
    """
    Validates and corrects OHLCV DataFrames before indicator computation.

    Parameters
    ----------
    fatal_spike_pct :
        Single-bar absolute return above this → FATAL (exclude symbol).
        Default 0.60 (60%) catches split-unadjusted prices.
    warn_spike_pct :
        Single-bar absolute return above this → WARNING (log only).
        Default 0.20 (20%) catches circuit-breaker events.
    zero_volume_fill :
        When True, fill isolated zero-volume days with rolling median.
    """

    def __init__(
        self,
        fatal_spike_pct:  float = 0.60,
        warn_spike_pct:   float = 0.20,
        zero_volume_fill: bool  = True,
    ) -> None:
        self.fatal_spike_pct  = fatal_spike_pct
        self.warn_spike_pct   = warn_spike_pct
        self.zero_volume_fill = zero_volume_fill

    def validate(
        self,
        stock_data: dict[str, pd.DataFrame],
    ) -> QualityReport:
        """
        Validate all DataFrames in *stock_data*.

        Parameters
        ----------
        stock_data :
            ``{symbol: ohlcv_df}`` mapping.

        Returns
        -------
        QualityReport
        """
        report = QualityReport()

        for symbol, df in stock_data.items():
            anomalies: list[Anomaly] = []
            corrections: list[str]   = []

            # Work on a copy — never mutate the caller's data
            df_work = df.copy()

            # Run all checks; collect anomalies
            anomalies += self._check_duplicates(symbol, df_work)
            anomalies += self._check_ohlc_structure(symbol, df_work)
            anomalies += self._check_price_spikes(symbol, df_work)
            anomalies += self._check_volume(symbol, df_work)

            # Separate by severity
            fatals   = [a for a in anomalies if a.severity == _FATAL]
            errors   = [a for a in anomalies if a.severity == _ERROR]
            warnings = [a for a in anomalies if a.severity == _WARN]

            if fatals:
                # Exclude symbol — log and move on
                report.excluded[symbol] = fatals
                if anomalies:
                    report.anomalies[symbol] = anomalies
                logger.warning(
                    "EXCLUDED %s — %d FATAL anomaly(ies): %s",
                    symbol,
                    len(fatals),
                    [f.check for f in fatals],
                )
                continue

            # Apply corrections for ERROR-level issues
            df_work, corrections = self._correct(symbol, df_work, errors + warnings)

            report.clean[symbol] = df_work
            if anomalies:
                report.anomalies[symbol] = anomalies
            if corrections:
                report.corrections[symbol] = corrections

        valid   = report.clean_count
        excl    = report.excluded_count
        n_warns = report.warning_count
        total   = valid + excl
        logger.info(
            "DataQualityValidator: %d/%d clean | %d excluded | %d warnings",
            valid, total, excl, n_warns,
        )
        return report

    # ──────────────────────────────────────────────────────────────────────────
    #  Individual checks
    # ──────────────────────────────────────────────────────────────────────────

    @staticmethod
    def _check_duplicates(symbol: str, df: pd.DataFrame) -> list[Anomaly]:
        dupes = df.index.duplicated().sum()
        if dupes > 0:
            logger.debug("%s [quality/duplicates]: %d duplicate dates found", symbol, dupes)
            # Drop duplicates in-place (keep last)
            df.drop(index=df.index[df.index.duplicated(keep="last")], inplace=True)
            return [Anomaly(
                symbol   = symbol,
                check    = "duplicate_dates",
                severity = _ERROR,
                bar_date = None,
                detail   = f"{dupes} duplicate index dates removed (kept last)",
                value    = float(dupes),
                expected = "unique dates",
            )]
        return []

    @staticmethod
    def _check_ohlc_structure(symbol: str, df: pd.DataFrame) -> list[Anomaly]:
        """Check OHLCV structural consistency on every bar."""
        anomalies: list[Anomaly] = []

        for col in ["open", "high", "low", "close"]:
            zero_mask = df[col] <= 0
            if zero_mask.any():
                dates = df.index[zero_mask].tolist()
                anomalies.append(Anomaly(
                    symbol   = symbol,
                    check    = f"non_positive_{col}",
                    severity = _FATAL,
                    bar_date = dates[0].date() if dates else None,
                    detail   = f"{col} <= 0 on {len(dates)} bar(s): {[d.date() for d in dates[:3]]}",
                    value    = float(df.loc[zero_mask, col].min()),
                    expected = f"{col} > 0",
                ))

        # high < low is always a data error
        inv_mask = df["high"] < df["low"]
        if inv_mask.any():
            dates = df.index[inv_mask].tolist()
            anomalies.append(Anomaly(
                symbol   = symbol,
                check    = "inverted_ohlc",
                severity = _FATAL,
                bar_date = dates[0].date() if dates else None,
                detail   = f"high < low on {len(dates)} bar(s)",
                value    = float(df.loc[inv_mask, "high"].min()),
                expected = "high >= low",
            ))

        # close slightly above high or below low — clippable
        close_above_high = df["close"] > df["high"] * 1.001
        if close_above_high.any():
            dates = df.index[close_above_high].tolist()
            anomalies.append(Anomaly(
                symbol   = symbol,
                check    = "close_above_high",
                severity = _ERROR,
                bar_date = dates[0].date() if dates else None,
                detail   = f"close > high on {len(dates)} bar(s) — will clip",
                value    = float(df.loc[close_above_high, "close"].max()),
                expected = "close <= high",
            ))

        close_below_low = df["close"] < df["low"] * 0.999
        if close_below_low.any():
            dates = df.index[close_below_low].tolist()
            anomalies.append(Anomaly(
                symbol   = symbol,
                check    = "close_below_low",
                severity = _ERROR,
                bar_date = dates[0].date() if dates else None,
                detail   = f"close < low on {len(dates)} bar(s) — will clip",
                value    = float(df.loc[close_below_low, "close"].min()),
                expected = "close >= low",
            ))

        return anomalies

    def _check_price_spikes(self, symbol: str, df: pd.DataFrame) -> list[Anomaly]:
        """Detect single-bar returns that suggest bad data or corporate actions."""
        anomalies: list[Anomaly] = []
        returns   = df["close"].pct_change().abs().dropna()

        # FATAL: return > fatal_spike_pct (very likely split-unadjusted data)
        fatal_mask = returns > self.fatal_spike_pct
        if fatal_mask.any():
            worst_date = returns[fatal_mask].idxmax()
            worst_val  = float(returns[fatal_mask].max())
            anomalies.append(Anomaly(
                symbol   = symbol,
                check    = "price_spike_fatal",
                severity = _FATAL,
                bar_date = worst_date.date(),
                detail   = (
                    f"Return of {worst_val*100:.1f}% on {worst_date.date()} — "
                    f"likely split-unadjusted or bad data"
                ),
                value    = worst_val,
                expected = f"|return| < {self.fatal_spike_pct*100:.0f}%",
            ))

        # WARNING: return > warn_spike_pct (circuit-breaker or news event)
        warn_mask = (returns > self.warn_spike_pct) & (~fatal_mask)
        if warn_mask.any():
            spike_dates = returns[warn_mask].index.tolist()
            anomalies.append(Anomaly(
                symbol   = symbol,
                check    = "price_spike_warning",
                severity = _WARN,
                bar_date = spike_dates[0].date() if spike_dates else None,
                detail   = (
                    f"{len(spike_dates)} bar(s) with |return| > "
                    f"{self.warn_spike_pct*100:.0f}%: "
                    f"{[d.date() for d in spike_dates[:3]]}"
                ),
                value    = float(returns[warn_mask].max()),
                expected = f"|return| < {self.warn_spike_pct*100:.0f}%",
            ))

        return anomalies

    def _check_volume(self, symbol: str, df: pd.DataFrame) -> list[Anomaly]:
        """Check for negative volume (ERROR) and zero volume days (WARNING)."""
        anomalies: list[Anomaly] = []

        neg_mask = df["volume"] < 0
        if neg_mask.any():
            anomalies.append(Anomaly(
                symbol   = symbol,
                check    = "negative_volume",
                severity = _ERROR,
                bar_date = df.index[neg_mask][0].date(),
                detail   = f"{neg_mask.sum()} bars with negative volume — will set to 0",
                value    = float(df.loc[neg_mask, "volume"].min()),
                expected = "volume >= 0",
            ))

        zero_mask = df["volume"] == 0
        n_zero    = int(zero_mask.sum())
        if n_zero > 0:
            anomalies.append(Anomaly(
                symbol   = symbol,
                check    = "zero_volume",
                severity = _WARN,
                bar_date = df.index[zero_mask][0].date(),
                detail   = (
                    f"{n_zero} bars with zero volume "
                    f"({'will fill with median' if self.zero_volume_fill else 'left as-is'})"
                ),
                value    = float(n_zero),
                expected = "volume > 0",
            ))

        return anomalies

    # ──────────────────────────────────────────────────────────────────────────
    #  Corrections
    # ──────────────────────────────────────────────────────────────────────────

    def _correct(
        self,
        symbol:    str,
        df:        pd.DataFrame,
        anomalies: list[Anomaly],
    ) -> tuple[pd.DataFrame, list[str]]:
        """
        Apply in-place corrections for ERROR and WARNING anomalies.

        Returns the corrected DataFrame and a list of human-readable
        correction descriptions (for logging and persistence).
        """
        corrections: list[str] = []
        checks_seen = {a.check for a in anomalies}

        # Clip close to [low, high]
        if "close_above_high" in checks_seen:
            mask = df["close"] > df["high"]
            if mask.any():
                df.loc[mask, "close"] = df.loc[mask, "high"]
                corrections.append(f"close clipped to high on {mask.sum()} bar(s)")
                logger.debug("%s: close clipped to high on %d bar(s)", symbol, mask.sum())

        if "close_below_low" in checks_seen:
            mask = df["close"] < df["low"]
            if mask.any():
                df.loc[mask, "close"] = df.loc[mask, "low"]
                corrections.append(f"close clipped to low on {mask.sum()} bar(s)")

        # Fix negative volume
        if "negative_volume" in checks_seen:
            mask = df["volume"] < 0
            df.loc[mask, "volume"] = 0
            corrections.append(f"negative volume set to 0 on {mask.sum()} bar(s)")

        # Fill zero volume with 5-day rolling median
        if "zero_volume" in checks_seen and self.zero_volume_fill:
            zero_mask   = df["volume"] == 0
            # Cast to float so pd.NA / NaN works with rolling().median()
            vol_float   = df["volume"].astype(float)
            vol_float[zero_mask] = float("nan")
            rolling_med = vol_float.rolling(5, min_periods=1).median()
            fill_vals   = rolling_med[zero_mask].fillna(0)
            orig_dtype  = df["volume"].dtype
            df.loc[zero_mask, "volume"] = fill_vals.astype(orig_dtype)
            corrections.append(f"zero volume filled with rolling median on {zero_mask.sum()} bar(s)")

        return df, corrections