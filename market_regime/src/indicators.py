"""
indicators.py — Technical indicator computation
================================================
Responsible for taking a raw OHLCV DataFrame and producing an
IndicatorSnapshot for the most-recent bar (or any requested row).

Dependencies: pandas, numpy only — no third-party TA library required.
All indicators are implemented from their canonical definitions so the
engine works on Python 3.11+ without any TA-library version conflicts.

All periods are read from EngineConfig — no hard-coded numbers here.
"""

from __future__ import annotations

import math
import warnings
from typing import Optional

import numpy as np
import pandas as pd

from .config_loader import EngineConfig
from .models import IndicatorSnapshot


class IndicatorCalculator:
    """
    Calculates all technical indicators required by the regime engine
    using pure pandas / numpy — no external TA library needed.

    Indicators implemented
    ----------------------
    - EMA (Exponential Moving Average)
    - ATR (Average True Range)  — Wilder smoothing
    - ADX (Average Directional Index) — Wilder smoothing
    - EMA slope  (% change over N bars)
    - Volume MA  (simple rolling mean)
    - ATR MA     (simple rolling mean of ATR)

    Usage
    -----
    calc     = IndicatorCalculator(config)
    snapshot = calc.compute(ohlcv_df)
    """

    def __init__(self, config: EngineConfig) -> None:
        self.cfg = config

    # ──────────────────────────────────────────────────────────────
    #  Public API
    # ──────────────────────────────────────────────────────────────

    def compute(self, df: pd.DataFrame, row_index: int = -1) -> IndicatorSnapshot:
        """
        Compute all indicators and return a snapshot for ``row_index``.

        Parameters
        ----------
        df : pd.DataFrame
            OHLCV data with columns: open, high, low, close, volume
            (case-insensitive).
        row_index : int
            Row to snapshot.  Default -1 → most recent bar.

        Returns
        -------
        IndicatorSnapshot
        """
        df = self._normalise_columns(df)
        self._validate_dataframe(df)
        enriched = self._add_all_indicators(df)
        return self._extract_snapshot(enriched, row_index)

    # ──────────────────────────────────────────────────────────────
    #  Private — column normalisation & validation
    # ──────────────────────────────────────────────────────────────

    @staticmethod
    def _normalise_columns(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df.columns = [c.lower() for c in df.columns]
        return df

    @staticmethod
    def _validate_dataframe(df: pd.DataFrame) -> None:
        required = {"open", "high", "low", "close", "volume"}
        missing  = required - set(df.columns)
        if missing:
            raise ValueError(f"DataFrame is missing columns: {missing}")
        if len(df) < 210:
            warnings.warn(
                f"Only {len(df)} rows supplied. "
                "EMA-200 needs ≥ 200 bars; early bars will return NaN.",
                UserWarning,
                stacklevel=4,
            )

    # ──────────────────────────────────────────────────────────────
    #  Private — indicator implementations
    # ──────────────────────────────────────────────────────────────

    @staticmethod
    def _ema(series: pd.Series, period: int) -> pd.Series:
        """
        Standard EMA using pandas ewm (equivalent to pandas-ta ema).
        adjust=False matches the classic recursive EMA formula.
        """
        return series.ewm(span=period, adjust=False).mean()

    @staticmethod
    def _atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int) -> pd.Series:
        """
        Average True Range using Wilder smoothing (RMA).

        True Range = max(H-L, |H-Cprev|, |L-Cprev|)
        ATR        = Wilder RMA of TR over `period` bars
        """
        prev_close = close.shift(1)
        tr = pd.concat([
            high - low,
            (high - prev_close).abs(),
            (low  - prev_close).abs(),
        ], axis=1).max(axis=1)

        # Wilder smoothing: equivalent to EMA with alpha = 1/period
        return tr.ewm(alpha=1.0 / period, adjust=False).mean()

    @staticmethod
    def _adx(
        high: pd.Series,
        low:  pd.Series,
        close: pd.Series,
        period: int,
    ) -> pd.Series:
        """
        Average Directional Index (Wilder, 1978).

        Steps
        -----
        1.  +DM / -DM  (directional movement)
        2.  Smooth with Wilder RMA → +DI / -DI
        3.  DX  = 100 * |+DI - -DI| / (+DI + -DI)
        4.  ADX = Wilder RMA of DX
        """
        alpha     = 1.0 / period
        prev_high = high.shift(1)
        prev_low  = low.shift(1)
        prev_close = close.shift(1)

        # True Range (reused for DI scaling)
        tr = pd.concat([
            high - low,
            (high - prev_close).abs(),
            (low  - prev_close).abs(),
        ], axis=1).max(axis=1)

        # Raw directional movement
        up   = high - prev_high
        down = prev_low - low

        plus_dm  = np.where((up > down) & (up > 0),  up,   0.0)
        minus_dm = np.where((down > up) & (down > 0), down, 0.0)

        plus_dm_s  = pd.Series(plus_dm,  index=high.index).ewm(alpha=alpha, adjust=False).mean()
        minus_dm_s = pd.Series(minus_dm, index=high.index).ewm(alpha=alpha, adjust=False).mean()
        atr_s      = tr.ewm(alpha=alpha, adjust=False).mean()

        # +DI / -DI as percentages
        plus_di  = 100 * plus_dm_s  / atr_s.replace(0, np.nan)
        minus_di = 100 * minus_dm_s / atr_s.replace(0, np.nan)

        # DX then ADX
        di_sum  = (plus_di + minus_di).replace(0, np.nan)
        dx      = 100 * (plus_di - minus_di).abs() / di_sum
        adx     = dx.ewm(alpha=alpha, adjust=False).mean()

        return adx

    # ──────────────────────────────────────────────────────────────
    #  Private — assemble all indicator columns
    # ──────────────────────────────────────────────────────────────

    def _add_all_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        cfg = self.cfg
        thr = self.cfg.thresholds

        # ── EMAs ──────────────────────────────────────────────────
        df[f"ema{cfg.ema_fast}"] = self._ema(df["close"], cfg.ema_fast)
        df[f"ema{cfg.ema_mid}"]  = self._ema(df["close"], cfg.ema_mid)
        df[f"ema{cfg.ema_slow}"] = self._ema(df["close"], cfg.ema_slow)

        # ── EMA slopes (% change over N bars) ─────────────────────
        slope_w = int(thr.ema_slope_window)
        df["ema20_slope"] = df[f"ema{cfg.ema_fast}"].pct_change(slope_w)
        df["ema50_slope"] = df[f"ema{cfg.ema_mid}"].pct_change(slope_w)

        # ── ADX ───────────────────────────────────────────────────
        df["adx"] = self._adx(df["high"], df["low"], df["close"], cfg.adx_period)

        # ── ATR + ATR moving average ───────────────────────────────
        df["atr"]    = self._atr(df["high"], df["low"], df["close"], cfg.atr_period)
        df["atr_ma"] = df["atr"].rolling(window=int(thr.atr_ma_period)).mean()

        # ── Volume MA ─────────────────────────────────────────────
        df["volume_ma"] = df["volume"].rolling(window=cfg.volume_ma_period).mean()

        return df

    # ──────────────────────────────────────────────────────────────
    #  Private — snapshot extraction
    # ──────────────────────────────────────────────────────────────

    def _extract_snapshot(self, df: pd.DataFrame, row_index: int) -> IndicatorSnapshot:
        cfg = self.cfg
        row = df.iloc[row_index]

        def _val(col: str) -> Optional[float]:
            v = row.get(col, float("nan"))
            if v is None:
                return None
            try:
                f = float(v)
                return None if math.isnan(f) else f
            except (TypeError, ValueError):
                return None

        return IndicatorSnapshot(
            close       = float(row["close"]),
            ema20       = _val(f"ema{cfg.ema_fast}"),
            ema50       = _val(f"ema{cfg.ema_mid}"),
            ema200      = _val(f"ema{cfg.ema_slow}"),
            ema20_slope = _val("ema20_slope"),
            ema50_slope = _val("ema50_slope"),
            adx         = _val("adx"),
            atr         = _val("atr"),
            atr_ma      = _val("atr_ma"),
            volume      = _val("volume"),
            volume_ma   = _val("volume_ma"),
        )
