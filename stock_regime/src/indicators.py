"""
stock_regime/src/indicators.py
================================
Computes all technical indicators from a normalised OHLCV DataFrame.

Changes from previous version
------------------------------
- Added ROC-10, ROC-21, acceleration (momentum separation)
- Added multi-period RS: rs_1m (21-bar), rs_3m (63-bar), rs_6m (126-bar)
- Added rs_trend: rolling slope of rs_3m over rs_trend_window bars
- Added higher_highs_count: trend quality metric
- Added ema_distance_pct: normalised distance of close from EMA-200
- legacy relative_strength field still populated from rs_3m for compatibility
"""

from __future__ import annotations

import math
import warnings
from typing import Optional

import numpy as np
import pandas as pd
import pandas_ta as ta

from .config_loader import StockEngineConfig
from .models import StockIndicatorSnapshot

import logging
logger = logging.getLogger(__name__)


class StockIndicatorCalculator:
    """Calculates all technical indicators for the Stock Regime Engine."""

    def __init__(self, config: StockEngineConfig) -> None:
        self.cfg = config

    def compute(
        self,
        df: pd.DataFrame,
        row_index: int = -1,
        benchmark_df: Optional[pd.DataFrame] = None,
    ) -> StockIndicatorSnapshot:
        df = self._normalise_columns(df)
        self._validate(df)
        enriched = self._add_indicators(df)
        rs_profile = self._compute_rs_profile(df, benchmark_df, row_index)
        return self._extract_snapshot(enriched, row_index, rs_profile)

    # ──────────────────────────────────────────────────────────────
    #  Private helpers
    # ──────────────────────────────────────────────────────────────

    @staticmethod
    def _normalise_columns(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df.columns = [c.lower().strip() for c in df.columns]
        return df

    def _validate(self, df: pd.DataFrame) -> None:
        required = {"open", "high", "low", "close", "volume"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"DataFrame is missing required columns: {missing}")

        min_bars = max(self.cfg.ema_slow, self.cfg.high_period) + 30
        if len(df) < min_bars:
            warnings.warn(
                f"Only {len(df)} rows supplied; {min_bars} recommended. "
                "Some indicators may be NaN.",
                UserWarning,
                stacklevel=5,
            )

    def _add_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        cfg = self.cfg
        thr = cfg.thresholds

        # ── EMAs ────────────────────────────────────────────────────
        df[f"ema{cfg.ema_fast}"]  = ta.ema(df["close"], length=cfg.ema_fast)
        df[f"ema{cfg.ema_mid}"]   = ta.ema(df["close"], length=cfg.ema_mid)
        df[f"ema{cfg.ema_slow}"]  = ta.ema(df["close"], length=cfg.ema_slow)

        # ── EMA slopes ──────────────────────────────────────────────
        w = int(thr.ema_slope_window)
        df["ema20_slope"] = df[f"ema{cfg.ema_fast}"].pct_change(w)
        df["ema50_slope"] = df[f"ema{cfg.ema_mid}"].pct_change(w)

        # ── ADX ─────────────────────────────────────────────────────
        adx_df  = ta.adx(df["high"], df["low"], df["close"], length=cfg.adx_period)
        adx_col = [c for c in adx_df.columns if c.startswith("ADX_")]
        df["adx"] = adx_df[adx_col[0]] if adx_col else float("nan")

        # ── ATR ─────────────────────────────────────────────────────
        df["atr"]    = ta.atr(df["high"], df["low"], df["close"], length=cfg.atr_period)
        df["atr_ma"] = df["atr"].rolling(window=int(thr.atr_ma_period)).mean()

        # ── Volume MA ───────────────────────────────────────────────
        df["volume_ma"] = df["volume"].rolling(window=cfg.volume_ma_period).mean()

        # ── Rolling high ────────────────────────────────────────────
        df["high_52w"] = df["close"].rolling(window=cfg.high_period, min_periods=50).max()

        # ── NEW: Rate of Change ──────────────────────────────────────
        # ROC = (close / close[n bars ago] - 1) × 100  (percentage)
        df["roc_10"] = df["close"].pct_change(10) * 100
        df["roc_21"] = df["close"].pct_change(21) * 100
        # Acceleration: positive = momentum building, negative = decelerating
        df["acceleration"] = df["roc_10"] - df["roc_21"]

        # ── NEW: EMA distance from macro trend ───────────────────────
        ema200_col = f"ema{cfg.ema_slow}"
        ema200     = df[ema200_col]
        df["ema_distance_pct"] = (df["close"] - ema200) / ema200.replace(0, pd.NA)

        # ── NEW: Higher highs count (trend quality) ──────────────────
        # Count how many of the last higher_highs_window bars made a new
        # rolling high relative to the bar before them.
        hh_window = int(getattr(thr, "higher_highs_window", 20))
        rolling_high_prev = df["high"].shift(1).rolling(hh_window, min_periods=5).max()
        df["is_higher_high"] = (df["high"] > rolling_high_prev).astype(int)
        df["higher_highs_count"] = (
            df["is_higher_high"].rolling(hh_window, min_periods=5).sum()
        )

        return df

    # ──────────────────────────────────────────────────────────────
    #  Multi-period RS profile
    # ──────────────────────────────────────────────────────────────

    def _compute_rs_profile(
        self,
        df: pd.DataFrame,
        benchmark_df: Optional[pd.DataFrame],
        row_index: int,
    ) -> dict[str, Optional[float]]:
        """
        Compute RS at three lookback periods and a rolling RS trend slope.

        Returns dict with keys: rs_1m, rs_3m, rs_6m, rs_trend.
        All None when no benchmark provided.
        """
        if benchmark_df is None:
            return {"rs_1m": None, "rs_3m": None, "rs_6m": None, "rs_trend": None}

        bench = benchmark_df.copy()
        bench.columns = [c.lower().strip() for c in bench.columns]
        if "close" not in bench.columns:
            return {"rs_1m": None, "rs_3m": None, "rs_6m": None, "rs_trend": None}

        n_stock = len(df)
        abs_idx  = row_index if row_index >= 0 else n_stock + row_index

        periods  = {"rs_1m": 21, "rs_3m": 63, "rs_6m": 126}
        result: dict[str, Optional[float]] = {}

        for key, period in periods.items():
            result[key] = self._single_rs(df, bench, abs_idx, period)

        # RS trend: linear slope of rs_3m over last rs_trend_window bars
        tw = int(getattr(self.cfg.indicators, "rs_trend_window", 10))
        result["rs_trend"] = self._rs_slope(df, bench, abs_idx, period=63, window=tw)

        return result

    def _single_rs(
        self,
        stock_df: pd.DataFrame,
        bench_df: pd.DataFrame,
        abs_idx: int,
        period: int,
    ) -> Optional[float]:
        """RS = (stock_return / bench_return) over `period` bars."""
        if abs_idx < period:
            return None
        abs_bench = min(abs_idx, len(bench_df) - 1)
        if abs_bench < period:
            return None

        stock_now   = stock_df["close"].iloc[abs_idx]
        stock_prev  = stock_df["close"].iloc[abs_idx - period]
        bench_now   = bench_df["close"].iloc[abs_bench]
        bench_prev  = bench_df["close"].iloc[abs_bench - period]

        if stock_prev <= 0 or bench_prev <= 0 or bench_now <= 0:
            return None

        stock_ret = stock_now / stock_prev
        bench_ret = bench_now / bench_prev
        if bench_ret <= 0:
            return None
        return round(stock_ret / bench_ret, 4)

    def _rs_slope(
        self,
        stock_df: pd.DataFrame,
        bench_df: pd.DataFrame,
        abs_idx: int,
        period: int,
        window: int,
    ) -> Optional[float]:
        """
        Compute the linear slope of rs_3m over the last `window` bars.

        Positive → RS improving (stock outperforming more over time)
        Negative → RS weakening
        """
        rs_series = []
        for lookback in range(window - 1, -1, -1):
            idx = abs_idx - lookback
            if idx < 0:
                continue
            rs = self._single_rs(stock_df, bench_df, idx, period)
            if rs is not None:
                rs_series.append(rs)

        if len(rs_series) < max(window // 2, 3):
            return None

        n      = len(rs_series)
        xs     = list(range(n))
        mean_x = sum(xs) / n
        mean_y = sum(rs_series) / n
        denom  = sum((x - mean_x) ** 2 for x in xs)
        if denom == 0:
            return 0.0
        slope  = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, rs_series)) / denom
        return round(slope, 6)

    # ──────────────────────────────────────────────────────────────
    #  Snapshot extraction
    # ──────────────────────────────────────────────────────────────

    def _extract_snapshot(
        self,
        df: pd.DataFrame,
        row_index: int,
        rs_profile: dict,
    ) -> StockIndicatorSnapshot:
        cfg = self.cfg
        row = df.iloc[row_index]

        def _val(col: str) -> Optional[float]:
            v = row.get(col, float("nan"))
            if v is None:
                return None
            try:
                fv = float(v)
                return None if math.isnan(fv) else fv
            except (TypeError, ValueError):
                return None

        def _int_val(col: str) -> Optional[int]:
            v = _val(col)
            return int(v) if v is not None else None

        rs_3m = rs_profile.get("rs_3m")

        return StockIndicatorSnapshot(
            close           = float(row["close"]),
            ema20           = _val(f"ema{cfg.ema_fast}"),
            ema50           = _val(f"ema{cfg.ema_mid}"),
            ema200          = _val(f"ema{cfg.ema_slow}"),
            ema20_slope     = _val("ema20_slope"),
            ema50_slope     = _val("ema50_slope"),
            adx             = _val("adx"),
            atr             = _val("atr"),
            atr_ma          = _val("atr_ma"),
            volume          = _val("volume"),
            volume_ma       = _val("volume_ma"),
            relative_strength = rs_3m,   # backward-compatible alias
            high_52w        = _val("high_52w"),
            # New
            roc_10          = _val("roc_10"),
            roc_21          = _val("roc_21"),
            acceleration    = _val("acceleration"),
            rs_1m           = rs_profile.get("rs_1m"),
            rs_3m           = rs_3m,
            rs_6m           = rs_profile.get("rs_6m"),
            rs_trend        = rs_profile.get("rs_trend"),
            higher_highs_count = _int_val("higher_highs_count"),
            ema_distance_pct   = _val("ema_distance_pct"),
        )