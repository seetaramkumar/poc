"""
stock_regime/src/indicators.py
================================
Responsible for taking a normalised OHLCV DataFrame and producing a
StockIndicatorSnapshot for the most-recent (or any requested) bar.

This is the only module that touches pandas-ta.  All indicator periods
are driven by StockEngineConfig — no hardcoded numbers here.

Relative Strength
-----------------
RS is computed as the ratio of cumulative returns over rs_period bars:
    RS = (stock_return_factor) / (benchmark_return_factor)

where return_factor = close[-1] / close[-rs_period].

    RS > 1  → stock is outperforming the benchmark
    RS < 1  → stock is underperforming the benchmark
    RS = None → no benchmark provided; RS signals will be silently disabled
"""

from __future__ import annotations

import math
import warnings
from typing import Optional

import pandas as pd
import pandas_ta as ta

from .config_loader import StockEngineConfig
from .models import StockIndicatorSnapshot


class StockIndicatorCalculator:
    """
    Calculates all technical indicators required by the Stock Regime Engine.

    Parameters
    ----------
    config : StockEngineConfig
    """

    def __init__(self, config: StockEngineConfig) -> None:
        self.cfg = config

    # ──────────────────────────────────────────────────────────────
    #  Public API
    # ──────────────────────────────────────────────────────────────

    def compute(
        self,
        df: pd.DataFrame,
        row_index: int = -1,
        benchmark_df: Optional[pd.DataFrame] = None,
    ) -> StockIndicatorSnapshot:
        """
        Compute all indicators and return a snapshot for ``row_index``.

        Parameters
        ----------
        df :
            Normalised OHLCV DataFrame with columns:
            ``open, high, low, close, volume`` (case-insensitive).
        row_index :
            Which row to extract.  Default -1 → last row.
        benchmark_df :
            Optional benchmark OHLCV DataFrame (same schema).
            Used to compute relative strength.  When ``None``, RS is
            left as ``None`` and RS-based signals are disabled.

        Returns
        -------
        StockIndicatorSnapshot
        """
        df = self._normalise_columns(df)
        self._validate(df)

        enriched = self._add_indicators(df)
        rs = self._compute_rs(df, benchmark_df, row_index)

        return self._extract_snapshot(enriched, row_index, rs)

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
        """Append all indicator columns onto a copy of df."""
        cfg = self.cfg
        thr = cfg.thresholds

        # ── EMAs ────────────────────────────────────────────────────
        df[f"ema{cfg.ema_fast}"]  = ta.ema(df["close"], length=cfg.ema_fast)
        df[f"ema{cfg.ema_mid}"]   = ta.ema(df["close"], length=cfg.ema_mid)
        df[f"ema{cfg.ema_slow}"]  = ta.ema(df["close"], length=cfg.ema_slow)

        # ── EMA slopes (% change over slope_window bars) ─────────────
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

        # ── Rolling high (52-week proxy) ────────────────────────────
        # Rolling max of close over high_period bars — used to detect
        # whether price is near a significant resistance / breakout level.
        df["high_52w"] = df["close"].rolling(window=cfg.high_period, min_periods=50).max()

        return df

    def _compute_rs(
        self,
        df: pd.DataFrame,
        benchmark_df: Optional[pd.DataFrame],
        row_index: int,
    ) -> Optional[float]:
        """
        Compute relative strength at ``row_index`` vs the benchmark.

        RS = (close[row] / close[row - rs_period]) /
             (bench_close[row] / bench_close[row - rs_period])

        Returns ``None`` when benchmark is absent or history is too short.
        """
        if benchmark_df is None:
            return None

        rs_period = self.cfg.rs_period

        # Align absolute positions: row_index may be negative
        n_stock = len(df)
        abs_idx  = row_index if row_index >= 0 else n_stock + row_index

        if abs_idx < rs_period:
            return None  # not enough history for this stock

        # Align benchmark to stock's index (inner join on date)
        bench = benchmark_df.copy()
        bench.columns = [c.lower().strip() for c in bench.columns]
        if "close" not in bench.columns:
            return None

        # Use integer location, not date alignment, to avoid join complexity.
        # Both frames are normalised and sorted ascending by the data layer.
        n_bench   = len(bench)
        abs_bench = min(abs_idx, n_bench - 1)

        if abs_bench < rs_period:
            return None

        stock_now  = df["close"].iloc[abs_idx]
        stock_prev = df["close"].iloc[abs_idx - rs_period]
        bench_now  = bench["close"].iloc[abs_bench]
        bench_prev = bench["close"].iloc[abs_bench - rs_period]

        # Guard against zero prices (bad data)
        if stock_prev <= 0 or bench_prev <= 0:
            return None

        stock_ret = stock_now / stock_prev
        bench_ret = bench_now / bench_prev

        if bench_ret <= 0:
            return None

        return round(stock_ret / bench_ret, 4)

    def _extract_snapshot(
        self,
        df: pd.DataFrame,
        row_index: int,
        rs: Optional[float],
    ) -> StockIndicatorSnapshot:
        """Pull the requested row into a StockIndicatorSnapshot."""
        cfg = self.cfg
        row = df.iloc[row_index]

        def _val(col: str) -> Optional[float]:
            """Return float or None for NaN / missing values."""
            v = row.get(col, float("nan"))
            if v is None:
                return None
            try:
                fv = float(v)
                return None if math.isnan(fv) else fv
            except (TypeError, ValueError):
                return None

        return StockIndicatorSnapshot(
            close        = float(row["close"]),
            ema20        = _val(f"ema{cfg.ema_fast}"),
            ema50        = _val(f"ema{cfg.ema_mid}"),
            ema200       = _val(f"ema{cfg.ema_slow}"),
            ema20_slope  = _val("ema20_slope"),
            ema50_slope  = _val("ema50_slope"),
            adx          = _val("adx"),
            atr          = _val("atr"),
            atr_ma       = _val("atr_ma"),
            volume       = _val("volume"),
            volume_ma    = _val("volume_ma"),
            relative_strength = rs,
            high_52w     = _val("high_52w"),
        )
