"""
stock_regime/src/indicators.py
================================
Computes all technical indicators from a normalised OHLCV DataFrame.

Changes in this version
------------------------
Phase 1 — Volatility fix:
  candle_instability    : mean |return| / ATR ratio (erratic = high)
  reversal_frequency    : fraction of bars where direction reversed
  gap_frequency         : fraction of bars with open gap > threshold
  wickiness_score       : mean wick-to-range ratio (indecision / rejection)

Phase 2 — Range detection:
  bb_width              : Bollinger Band width (close normalised)
  directional_efficiency: ROC over N bars / sum of |daily returns| (0=ranging, 1=trending)
  ema_spread            : (ema20 - ema50) / close — compressed when near 0
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
        enriched   = self._add_indicators(df)
        rs_profile = self._compute_rs_profile(df, benchmark_df, row_index)
        return self._extract_snapshot(enriched, row_index, rs_profile)

    @staticmethod
    def _normalise_columns(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df.columns = [c.lower().strip() for c in df.columns]
        return df

    def _validate(self, df: pd.DataFrame) -> None:
        required = {"open", "high", "low", "close", "volume"}
        missing  = required - set(df.columns)
        if missing:
            raise ValueError(f"DataFrame is missing required columns: {missing}")
        min_bars = max(self.cfg.ema_slow, self.cfg.high_period) + 30
        if len(df) < min_bars:
            warnings.warn(
                f"Only {len(df)} rows supplied; {min_bars} recommended. "
                "Some indicators may be NaN.",
                UserWarning, stacklevel=5,
            )

    def _add_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        cfg = self.cfg
        thr = cfg.thresholds

        # ── EMAs ────────────────────────────────────────────────────
        df[f"ema{cfg.ema_fast}"]  = ta.ema(df["close"], length=cfg.ema_fast)
        df[f"ema{cfg.ema_mid}"]   = ta.ema(df["close"], length=cfg.ema_mid)
        df[f"ema{cfg.ema_slow}"]  = ta.ema(df["close"], length=cfg.ema_slow)

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

        # ── ROC / acceleration ──────────────────────────────────────
        df["roc_10"]      = df["close"].pct_change(10) * 100
        df["roc_21"]      = df["close"].pct_change(21) * 100
        df["acceleration"]= df["roc_10"] - df["roc_21"]

        # ── EMA distance ────────────────────────────────────────────
        ema200 = df[f"ema{cfg.ema_slow}"]
        df["ema_distance_pct"] = (df["close"] - ema200) / ema200.replace(0, np.nan)

        # ── Higher highs ────────────────────────────────────────────
        hh_window = int(getattr(thr, "higher_highs_window", 20))
        rolling_high_prev = df["high"].shift(1).rolling(hh_window, min_periods=5).max()
        df["is_higher_high"]      = (df["high"] > rolling_high_prev).astype(int)
        df["higher_highs_count"]  = df["is_higher_high"].rolling(hh_window, min_periods=5).sum()

        # ────────────────────────────────────────────────────────────
        # PHASE 1: Volatility quality indicators
        # ────────────────────────────────────────────────────────────
        vi_window = int(getattr(thr, "volatility_instability_window", 20))
        returns   = df["close"].pct_change()

        # candle_instability: mean |return| relative to ATR baseline.
        # A strong trend has big moves in ONE direction → low instability.
        # An erratic market has big moves in BOTH directions → high.
        abs_ret = returns.abs()
        _atr_pct = (df["atr"] / df["close"].replace(0, np.nan)).astype(float)
        _atr_pct_ma = _atr_pct.rolling(vi_window, min_periods=5).mean()
        df["candle_instability"] = (
            abs_ret.astype(float).rolling(vi_window, min_periods=5).mean() /
            _atr_pct_ma.replace(0, np.nan)
        ).astype(float)

        # reversal_frequency: fraction of bars where daily direction flipped.
        # Trending stocks have low reversal_frequency.
        direction    = returns.apply(lambda x: 1 if x > 0 else (-1 if x < 0 else 0))
        dir_change   = (direction != direction.shift(1)).astype(int)
        df["reversal_frequency"] = dir_change.rolling(vi_window, min_periods=5).mean()

        # gap_frequency: fraction of bars where |open - prev_close| > gap threshold.
        gap_thr = float(getattr(thr, "gap_threshold_pct", 0.01))
        prev_close = df["close"].shift(1)
        gap_pct    = ((df["open"] - prev_close) / prev_close.replace(0, np.nan)).abs().astype(float)
        df["gap_frequency"] = (gap_pct > gap_thr).astype(float).rolling(
            vi_window, min_periods=5
        ).mean()

        # wickiness_score: mean (high - low - |close - open|) / (high - low + 1e-9)
        # High wicks = rejection / indecision — hallmark of unstable markets.
        body     = (df["close"] - df["open"]).abs()
        total    = (df["high"] - df["low"]).replace(0, np.nan).astype(float)
        wick_pct = ((total - body) / total).astype(float)
        df["wickiness_score"] = wick_pct.rolling(vi_window, min_periods=5).mean()

        # ────────────────────────────────────────────────────────────
        # PHASE 2: Range detection indicators
        # ────────────────────────────────────────────────────────────
        bb_period = int(getattr(thr, "bb_period", 20))
        bb_std    = float(getattr(thr, "bb_std", 2.0))

        # Bollinger Band width: (upper - lower) / middle
        bb_mid   = df["close"].rolling(bb_period, min_periods=10).mean()
        bb_s     = df["close"].rolling(bb_period, min_periods=10).std()
        bb_upper = bb_mid + bb_std * bb_s
        bb_lower = bb_mid - bb_std * bb_s
        df["bb_width"] = (bb_upper - bb_lower) / bb_mid.replace(0, np.nan)

        # Directional efficiency ratio (DER): net ROC / cumulative |returns|
        # 1.0 = perfectly trending; 0.0 = perfectly ranging
        der_window = int(getattr(thr, "der_window", 14))
        net_move   = (df["close"] - df["close"].shift(der_window)).abs()
        path_len   = abs_ret.rolling(der_window, min_periods=5).sum() * df["close"].shift(der_window)
        df["directional_efficiency"] = (net_move.astype(float) / path_len.replace(0, np.nan).astype(float)).clip(0, 1)

        # EMA spread compression: (ema20 - ema50) / close
        ema20 = df[f"ema{cfg.ema_fast}"]
        ema50 = df[f"ema{cfg.ema_mid}"]
        df["ema_spread"] = (ema20 - ema50) / df["close"].replace(0, np.nan)

        return df

    # ── Multi-period RS ──────────────────────────────────────────────

    def _compute_rs_profile(self, df, benchmark_df, row_index):
        if benchmark_df is None:
            return {"rs_1m": None, "rs_3m": None, "rs_6m": None, "rs_trend": None}
        bench = benchmark_df.copy()
        bench.columns = [c.lower().strip() for c in bench.columns]
        if "close" not in bench.columns:
            return {"rs_1m": None, "rs_3m": None, "rs_6m": None, "rs_trend": None}
        n      = len(df)
        abs_idx= row_index if row_index >= 0 else n + row_index
        result = {}
        for key, period in {"rs_1m": 21, "rs_3m": 63, "rs_6m": 126}.items():
            result[key] = self._single_rs(df, bench, abs_idx, period)
        tw = int(getattr(self.cfg.indicators, "rs_trend_window", 10))
        result["rs_trend"] = self._rs_slope(df, bench, abs_idx, 63, tw)
        return result

    def _single_rs(self, stock_df, bench_df, abs_idx, period):
        if abs_idx < period: return None
        ab = min(abs_idx, len(bench_df) - 1)
        if ab < period: return None
        sn, sp = stock_df["close"].iloc[abs_idx], stock_df["close"].iloc[abs_idx - period]
        bn, bp = bench_df["close"].iloc[ab],      bench_df["close"].iloc[ab - period]
        if sp <= 0 or bp <= 0 or bn <= 0: return None
        br = bn / bp
        return round((sn / sp) / br, 4) if br > 0 else None

    def _rs_slope(self, stock_df, bench_df, abs_idx, period, window):
        rs_series = []
        for lb in range(window - 1, -1, -1):
            idx = abs_idx - lb
            if idx >= 0:
                rs = self._single_rs(stock_df, bench_df, idx, period)
                if rs is not None:
                    rs_series.append(rs)
        if len(rs_series) < max(window // 2, 3): return None
        n      = len(rs_series)
        xs     = list(range(n))
        mx, my = sum(xs)/n, sum(rs_series)/n
        denom  = sum((x - mx)**2 for x in xs)
        return round(sum((x-mx)*(y-my) for x,y in zip(xs,rs_series))/denom, 6) if denom else 0.0

    # ── Snapshot extraction ──────────────────────────────────────────

    def _extract_snapshot(self, df, row_index, rs_profile):
        cfg = self.cfg
        row = df.iloc[row_index]

        def _val(col):
            v = row.get(col, float("nan"))
            if v is None: return None
            try:
                fv = float(v)
                return None if math.isnan(fv) else fv
            except: return None

        def _int_val(col):
            v = _val(col)
            return int(v) if v is not None else None

        rs_3m = rs_profile.get("rs_3m")
        return StockIndicatorSnapshot(
            close                = float(row["close"]),
            ema20                = _val(f"ema{cfg.ema_fast}"),
            ema50                = _val(f"ema{cfg.ema_mid}"),
            ema200               = _val(f"ema{cfg.ema_slow}"),
            ema20_slope          = _val("ema20_slope"),
            ema50_slope          = _val("ema50_slope"),
            adx                  = _val("adx"),
            atr                  = _val("atr"),
            atr_ma               = _val("atr_ma"),
            volume               = _val("volume"),
            volume_ma            = _val("volume_ma"),
            relative_strength    = rs_3m,
            high_52w             = _val("high_52w"),
            roc_10               = _val("roc_10"),
            roc_21               = _val("roc_21"),
            acceleration         = _val("acceleration"),
            rs_1m                = rs_profile.get("rs_1m"),
            rs_3m                = rs_3m,
            rs_6m                = rs_profile.get("rs_6m"),
            rs_trend             = rs_profile.get("rs_trend"),
            higher_highs_count   = _int_val("higher_highs_count"),
            ema_distance_pct     = _val("ema_distance_pct"),
            # Phase 1 — volatility instability
            candle_instability   = _val("candle_instability"),
            reversal_frequency   = _val("reversal_frequency"),
            gap_frequency        = _val("gap_frequency"),
            wickiness_score      = _val("wickiness_score"),
            # Phase 2 — range detection
            bb_width             = _val("bb_width"),
            directional_efficiency = _val("directional_efficiency"),
            ema_spread           = _val("ema_spread"),
        )