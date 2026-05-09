"""
indicators.py — Technical indicator computation
================================================
Responsible for taking a raw OHLCV DataFrame and producing an
IndicatorSnapshot for the most-recent bar (or any requested row).

Dependencies: pandas, pandas-ta
All periods are read from EngineConfig — no hard-coded numbers here.
"""

from __future__ import annotations

import math
import warnings
from typing import Optional

import pandas as pd
import pandas_ta as ta

from .config_loader import EngineConfig
from .models import IndicatorSnapshot


class IndicatorCalculator:
    """
    Calculates all technical indicators required by the regime engine.

    Usage
    -----
    calc = IndicatorCalculator(config)
    snapshot = calc.compute(ohlcv_df)

    Parameters
    ----------
    config : EngineConfig
        Loaded configuration (periods, thresholds).
    """

    def __init__(self, config: EngineConfig) -> None:
        self.cfg = config

    # ──────────────────────────────────────────────────────────────
    #  Public API
    # ──────────────────────────────────────────────────────────────

    def compute(self, df: pd.DataFrame, row_index: int = -1) -> IndicatorSnapshot:
        """
        Compute all indicators and return a snapshot for `row_index`.

        Parameters
        ----------
        df : pd.DataFrame
            Must contain columns: open, high, low, close, volume
            (case-insensitive; will be lower-cased internally).
        row_index : int
            Which row to extract the snapshot from.  Default -1 → last row.

        Returns
        -------
        IndicatorSnapshot
        """
        df = self._normalise_columns(df)
        self._validate_dataframe(df)

        enriched = self._add_all_indicators(df)
        return self._extract_snapshot(enriched, row_index)

    # ──────────────────────────────────────────────────────────────
    #  Private helpers
    # ──────────────────────────────────────────────────────────────

    @staticmethod
    def _normalise_columns(df: pd.DataFrame) -> pd.DataFrame:
        """Lower-case all column names for uniform access."""
        df = df.copy()
        df.columns = [c.lower() for c in df.columns]
        return df

    @staticmethod
    def _validate_dataframe(df: pd.DataFrame) -> None:
        """Raise if mandatory OHLCV columns are missing."""
        required = {"open", "high", "low", "close", "volume"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"DataFrame is missing columns: {missing}")
        if len(df) < 210:
            warnings.warn(
                f"Only {len(df)} rows supplied. "
                "EMA-200 needs ≥ 200 bars; some indicators may be NaN.",
                UserWarning,
                stacklevel=4,
            )

    def _add_all_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Append every indicator column onto a copy of df."""
        cfg  = self.cfg
        thr  = self.cfg.thresholds

        # ── EMAs ──────────────────────────────────────────────────
        df[f"ema{cfg.ema_fast}"]  = ta.ema(df["close"], length=cfg.ema_fast)
        df[f"ema{cfg.ema_mid}"]   = ta.ema(df["close"], length=cfg.ema_mid)
        df[f"ema{cfg.ema_slow}"]  = ta.ema(df["close"], length=cfg.ema_slow)

        # ── EMA slopes (% change over N bars) ─────────────────────
        slope_window = int(thr.ema_slope_window)
        df["ema20_slope"] = (
            df[f"ema{cfg.ema_fast}"].pct_change(slope_window)
        )
        df["ema50_slope"] = (
            df[f"ema{cfg.ema_mid}"].pct_change(slope_window)
        )

        # ── ADX ───────────────────────────────────────────────────
        adx_df = ta.adx(
            df["high"], df["low"], df["close"],
            length=cfg.adx_period
        )
        # pandas-ta returns a DataFrame; grab the ADX column
        adx_col = [c for c in adx_df.columns if c.startswith("ADX_")]
        if adx_col:
            df["adx"] = adx_df[adx_col[0]]
        else:
            df["adx"] = float("nan")

        # ── ATR ───────────────────────────────────────────────────
        df["atr"] = ta.atr(
            df["high"], df["low"], df["close"],
            length=cfg.atr_period
        )
        atr_ma_period = int(thr.atr_ma_period)
        df["atr_ma"] = df["atr"].rolling(window=atr_ma_period).mean()

        # ── Volume MA ─────────────────────────────────────────────
        df["volume_ma"] = (
            df["volume"].rolling(window=cfg.volume_ma_period).mean()
        )

        return df

    def _extract_snapshot(
        self, df: pd.DataFrame, row_index: int
    ) -> IndicatorSnapshot:
        """Pull the requested row into an IndicatorSnapshot."""
        cfg = self.cfg
        row = df.iloc[row_index]

        def _val(col: str) -> Optional[float]:
            """Return float or None if NaN / missing."""
            v = row.get(col, float("nan"))
            return None if (v is None or (isinstance(v, float) and math.isnan(v))) else float(v)

        return IndicatorSnapshot(
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
        )
