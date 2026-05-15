"""
stock_regime/src/signals.py
============================
Converts StockIndicatorSnapshot → boolean StockSignals.

Changes from previous version
------------------------------
- Added roc_positive, roc_accelerating (momentum separation)
- Added rs_improving, rs_weakening (improved RS logic)
- Added higher_highs, ema_extended (trend quality)
- rs_positive/rs_negative/rs_strong now derive from rs_3m (not legacy rs)
  while falling back to relative_strength if rs_3m unavailable
"""

from __future__ import annotations

import logging

from .config_loader import StockEngineConfig
from .models import StockIndicatorSnapshot, StockSignals

logger = logging.getLogger(__name__)


class StockSignalExtractor:
    """Derives boolean signals from a StockIndicatorSnapshot."""

    def __init__(self, config: StockEngineConfig) -> None:
        self.thr = config.thresholds

    def extract(self, snap: StockIndicatorSnapshot) -> StockSignals:
        sig = StockSignals()
        thr = self.thr

        # ── Price vs EMAs ────────────────────────────────────────────
        if snap.close is not None and snap.ema200 is not None:
            sig.price_above_ema200 = bool(snap.close > snap.ema200)
            sig.price_below_ema200 = bool(snap.close < snap.ema200)

        # ── EMA cross ────────────────────────────────────────────────
        if snap.ema20 is not None and snap.ema50 is not None:
            sig.ema20_above_ema50 = bool(snap.ema20 > snap.ema50)
            sig.ema20_below_ema50 = bool(snap.ema20 < snap.ema50)

        # ── ADX strength ─────────────────────────────────────────────
        if snap.adx is not None:
            sig.adx_strong = bool(snap.adx >= thr.adx_strong_trend)
            sig.adx_weak   = bool(snap.adx <  thr.adx_weak_trend)

        # ── EMA flatness ─────────────────────────────────────────────
        flat_thr = thr.ema_flat_threshold
        if snap.ema20_slope is not None:
            sig.ema20_flat = bool(abs(snap.ema20_slope) < flat_thr)
        if snap.ema50_slope is not None:
            sig.ema50_flat = bool(abs(snap.ema50_slope) < flat_thr)

        # ── ATR regime ───────────────────────────────────────────────
        if snap.atr is not None and snap.atr_ma is not None and snap.atr_ma > 0:
            ratio = snap.atr / snap.atr_ma
            sig.atr_high       = bool(ratio >= thr.atr_volatile_ratio)
            sig.atr_low        = bool(ratio <= thr.atr_quiet_ratio)
            sig.atr_compressed = bool(ratio <= thr.atr_compressed_ratio)
            sig.atr_expanding  = bool(ratio >  thr.atr_expanding_ratio)

        # ── Volume confirmation ──────────────────────────────────────
        if snap.volume is not None and snap.volume_ma is not None and snap.volume_ma > 0:
            sig.volume_confirmed = bool(
                snap.volume >= snap.volume_ma * thr.volume_surge_ratio
            )

        # ── Relative Strength (use rs_3m preferentially) ─────────────
        # rs_3m is the improved multi-period RS; fall back to legacy rs
        rs_val = snap.rs_3m if snap.rs_3m is not None else snap.relative_strength
        if rs_val is not None:
            sig.rs_positive = bool(rs_val >= thr.rs_positive_threshold)
            sig.rs_negative = bool(rs_val <= thr.rs_negative_threshold)
            sig.rs_strong   = bool(rs_val >= thr.rs_strong_threshold)

        # ── NEW: RS trend (improving vs weakening) ────────────────────
        if snap.rs_trend is not None:
            rs_trend_thr = float(getattr(thr, "rs_trend_positive_threshold", 0.0005))
            sig.rs_improving = bool(snap.rs_trend >  rs_trend_thr)
            sig.rs_weakening = bool(snap.rs_trend < -rs_trend_thr)

        # ── Near 52-week high ────────────────────────────────────────
        if snap.close is not None and snap.high_52w is not None and snap.high_52w > 0:
            pct_from_high = (snap.high_52w - snap.close) / snap.high_52w
            sig.price_near_52w_high = bool(pct_from_high <= thr.near_high_pct)

        # ── NEW: Rate of change (momentum) ────────────────────────────
        if snap.roc_10 is not None:
            sig.roc_positive = bool(snap.roc_10 > 0)

        if snap.acceleration is not None:
            accel_thr = float(getattr(thr, "acceleration_threshold", 0.5))
            sig.roc_accelerating = bool(snap.acceleration > accel_thr)

        # ── NEW: Trend quality — higher highs ────────────────────────
        if snap.higher_highs_count is not None:
            hh_thr = int(getattr(thr, "higher_highs_min_count", 5))
            sig.higher_highs = bool(snap.higher_highs_count >= hh_thr)

        # ── NEW: EMA extended (price far above trend) ─────────────────
        if snap.ema_distance_pct is not None:
            ext_thr = float(getattr(thr, "ema_extended_pct", 0.10))
            sig.ema_extended = bool(snap.ema_distance_pct > ext_thr)

        return sig