"""
stock_regime/src/signals.py
============================
Converts a StockIndicatorSnapshot into named boolean StockSignals.

All threshold comparisons live exclusively in this module — not in the
scorer, not in the classifier.  Keeping this layer thin and pure means
the scoring and classification layers remain testable in isolation.

Design note
-----------
Each boolean extraction is guarded against None.  When an indicator has
not yet been computed (insufficient history), the corresponding signal
defaults to False.  The downstream scorer handles this gracefully because
False contributes 0.0 to the weighted sum.
"""

from __future__ import annotations

from .config_loader import StockEngineConfig
from .models import StockIndicatorSnapshot, StockSignals


class StockSignalExtractor:
    """
    Derives boolean signals from a StockIndicatorSnapshot.

    Parameters
    ----------
    config : StockEngineConfig
        Provides all numeric thresholds.
    """

    def __init__(self, config: StockEngineConfig) -> None:
        self.thr = config.thresholds

    def extract(self, snap: StockIndicatorSnapshot) -> StockSignals:
        """
        Convert numeric indicator values into boolean StockSignals.

        Parameters
        ----------
        snap : StockIndicatorSnapshot

        Returns
        -------
        StockSignals
        """
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

        # ── Relative Strength ────────────────────────────────────────
        if snap.relative_strength is not None:
            rs = snap.relative_strength
            sig.rs_positive = bool(rs >= thr.rs_positive_threshold)
            sig.rs_negative = bool(rs <= thr.rs_negative_threshold)
            sig.rs_strong   = bool(rs >= thr.rs_strong_threshold)

        # ── Near 52-week high ────────────────────────────────────────
        if snap.close is not None and snap.high_52w is not None and snap.high_52w > 0:
            pct_from_high = (snap.high_52w - snap.close) / snap.high_52w
            sig.price_near_52w_high = bool(pct_from_high <= thr.near_high_pct)

        return sig
