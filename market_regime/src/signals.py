"""
signals.py — Convert IndicatorSnapshot → RegimeSignals
=======================================================
This module contains pure boolean logic.
It takes numeric indicator values and maps them to named signals
that the scorer can then weight and combine.

Keep this layer thin and side-effect-free.
"""

from __future__ import annotations

from .config_loader import EngineConfig
from .models import IndicatorSnapshot, RegimeSignals


class SignalExtractor:
    """
    Derives boolean signals from a computed IndicatorSnapshot.

    All threshold comparisons live here — not in the classifier
    and not in the scorer.

    Parameters
    ----------
    config : EngineConfig
        Provides all numeric thresholds.
    """

    def __init__(self, config: EngineConfig) -> None:
        self.thr = config.thresholds

    def extract(self, snap: IndicatorSnapshot) -> RegimeSignals:
        """
        Convert numeric indicator values into boolean regime signals.

        Parameters
        ----------
        snap : IndicatorSnapshot

        Returns
        -------
        RegimeSignals
        """
        sig = RegimeSignals()

        # ── Guard: some indicators may not have computed yet ────────
        #    (e.g. first 200 bars lack EMA-200).  Default to False.

        # ── Price vs EMAs ──────────────────────────────────────────
        if snap.close is not None and snap.ema200 is not None:
            sig.price_above_ema200 = snap.close > snap.ema200
            sig.price_below_ema200 = snap.close < snap.ema200

        # ── EMA cross ─────────────────────────────────────────────
        if snap.ema20 is not None and snap.ema50 is not None:
            sig.ema20_above_ema50 = snap.ema20 > snap.ema50
            sig.ema20_below_ema50 = snap.ema20 < snap.ema50

        # ── ADX strength ──────────────────────────────────────────
        if snap.adx is not None:
            sig.adx_strong = snap.adx >= self.thr.adx_strong_trend
            sig.adx_weak   = snap.adx <  self.thr.adx_weak_trend

        # ── EMA flatness (used for SIDEWAYS detection) ────────────
        flat_thr = self.thr.ema_flat_threshold
        if snap.ema20_slope is not None:
            sig.ema20_flat = abs(snap.ema20_slope) < flat_thr
        if snap.ema50_slope is not None:
            sig.ema50_flat = abs(snap.ema50_slope) < flat_thr

        # ── ATR regime ────────────────────────────────────────────
        if snap.atr is not None and snap.atr_ma is not None and snap.atr_ma > 0:
            ratio = snap.atr / snap.atr_ma
            sig.atr_high = ratio >= self.thr.atr_volatile_ratio
            sig.atr_low  = ratio <= self.thr.atr_quiet_ratio

        # ── Volume confirmation ───────────────────────────────────
        if snap.volume is not None and snap.volume_ma is not None and snap.volume_ma > 0:
            sig.volume_confirms = (
                snap.volume >= snap.volume_ma * self.thr.volume_surge_ratio
            )

        return sig
