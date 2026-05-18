"""
stock_regime/src/signals.py
============================
Phase 1: Added volatile_instability composite, candle_erratic, high_reversal_freq.
Phase 2: Added range_bound, bb_compressed, ema_compressed.

Key fix: VOLATILE now requires erratic/unstable behavior signals, NOT just ATR expansion.
A strong trending stock with expanding ATR is NOT volatile — it's trending.
"""

from __future__ import annotations
import logging
from .config_loader import StockEngineConfig
from .models import StockIndicatorSnapshot, StockSignals

logger = logging.getLogger(__name__)


class StockSignalExtractor:
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

        # ── ADX ──────────────────────────────────────────────────────
        if snap.adx is not None:
            sig.adx_strong = bool(snap.adx >= thr.adx_strong_trend)
            sig.adx_weak   = bool(snap.adx <  thr.adx_weak_trend)

        # ── EMA flatness ─────────────────────────────────────────────
        if snap.ema20_slope is not None:
            sig.ema20_flat = bool(abs(snap.ema20_slope) < thr.ema_flat_threshold)
        if snap.ema50_slope is not None:
            sig.ema50_flat = bool(abs(snap.ema50_slope) < thr.ema_flat_threshold)

        # ── ATR ──────────────────────────────────────────────────────
        if snap.atr is not None and snap.atr_ma is not None and snap.atr_ma > 0:
            ratio = snap.atr / snap.atr_ma
            sig.atr_high       = bool(ratio >= thr.atr_volatile_ratio)
            sig.atr_low        = bool(ratio <= thr.atr_quiet_ratio)
            sig.atr_compressed = bool(ratio <= thr.atr_compressed_ratio)
            sig.atr_expanding  = bool(ratio >  thr.atr_expanding_ratio)

        # ── Volume ───────────────────────────────────────────────────
        if snap.volume is not None and snap.volume_ma is not None and snap.volume_ma > 0:
            sig.volume_confirmed = bool(snap.volume >= snap.volume_ma * thr.volume_surge_ratio)

        # ── RS ───────────────────────────────────────────────────────
        rs_val = snap.rs_3m if snap.rs_3m is not None else snap.relative_strength
        if rs_val is not None:
            sig.rs_positive = bool(rs_val >= thr.rs_positive_threshold)
            sig.rs_negative = bool(rs_val <= thr.rs_negative_threshold)
            sig.rs_strong   = bool(rs_val >= thr.rs_strong_threshold)

        if snap.rs_trend is not None:
            rs_thr = float(getattr(thr, "rs_trend_positive_threshold", 0.0005))
            sig.rs_improving = bool(snap.rs_trend >  rs_thr)
            sig.rs_weakening = bool(snap.rs_trend < -rs_thr)

        # ── Breakout ─────────────────────────────────────────────────
        if snap.close is not None and snap.high_52w is not None and snap.high_52w > 0:
            sig.price_near_52w_high = bool(
                (snap.high_52w - snap.close) / snap.high_52w <= thr.near_high_pct
            )

        # ── ROC / momentum ───────────────────────────────────────────
        if snap.roc_10 is not None:
            sig.roc_positive = bool(snap.roc_10 > 0)
        if snap.acceleration is not None:
            sig.roc_accelerating = bool(
                snap.acceleration > float(getattr(thr, "acceleration_threshold", 0.5))
            )

        # ── Trend quality ────────────────────────────────────────────
        if snap.higher_highs_count is not None:
            sig.higher_highs = bool(
                snap.higher_highs_count >= int(getattr(thr, "higher_highs_min_count", 5))
            )
        if snap.ema_distance_pct is not None:
            sig.ema_extended = bool(
                snap.ema_distance_pct > float(getattr(thr, "ema_extended_pct", 0.10))
            )

        # ────────────────────────────────────────────────────────────
        # PHASE 1: Volatility instability signals
        #
        # IMPORTANT: These measure ERRATIC / UNSTABLE behavior.
        # A strong trending stock with high ATR is NOT volatile here.
        # These signals require incoherent price action.
        # ────────────────────────────────────────────────────────────
        ci_thr  = float(getattr(thr, "candle_instability_threshold", 1.20))
        rf_thr  = float(getattr(thr, "reversal_frequency_threshold", 0.55))
        gf_thr  = float(getattr(thr, "gap_frequency_threshold",      0.15))
        wk_thr  = float(getattr(thr, "wickiness_threshold",          0.55))

        if snap.candle_instability is not None:
            sig.candle_erratic = bool(snap.candle_instability > ci_thr)

        if snap.reversal_frequency is not None:
            sig.high_reversal_freq = bool(snap.reversal_frequency > rf_thr)

        # Composite volatile_instability: needs at least 2 of 4 instability signals
        instability_count = sum([
            bool(snap.candle_instability is not None and snap.candle_instability > ci_thr),
            bool(snap.reversal_frequency is not None and snap.reversal_frequency > rf_thr),
            bool(snap.gap_frequency      is not None and snap.gap_frequency      > gf_thr),
            bool(snap.wickiness_score    is not None and snap.wickiness_score    > wk_thr),
        ])
        sig.volatile_instability = bool(instability_count >= 2)

        # ────────────────────────────────────────────────────────────
        # PHASE 2: Range / sideways detection signals
        # ────────────────────────────────────────────────────────────
        der_thr  = float(getattr(thr, "der_range_threshold",      0.35))
        bb_n_thr = float(getattr(thr, "bb_narrow_threshold",      0.04))
        es_thr   = float(getattr(thr, "ema_compressed_threshold", 0.005))

        if snap.directional_efficiency is not None:
            sig.range_bound = bool(snap.directional_efficiency < der_thr)

        if snap.bb_width is not None:
            sig.bb_compressed = bool(snap.bb_width < bb_n_thr)

        if snap.ema_spread is not None:
            sig.ema_compressed = bool(abs(snap.ema_spread) < es_thr)

        return sig