"""
stock_regime/src/scorer.py
============================
Computes regime scores and dimensional scores.

Changes from previous version
------------------------------
- score_regimes(): unchanged (binary signals drive classification — stable)
- score_dimensions(): now uses ContinuousScores computed from raw indicator
  values so trend/momentum have realistic gradient, not binary saturation
- _build_continuous_scores(): new private method normalising raw indicators
- Trend and momentum scores now use distinct indicator sets so rankings diverge:
    Trend    = EMA alignment + ADX strength + EMA distance + RS level
    Momentum = ROC + acceleration + RS trend direction + volume expansion
"""

from __future__ import annotations

import logging
from typing import Optional

from .config_loader import StockEngineConfig
from .models import (
    ContinuousScores,
    DimensionalScores,
    StockIndicatorSnapshot,
    StockRegime,
    StockSignals,
)

logger = logging.getLogger(__name__)


class StockRegimeScorer:
    """Computes regime confidence scores and dimensional scores."""

    def __init__(self, config: StockEngineConfig) -> None:
        self.cfg = config

    # ──────────────────────────────────────────────────────────────
    #  Regime classification scores (boolean — unchanged)
    # ──────────────────────────────────────────────────────────────

    def score_regimes(self, signals: StockSignals) -> dict[StockRegime, float]:
        """
        Boolean dot-product scoring per regime. Unchanged from prior version —
        regime classification stays deterministic and binary.
        """
        sig = signals
        sc  = self.cfg.scoring
        i   = int

        return {
            StockRegime.TREND_UP: (
                sc.trend_up.price_above_ema200 * i(sig.price_above_ema200) +
                sc.trend_up.ema20_above_ema50  * i(sig.ema20_above_ema50)  +
                sc.trend_up.adx_strong         * i(sig.adx_strong)         +
                sc.trend_up.rs_positive        * i(sig.rs_positive)        +
                sc.trend_up.volume_confirmed   * i(sig.volume_confirmed)
            ),
            StockRegime.TREND_DOWN: (
                sc.trend_down.price_below_ema200 * i(sig.price_below_ema200) +
                sc.trend_down.ema20_below_ema50  * i(sig.ema20_below_ema50)  +
                sc.trend_down.adx_strong         * i(sig.adx_strong)         +
                sc.trend_down.rs_negative        * i(sig.rs_negative)        +
                sc.trend_down.volume_confirmed   * i(sig.volume_confirmed)
            ),
            StockRegime.RANGE: (
                sc.range.adx_weak   * i(sig.adx_weak)   +
                sc.range.ema20_flat * i(sig.ema20_flat) +
                sc.range.ema50_flat * i(sig.ema50_flat) +
                sc.range.atr_low    * i(sig.atr_low)
            ),
            StockRegime.MOMENTUM: (
                sc.momentum.rs_strong        * i(sig.rs_strong)        +
                sc.momentum.adx_strong       * i(sig.adx_strong)       +
                sc.momentum.volume_confirmed * i(sig.volume_confirmed) +
                sc.momentum.atr_expanding    * i(sig.atr_expanding)
            ),
            StockRegime.BREAKOUT_SETUP: (
                sc.breakout_setup.atr_compressed      * i(sig.atr_compressed)      +
                sc.breakout_setup.price_near_52w_high * i(sig.price_near_52w_high) +
                sc.breakout_setup.volume_confirmed    * i(sig.volume_confirmed)
            ),
            StockRegime.VOLATILE: (
                sc.volatile.atr_high   * i(sig.atr_high)   +
                sc.volatile.adx_strong * i(sig.adx_strong)
            ),
            StockRegime.QUIET: (
                sc.quiet.atr_low  * i(sig.atr_low)  +
                sc.quiet.adx_weak * i(sig.adx_weak)
            ),
        }

    # ──────────────────────────────────────────────────────────────
    #  Dimensional scores (continuous — replaces binary version)
    # ──────────────────────────────────────────────────────────────

    def score_dimensions(
        self,
        signals: StockSignals,
        snap: Optional[StockIndicatorSnapshot] = None,
    ) -> DimensionalScores:
        """
        Compute trend, momentum, and volatility dimensional scores.

        Now uses ContinuousScores derived from raw indicator values so that
        results are realistically distributed across stocks rather than
        saturating at 1.0 whenever most signals are True.

        Trend and Momentum use distinct indicator sets so their rankings
        naturally diverge:

        TREND    = EMA structure + ADX + distance from macro trend + RS level
        MOMENTUM = ROC/acceleration + RS trend direction + volume expansion + ATR

        Parameters
        ----------
        signals : StockSignals
            Boolean signals (still used for volatility score).
        snap : StockIndicatorSnapshot
            Raw indicator values used for continuous normalisation.
        """
        cs = self._build_continuous_scores(snap) if snap is not None else ContinuousScores()

        # ── TREND score ───────────────────────────────────────────────
        # Measures directional persistence and structural alignment.
        # Components: EMA alignment (structure), ADX (strength),
        #             EMA distance (extension), RS level (leadership)
        dm = self.cfg.dimensional
        trend_score = (
            dm.trend.price_above_ema200 * cs.ema_alignment_score +
            dm.trend.ema20_above_ema50  * cs.ema_alignment_score * 0.6 +
            dm.trend.adx_strong         * cs.adx_score           +
            dm.trend.rs_positive        * cs.rs_score
        )
        # Re-normalise to [0, 1] (weights may sum > 1 after continuous mixing)
        trend_w_sum = (
            dm.trend.price_above_ema200 +
            dm.trend.ema20_above_ema50  * 0.6 +
            dm.trend.adx_strong +
            dm.trend.rs_positive
        )
        if trend_w_sum > 0:
            trend_score = trend_score / trend_w_sum

        # ── MOMENTUM score ────────────────────────────────────────────
        # Measures acceleration and recent outperformance velocity.
        # Components: ROC (price velocity), RS trend direction (improving),
        #             volume expansion, ATR expansion (range breakout)
        momentum_score = (
            dm.momentum.rs_strong        * cs.roc_score        +
            dm.momentum.volume_confirmed * cs.volume_score     +
            dm.momentum.adx_strong       * cs.rs_trend_score   +
            dm.momentum.atr_expanding    * cs.atr_expansion_score
        )
        mom_w_sum = (
            dm.momentum.rs_strong +
            dm.momentum.volume_confirmed +
            dm.momentum.adx_strong +
            dm.momentum.atr_expanding
        )
        if mom_w_sum > 0:
            momentum_score = momentum_score / mom_w_sum

        # ── VOLATILITY score ──────────────────────────────────────────
        # Raw ATR ratio normalised [0, 1]. Unchanged from prior version.
        volatility_score = cs.atr_expansion_score

        # Clip all to [0, 1]
        def _clip(v: float) -> float:
            return round(min(max(v, 0.0), 1.0), 4)

        result = DimensionalScores(
            trend      = _clip(trend_score),
            momentum   = _clip(momentum_score),
            volatility = _clip(volatility_score),
            continuous = cs,
        )

        logger.debug(
            "scores  trend=%.3f  momentum=%.3f  vol=%.3f  "
            "[adx=%.2f  ema_align=%.2f  rs=%.2f  roc=%.2f]",
            result.trend, result.momentum, result.volatility,
            cs.adx_score, cs.ema_alignment_score, cs.rs_score, cs.roc_score,
        )
        return result

    # ──────────────────────────────────────────────────────────────
    #  Continuous score normalisation
    # ──────────────────────────────────────────────────────────────

    def _build_continuous_scores(
        self, snap: StockIndicatorSnapshot
    ) -> ContinuousScores:
        """
        Map raw indicator values to [0, 1] continuous component scores.

        All mappings are monotone functions calibrated to realistic ranges.
        Values outside the calibrated range are clipped — never extrapolated.
        """

        def _clip01(v: float) -> float:
            return min(max(v, 0.0), 1.0)

        # ── ADX score ────────────────────────────────────────────────
        # Linear: 0 at ADX=0, 0.5 at ADX=25, 1.0 at ADX=50
        adx_score = 0.0
        if snap.adx is not None:
            adx_score = _clip01(snap.adx / 50.0)

        # ── EMA alignment score ──────────────────────────────────────
        # 3 binary sub-components each worth 0.333:
        #   close > EMA200, EMA20 > EMA50, EMA50 > EMA200
        ema_align = 0.0
        if snap.close is not None and snap.ema200 is not None:
            ema_align += 0.40 if snap.close > snap.ema200 else 0.0
        if snap.ema20 is not None and snap.ema50 is not None:
            ema_align += 0.40 if snap.ema20 > snap.ema50 else 0.0
        if snap.ema50 is not None and snap.ema200 is not None:
            ema_align += 0.20 if snap.ema50 > snap.ema200 else 0.0

        # ── EMA distance score ───────────────────────────────────────
        # (close - EMA200) / EMA200 → normalised to [-20%, +20%] range
        # Maps: -20% → 0.0, 0% → 0.5, +20% → 1.0
        ema_dist_score = 0.5  # neutral default
        if snap.ema_distance_pct is not None:
            raw = snap.ema_distance_pct
            ema_dist_score = _clip01((raw + 0.20) / 0.40)

        # ── ATR expansion score ──────────────────────────────────────
        # ratio = ATR / ATR_MA; maps: 0.5→0, 1.0→0.5, 2.0→1.0 (linear)
        atr_score = 0.5
        if snap.atr is not None and snap.atr_ma is not None and snap.atr_ma > 0:
            ratio     = snap.atr / snap.atr_ma
            atr_score = _clip01((ratio - 0.5) / 1.5)

        # ── RS score (level) ─────────────────────────────────────────
        # RS range [0.90, 1.10] → [0, 1]
        # Use rs_3m preferentially; fall back to legacy relative_strength
        rs_val = snap.rs_3m if snap.rs_3m is not None else snap.relative_strength
        rs_score = 0.5  # neutral when no benchmark
        if rs_val is not None:
            rs_score = _clip01((rs_val - 0.90) / 0.20)

        # ── RS trend score (direction of RS change) ──────────────────
        # rs_trend is a slope value; typical range [-0.005, +0.005]
        # Maps: -0.005→0, 0→0.5, +0.005→1.0
        rs_trend_score = 0.5  # neutral
        if snap.rs_trend is not None:
            rs_trend_score = _clip01((snap.rs_trend + 0.005) / 0.010)

        # ── ROC score (price velocity / momentum) ────────────────────
        # roc_10 range [-5%, +5%] → [0, 1]
        # Positive ROC = momentum building
        roc_score = 0.5  # neutral
        if snap.roc_10 is not None:
            roc_score = _clip01((snap.roc_10 + 5.0) / 10.0)

        # ── Volume score ─────────────────────────────────────────────
        # vol/vol_ma range [0.5, 2.0] → [0, 1]
        vol_score = 0.0
        if snap.volume is not None and snap.volume_ma is not None and snap.volume_ma > 0:
            ratio     = snap.volume / snap.volume_ma
            vol_score = _clip01((ratio - 0.5) / 1.5)

        return ContinuousScores(
            adx_score           = round(adx_score,       4),
            ema_alignment_score = round(ema_align,        4),
            ema_distance_score  = round(ema_dist_score,   4),
            atr_expansion_score = round(atr_score,        4),
            rs_score            = round(rs_score,         4),
            rs_trend_score      = round(rs_trend_score,   4),
            roc_score           = round(roc_score,        4),
            volume_score        = round(vol_score,        4),
        )