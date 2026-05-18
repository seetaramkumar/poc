"""
stock_regime/src/scorer.py
============================
Phase 1: VOLATILE scoring now requires volatile_instability (erratic behavior),
         not just ATR expansion. atr_high alone no longer drives VOLATILE.
Phase 2: RANGE scoring enhanced with range_bound, bb_compressed, ema_compressed.
Phase 3: Market-context-aware confidence adjustment applied post-classification.
"""

from __future__ import annotations

import logging
from typing import Optional

from .config_loader import StockEngineConfig
from .models import (
    ContinuousScores, DimensionalScores,
    MarketRegimeInput, StockIndicatorSnapshot,
    StockRegime, StockSignals,
)

logger = logging.getLogger(__name__)


class StockRegimeScorer:
    def __init__(self, config: StockEngineConfig) -> None:
        self.cfg = config

    # ── Regime scores (classification) ──────────────────────────────

    def score_regimes(self, signals: StockSignals) -> dict[StockRegime, float]:
        sig = signals
        sc  = self.cfg.scoring
        i   = int

        scores: dict[StockRegime, float] = {

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

            # Phase 2: RANGE now uses 5 signals instead of 4
            StockRegime.RANGE: (
                sc.range.adx_weak       * i(sig.adx_weak)       +
                sc.range.ema20_flat     * i(sig.ema20_flat)     +
                sc.range.ema50_flat     * i(sig.ema50_flat)     +
                sc.range.atr_low        * i(sig.atr_low)        +
                sc.range.range_bound    * i(sig.range_bound)    +
                sc.range.bb_compressed  * i(sig.bb_compressed)  +
                sc.range.ema_compressed * i(sig.ema_compressed)
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

            # Phase 1: VOLATILE now requires volatile_instability.
            # atr_high alone is no longer sufficient — healthy trends
            # have expanding ATR but are NOT volatile.
            StockRegime.VOLATILE: (
                sc.volatile.volatile_instability * i(sig.volatile_instability) +
                sc.volatile.atr_high             * i(sig.atr_high)             +
                sc.volatile.high_reversal_freq   * i(sig.high_reversal_freq)
            ),

            StockRegime.QUIET: (
                sc.quiet.atr_low  * i(sig.atr_low)  +
                sc.quiet.adx_weak * i(sig.adx_weak)
            ),
        }

        return scores

    # ── Dimensional scores (ranking) ────────────────────────────────

    def score_dimensions(
        self,
        signals: StockSignals,
        snap: Optional[StockIndicatorSnapshot] = None,
    ) -> DimensionalScores:
        cs = self._build_continuous_scores(snap) if snap is not None else ContinuousScores()
        dm = self.cfg.dimensional

        # Trend: EMA structure + ADX + RS level
        trend_w = (dm.trend.price_above_ema200 +
                   dm.trend.ema20_above_ema50 * 0.6 +
                   dm.trend.adx_strong +
                   dm.trend.rs_positive)
        trend_score = (
            dm.trend.price_above_ema200 * cs.ema_alignment_score +
            dm.trend.ema20_above_ema50  * cs.ema_alignment_score * 0.6 +
            dm.trend.adx_strong         * cs.adx_score           +
            dm.trend.rs_positive        * cs.rs_score
        ) / max(trend_w, 1e-9)

        # Momentum: ROC + RS direction + volume + ATR expansion
        mom_w = (dm.momentum.rs_strong + dm.momentum.volume_confirmed +
                 dm.momentum.adx_strong + dm.momentum.atr_expanding)
        momentum_score = (
            dm.momentum.rs_strong        * cs.roc_score        +
            dm.momentum.volume_confirmed * cs.volume_score     +
            dm.momentum.adx_strong       * cs.rs_trend_score   +
            dm.momentum.atr_expanding    * cs.atr_expansion_score
        ) / max(mom_w, 1e-9)

        # Volatility: ATR expansion BUT penalised by instability
        # A clean expanding ATR = healthy trend → lower volatility risk score
        vol_raw = cs.atr_expansion_score
        # If instability is high, boost vol score; if trend is clean, reduce it
        instability_boost = cs.instability_score * 0.30
        trend_penalty     = cs.ema_alignment_score * 0.15
        volatility_score  = min(vol_raw + instability_boost - trend_penalty, 1.0)

        def _clip(v): return round(min(max(v, 0.0), 1.0), 4)

        result = DimensionalScores(
            trend      = _clip(trend_score),
            momentum   = _clip(momentum_score),
            volatility = _clip(volatility_score),
            continuous = cs,
        )

        logger.debug(
            "scores trend=%.3f  mom=%.3f  vol=%.3f  "
            "[adx=%.2f align=%.2f rs=%.2f roc=%.2f instab=%.2f ranging=%.2f]",
            result.trend, result.momentum, result.volatility,
            cs.adx_score, cs.ema_alignment_score, cs.rs_score,
            cs.roc_score, cs.instability_score, cs.ranging_score,
        )
        return result

    # ── Phase 3: Market-context-aware confidence adjustment ──────────

    def apply_market_context(
        self,
        regime_scores: dict[StockRegime, float],
        market_regime: MarketRegimeInput,
    ) -> dict[StockRegime, float]:
        """
        Soft probabilistic weighting of regime scores based on market regime.

        Rules (all soft — no hard blocks):
        - BEARISH market → reduce bullish scores slightly, boost bearish
        - SIDEWAYS market → favour RANGE/QUIET, penalise trending
        - BULLISH market → favour trending/momentum, mild range penalty
        - VOLATILE market → boost VOLATILE score slightly

        Adjustments are bounded ±15% to avoid overwhelming the raw signal.
        """
        mr   = market_regime.regime
        conf = market_regime.confidence  # scale adjustment by market confidence
        adjusted = dict(regime_scores)

        def _adj(regime: StockRegime, factor: float) -> None:
            if regime in adjusted:
                delta = (factor - 1.0) * conf * 0.15
                adjusted[regime] = min(max(adjusted[regime] + delta, 0.0), 1.0)

        if mr == "BEARISH_TREND":
            _adj(StockRegime.TREND_UP,   0.85)  # slight headwind for bulls
            _adj(StockRegime.MOMENTUM,   0.85)
            _adj(StockRegime.TREND_DOWN, 1.15)  # tailwind for bears
            _adj(StockRegime.VOLATILE,   1.10)

        elif mr == "BULLISH_TREND":
            _adj(StockRegime.TREND_UP,   1.10)
            _adj(StockRegime.MOMENTUM,   1.10)
            _adj(StockRegime.TREND_DOWN, 0.90)
            _adj(StockRegime.RANGE,      0.90)

        elif mr == "SIDEWAYS":
            _adj(StockRegime.RANGE,      1.15)
            _adj(StockRegime.QUIET,      1.10)
            _adj(StockRegime.TREND_UP,   0.90)
            _adj(StockRegime.MOMENTUM,   0.90)
            _adj(StockRegime.BREAKOUT_SETUP, 0.85)

        elif mr == "VOLATILE":
            _adj(StockRegime.VOLATILE,   1.10)
            _adj(StockRegime.TREND_UP,   0.92)
            _adj(StockRegime.MOMENTUM,   0.92)

        return adjusted

    # ── Continuous score normalisation ──────────────────────────────

    def _build_continuous_scores(self, snap: StockIndicatorSnapshot) -> ContinuousScores:
        def _c(v): return min(max(v, 0.0), 1.0)

        # ADX: 0→0, 25→0.5, 50→1.0
        adx_score = _c(snap.adx / 50.0) if snap.adx is not None else 0.0

        # EMA alignment
        ema_align = 0.0
        if snap.close  is not None and snap.ema200 is not None:
            ema_align += 0.40 if snap.close > snap.ema200 else 0.0
        if snap.ema20  is not None and snap.ema50  is not None:
            ema_align += 0.40 if snap.ema20  > snap.ema50  else 0.0
        if snap.ema50  is not None and snap.ema200 is not None:
            ema_align += 0.20 if snap.ema50  > snap.ema200 else 0.0

        # EMA distance: -20%→0, 0%→0.5, +20%→1.0
        ema_dist = _c((snap.ema_distance_pct + 0.20) / 0.40) if snap.ema_distance_pct is not None else 0.5

        # ATR expansion: 0.5→0, 1.0→0.5, 2.0→1.0 (but NOT = instability)
        atr_score = 0.5
        if snap.atr is not None and snap.atr_ma is not None and snap.atr_ma > 0:
            atr_score = _c((snap.atr / snap.atr_ma - 0.5) / 1.5)

        # RS level: 0.90→0, 1.00→0.5, 1.10→1.0
        rs_val    = snap.rs_3m if snap.rs_3m is not None else snap.relative_strength
        rs_score  = _c((rs_val - 0.90) / 0.20) if rs_val is not None else 0.5

        # RS trend slope: -0.005→0, 0→0.5, +0.005→1.0
        rs_trend  = _c((snap.rs_trend + 0.005) / 0.010) if snap.rs_trend is not None else 0.5

        # ROC: -5%→0, 0%→0.5, +5%→1.0
        roc_score = _c((snap.roc_10 + 5.0) / 10.0) if snap.roc_10 is not None else 0.5

        # Volume ratio: 0.5→0, 1.0→0.5, 2.0→1.0
        vol_score = 0.0
        if snap.volume is not None and snap.volume_ma is not None and snap.volume_ma > 0:
            vol_score = _c((snap.volume / snap.volume_ma - 0.5) / 1.5)

        # ── Phase 1: Instability score ───────────────────────────────
        # Composite of 4 instability metrics → [0, 1]
        instab_components = []
        if snap.candle_instability is not None:
            # candle_instability: 1.0=normal, 1.5=moderate, 2.0=high
            instab_components.append(_c((snap.candle_instability - 1.0) / 1.0))
        if snap.reversal_frequency is not None:
            # 0.4=normal random, 0.6=erratic, 0.8=chaotic
            instab_components.append(_c((snap.reversal_frequency - 0.40) / 0.40))
        if snap.gap_frequency is not None:
            # 0.05=normal, 0.15=gappy, 0.30=very gappy
            instab_components.append(_c(snap.gap_frequency / 0.30))
        if snap.wickiness_score is not None:
            # 0.30=trending candles, 0.55=mixed, 0.80=doji/reversal
            instab_components.append(_c((snap.wickiness_score - 0.30) / 0.50))
        instability_score = (
            sum(instab_components) / len(instab_components)
            if instab_components else 0.0
        )

        # ── Phase 2: Ranging score ───────────────────────────────────
        ranging_components = []
        if snap.directional_efficiency is not None:
            # DER: 0=ranging, 1=trending → invert for ranging score
            ranging_components.append(1.0 - snap.directional_efficiency)
        if snap.bb_width is not None:
            # bb_width: 0.02=very tight, 0.05=normal, 0.10=wide
            ranging_components.append(_c(1.0 - snap.bb_width / 0.10))
        if snap.ema_spread is not None:
            # |ema_spread|: 0=compressed, 0.01=normal, 0.03=diverged
            ranging_components.append(_c(1.0 - abs(snap.ema_spread) / 0.03))
        ranging_score = (
            sum(ranging_components) / len(ranging_components)
            if ranging_components else 0.0
        )

        return ContinuousScores(
            adx_score           = round(adx_score,           4),
            ema_alignment_score = round(ema_align,            4),
            ema_distance_score  = round(ema_dist,             4),
            atr_expansion_score = round(atr_score,            4),
            rs_score            = round(rs_score,             4),
            rs_trend_score      = round(rs_trend,             4),
            roc_score           = round(roc_score,            4),
            volume_score        = round(vol_score,            4),
            instability_score   = round(instability_score,    4),
            ranging_score       = round(ranging_score,        4),
        )