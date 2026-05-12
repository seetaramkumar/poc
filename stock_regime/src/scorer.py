"""
stock_regime/src/scorer.py
============================
Computes two sets of scores from boolean signals:

1. **Regime scores** — one weighted float in [0, 1] per StockRegime.
   Used by the classifier to pick the winning regime.

2. **Dimensional scores** — trend, momentum, and volatility scores.
   Used by the ranker and exposed to downstream consumers.
   These are intentionally independent of which regime wins.

Architecture note
-----------------
The scorer knows about weights (from config) and signal names.
It does NOT know about thresholds (SignalExtractor's job) or
about which regime wins (Classifier's job).
"""

from __future__ import annotations

import math
from typing import Optional

from .config_loader import StockEngineConfig
from .models import DimensionalScores, StockIndicatorSnapshot, StockRegime, StockSignals


class StockRegimeScorer:
    """
    Computes regime confidence scores and dimensional scores from signals.

    Parameters
    ----------
    config : StockEngineConfig
    """

    def __init__(self, config: StockEngineConfig) -> None:
        self.cfg = config

    def score_regimes(self, signals: StockSignals) -> dict[StockRegime, float]:
        """
        Return a dict mapping every StockRegime to its score in [0, 1].

        Each score is the dot-product of configured weights and the
        corresponding boolean signal values.

        Parameters
        ----------
        signals : StockSignals

        Returns
        -------
        dict[StockRegime, float]
        """
        sig = signals   # alias for brevity
        sc  = self.cfg.scoring

        # Integer conversion of booleans keeps arithmetic explicit and readable.
        i = int

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

        return scores

    def score_dimensions(
        self,
        signals: StockSignals,
        snap: Optional[StockIndicatorSnapshot] = None,
    ) -> DimensionalScores:
        """
        Compute trend, momentum, and volatility dimensional scores.

        These scores are independent of regime classification.
        They serve the ranking layer and downstream consumers.

        Parameters
        ----------
        signals : StockSignals
        snap :
            Optional snapshot used to compute the raw ATR ratio for the
            volatility score.  When ``None``, volatility defaults to 0.0.

        Returns
        -------
        DimensionalScores
        """
        sig = signals
        dm  = self.cfg.dimensional
        i   = int

        # Trend score: measures how convincingly a stock is in an uptrend.
        trend_score = (
            dm.trend.price_above_ema200 * i(sig.price_above_ema200) +
            dm.trend.ema20_above_ema50  * i(sig.ema20_above_ema50)  +
            dm.trend.adx_strong         * i(sig.adx_strong)         +
            dm.trend.rs_positive        * i(sig.rs_positive)
        )

        # Momentum score: measures relative outperformance and market
        # participation (volume + ADX + range expansion).
        momentum_score = (
            dm.momentum.rs_strong        * i(sig.rs_strong)        +
            dm.momentum.volume_confirmed * i(sig.volume_confirmed) +
            dm.momentum.adx_strong       * i(sig.adx_strong)       +
            dm.momentum.atr_expanding    * i(sig.atr_expanding)
        )

        # Volatility score: raw ATR ratio normalised to [0, 1].
        # ATR/ATR_MA = 2.0 maps to 1.0; 0.5 maps to 0.25.
        # We cap at 2.0 so extreme outliers don't distort rankings.
        volatility_score = 0.0
        if (
            snap is not None
            and snap.atr is not None
            and snap.atr_ma is not None
            and snap.atr_ma > 0
        ):
            raw_ratio = snap.atr / snap.atr_ma
            volatility_score = min(raw_ratio, 2.0) / 2.0

        return DimensionalScores(
            trend      = round(min(max(trend_score,      0.0), 1.0), 4),
            momentum   = round(min(max(momentum_score,   0.0), 1.0), 4),
            volatility = round(min(max(volatility_score, 0.0), 1.0), 4),
        )
