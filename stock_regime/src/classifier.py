"""
stock_regime/src/classifier.py
================================
Phase 3 change: market-context score adjustment (apply_market_context from scorer)
is now called BEFORE picking the winner so scores reflect market conditions
before classification, not just after. This reduces market/stock inconsistency.

Tiebreaking priority adjusted: VOLATILE moved down since it now requires
genuine instability signals (less likely to tie with trending regimes).
"""

from __future__ import annotations
import logging

from .config_loader import StockEngineConfig
from .models import (
    DimensionalScores, MarketRegimeInput,
    StockIndicatorSnapshot, StockRegime,
    StockRegimeResult, StockSignals,
)

logger = logging.getLogger(__name__)

# VOLATILE moved down — genuine instability means it won't falsely tie
_TIEBREAK_PRIORITY: list[StockRegime] = [
    StockRegime.TREND_DOWN,
    StockRegime.VOLATILE,
    StockRegime.MOMENTUM,
    StockRegime.TREND_UP,
    StockRegime.BREAKOUT_SETUP,
    StockRegime.RANGE,
    StockRegime.QUIET,
]


class StockRegimeClassifier:
    def __init__(self, config: StockEngineConfig) -> None:
        self.cfg = config

    def classify(
        self,
        symbol:             str,
        market:             str,
        regime_scores:      dict[StockRegime, float],
        signals:            StockSignals,
        dimensional_scores: DimensionalScores,
        market_regime:      MarketRegimeInput,
        snap:               StockIndicatorSnapshot,
        scorer=None,        # StockRegimeScorer, optional — for context adjustment
    ) -> StockRegimeResult:
        """
        Phase 3: apply market-context score adjustment before picking winner.
        This makes stock classifications more consistent with market regime.
        """
        # Apply market-context soft weighting to raw scores first
        if scorer is not None:
            adjusted_scores = scorer.apply_market_context(regime_scores, market_regime)
        else:
            adjusted_scores = regime_scores

        winning_regime, raw_confidence = self._pick_winner(adjusted_scores)

        if raw_confidence < self.cfg.min_confidence:
            winning_regime = StockRegime.UNCERTAIN

        # Post-classification confidence boost/penalty (existing logic)
        confidence = self._apply_context(winning_regime, raw_confidence, market_regime)

        scores_dict = {r.value: round(s, 4) for r, s in adjusted_scores.items()}

        logger.debug(
            "%s → %s (conf=%.2f  market=%s)",
            symbol, winning_regime.value, confidence, market_regime.regime,
        )

        return StockRegimeResult(
            symbol             = symbol,
            market             = market,
            stock_regime       = winning_regime,
            confidence         = round(confidence, 4),
            dimensional_scores = dimensional_scores,
            regime_scores      = scores_dict,
            signals            = signals,
            indicators         = snap,
        )

    @staticmethod
    def _pick_winner(scores: dict[StockRegime, float]) -> tuple[StockRegime, float]:
        if not scores:
            return StockRegime.UNCERTAIN, 0.0
        max_score  = max(scores.values())
        candidates = [r for r, s in scores.items() if s == max_score]
        for regime in _TIEBREAK_PRIORITY:
            if regime in candidates:
                return regime, max_score
        return candidates[0], max_score

    def _apply_context(
        self,
        stock_regime:  StockRegime,
        confidence:    float,
        market_regime: MarketRegimeInput,
    ) -> float:
        if stock_regime == StockRegime.UNCERTAIN:
            return confidence
        ctx         = self.cfg.market_ctx
        market_conf = market_regime.confidence
        is_aligned  = ctx.is_aligned(market_regime.regime, stock_regime.value)
        if is_aligned:
            adjusted = confidence * (1.0 + (ctx.alignment_boost - 1.0) * market_conf)
        else:
            adjusted = confidence * (1.0 - (1.0 - ctx.misalignment_penalty) * market_conf)
        return round(min(max(adjusted, 0.0), 1.0), 4)