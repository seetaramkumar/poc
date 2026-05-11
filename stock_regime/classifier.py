"""
stock_regime/src/classifier.py
================================
Selects the winning stock regime from the scored candidates, applies
tiebreaking, enforces minimum confidence, and adjusts confidence using
the market regime context.

Design notes
------------
• Stateless — receives scores and returns a result.
• Tiebreaking priority mirrors intuitive market logic:
    VOLATILE > TREND_DOWN > MOMENTUM > TREND_UP > BREAKOUT_SETUP > RANGE > QUIET
  Extreme/dangerous regimes take precedence so they are not masked.
• Market context adjustment is the only place where the market regime
  input touches the classification logic.  All other layers are ignorant
  of the market regime.
• Confidence is clamped to [0.0, 1.0] after adjustment to prevent
  values outside the expected range.
"""

from __future__ import annotations

from .config_loader import StockEngineConfig
from .models import (
    DimensionalScores,
    MarketRegimeInput,
    StockIndicatorSnapshot,
    StockRegime,
    StockRegimeResult,
    StockSignals,
)


# Fixed tiebreaking order when two regimes share the maximum score.
# Extreme regimes are prioritised so they are never silenced by trend signals.
_TIEBREAK_PRIORITY: list[StockRegime] = [
    StockRegime.VOLATILE,
    StockRegime.TREND_DOWN,
    StockRegime.MOMENTUM,
    StockRegime.TREND_UP,
    StockRegime.BREAKOUT_SETUP,
    StockRegime.RANGE,
    StockRegime.QUIET,
]


class StockRegimeClassifier:
    """
    Picks the winning stock regime and assembles the final result.

    Parameters
    ----------
    config : StockEngineConfig
    """

    def __init__(self, config: StockEngineConfig) -> None:
        self.cfg = config

    def classify(
        self,
        symbol: str,
        market: str,
        regime_scores: dict[StockRegime, float],
        signals: StockSignals,
        dimensional_scores: DimensionalScores,
        market_regime: MarketRegimeInput,
        snap: StockIndicatorSnapshot,
    ) -> StockRegimeResult:
        """
        Assemble the final StockRegimeResult.

        Steps
        -----
        1. Pick the highest-scoring regime (tiebreak if needed).
        2. Reject as UNCERTAIN if score is below min_confidence.
        3. Apply market context alignment boost / penalty.
        4. Clamp confidence to [0, 1].

        Parameters
        ----------
        symbol :
            Stock ticker.
        market :
            Universe label (e.g. ``"NIFTY500"``).
        regime_scores :
            Output of StockRegimeScorer.score_regimes().
        signals :
            Boolean signals (passed through to result for transparency).
        dimensional_scores :
            Trend / momentum / volatility scores.
        market_regime :
            The current market-level regime context.
        snap :
            Indicator snapshot (passed through for persistence).

        Returns
        -------
        StockRegimeResult
        """
        winning_regime, raw_confidence = self._pick_winner(regime_scores)

        # Reject low-confidence classifications before context adjustment.
        if raw_confidence < self.cfg.min_confidence:
            winning_regime = StockRegime.UNCERTAIN

        # Apply market context alignment only on valid (non-UNCERTAIN) regimes.
        confidence = self._apply_context(
            winning_regime, raw_confidence, market_regime
        )

        # Convert regime scores to string-keyed dict for JSON compatibility.
        scores_dict = {r.value: round(s, 4) for r, s in regime_scores.items()}

        return StockRegimeResult(
            symbol=symbol,
            market=market,
            stock_regime=winning_regime,
            confidence=round(confidence, 4),
            dimensional_scores=dimensional_scores,
            regime_scores=scores_dict,
            signals=signals,
            indicators=snap,
        )

    # ──────────────────────────────────────────────────────────────
    #  Private helpers
    # ──────────────────────────────────────────────────────────────

    @staticmethod
    def _pick_winner(
        scores: dict[StockRegime, float],
    ) -> tuple[StockRegime, float]:
        """
        Find the regime with the highest score, with tiebreaking.

        Returns
        -------
        tuple[StockRegime, float]
            (winning regime, its score)
        """
        if not scores:
            return StockRegime.UNCERTAIN, 0.0

        max_score = max(scores.values())

        # Gather all regimes that share the maximum score.
        candidates = [r for r, s in scores.items() if s == max_score]

        # Apply tiebreaking priority order.
        for regime in _TIEBREAK_PRIORITY:
            if regime in candidates:
                return regime, max_score

        # Fallback: should never reach here if _TIEBREAK_PRIORITY is exhaustive.
        return candidates[0], max_score

    def _apply_context(
        self,
        stock_regime: StockRegime,
        confidence: float,
        market_regime: MarketRegimeInput,
    ) -> float:
        """
        Adjust confidence based on market-level alignment.

        An aligned regime (e.g. TREND_UP during BULLISH_TREND) gets a
        configurable boost.  A contradictory regime gets a penalty.
        UNCERTAIN regimes are not adjusted — they carry no directional signal.

        The adjustment is proportional to the market regime confidence so
        a low-confidence market reading has less influence.
        """
        if stock_regime == StockRegime.UNCERTAIN:
            return confidence

        ctx = self.cfg.market_ctx
        market_conf = market_regime.confidence  # [0, 1]

        is_aligned = ctx.is_aligned(market_regime.regime, stock_regime.value)

        if is_aligned:
            # Scale boost by market confidence so a weak market signal has
            # proportionally less impact.
            boost = 1.0 + (ctx.alignment_boost - 1.0) * market_conf
            adjusted = confidence * boost
        else:
            # Apply a mild penalty for regimes that contradict the market.
            penalty = 1.0 - (1.0 - ctx.misalignment_penalty) * market_conf
            adjusted = confidence * penalty

        # Clamp to valid range.
        return min(max(adjusted, 0.0), 1.0)
