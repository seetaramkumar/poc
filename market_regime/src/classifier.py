"""
classifier.py — Select the winning regime from scored candidates
================================================================
Given a dict of {regime → score} from the scorer, this module
decides which regime wins and whether its confidence meets the
minimum threshold.

Design notes
------------
• The classifier is intentionally stateless — it receives scores
  and returns a result.  All history and state sits in the engine.
• Tie-breaking priority order mirrors intuitive market logic:
  VOLATILE > BEARISH_TREND > BULLISH_TREND > SIDEWAYS > QUIET
  (extremes take precedence so they aren't masked by trend signals)
"""

from __future__ import annotations

from typing import Dict, Tuple

from .config_loader import EngineConfig
from .models import MarketRegime, RegimeSignals, RegimeResult


# Priority order used when two regimes tie for the top score.
_TIEBREAK_PRIORITY: list[MarketRegime] = [
    MarketRegime.VOLATILE,
    MarketRegime.BEARISH_TREND,
    MarketRegime.BULLISH_TREND,
    MarketRegime.SIDEWAYS,
    MarketRegime.QUIET,
]


class RegimeClassifier:
    """
    Selects the market regime from a scored dict.

    Parameters
    ----------
    config : EngineConfig
    """

    def __init__(self, config: EngineConfig) -> None:
        self.cfg = config

    def classify(
        self,
        scores: Dict[MarketRegime, float],
        signals: RegimeSignals,
        snapshot=None,
    ) -> RegimeResult:
        """
        Pick the best regime, apply tiebreaking, and wrap everything
        into a RegimeResult.

        Parameters
        ----------
        scores   : Output of RegimeScorer.score_all()
        signals  : The boolean signal flags (passed through to output)
        snapshot : IndicatorSnapshot (optional, for transparency)

        Returns
        -------
        RegimeResult
        """
        winning_regime, confidence = self._pick_winner(scores)

        min_conf = self.cfg.scoring.min_confidence
        if confidence < min_conf:
            winning_regime = MarketRegime.UNCERTAIN

        return RegimeResult(
            regime=winning_regime,
            confidence=confidence,
            signals=signals,
            scores={r.value: s for r, s in scores.items()},
            indicator_snapshot=snapshot,
        )

    # ──────────────────────────────────────────────────────────────
    #  Private helpers
    # ──────────────────────────────────────────────────────────────

    @staticmethod
    def _pick_winner(
        scores: Dict[MarketRegime, float],
    ) -> Tuple[MarketRegime, float]:
        """
        Find the regime with the highest score.
        Ties are broken by _TIEBREAK_PRIORITY order.
        """
        if not scores:
            return MarketRegime.UNCERTAIN, 0.0

        max_score = max(scores.values())

        # Collect all regimes that share the maximum score
        candidates = [r for r, s in scores.items() if s == max_score]

        # Apply priority tiebreaking
        for regime in _TIEBREAK_PRIORITY:
            if regime in candidates:
                return regime, max_score

        # Fallback (should never reach here)
        return candidates[0], max_score
