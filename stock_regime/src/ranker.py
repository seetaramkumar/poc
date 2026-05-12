"""
stock_regime/src/ranker.py
============================
Ranks a batch of StockRegimeResult objects across three dimensions:

  - strongest_trends   : highest trend dimensional score
  - strongest_momentum : highest momentum dimensional score
  - highest_volatility : highest volatility expansion score

The ranker is intentionally separated from the engine so that ranking
logic can be changed, filtered, or extended without touching classification.

Design
------
• Pure functions — no state, no side effects.
• Each ranking list contains only valid (non-UNCERTAIN, no-error) results.
• Rankings are returned as a RankingOutput dataclass for easy persistence.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .config_loader import StockEngineConfig
from .models import StockRegime, StockRegimeResult


# ─────────────────────────────────────────────────────────────
#  Ranked entry (one row in a ranking list)
# ─────────────────────────────────────────────────────────────

@dataclass
class RankedStock:
    """A single entry in a ranking list."""
    rank:         int
    symbol:       str
    market:       str
    stock_regime: str
    confidence:   float
    score:        float   # the dimension score that drove this ranking
    ranking_type: str     # "trend" | "momentum" | "volatility"

    def to_dict(self) -> dict:
        return {
            "rank":         self.rank,
            "symbol":       self.symbol,
            "market":       self.market,
            "stock_regime": self.stock_regime,
            "confidence":   self.confidence,
            "score":        self.score,
            "ranking_type": self.ranking_type,
        }


# ─────────────────────────────────────────────────────────────
#  Ranking output (all three lists bundled)
# ─────────────────────────────────────────────────────────────

@dataclass
class RankingOutput:
    """Container for all three ranking lists produced in one run."""
    strongest_trends:    list[RankedStock] = field(default_factory=list)
    strongest_momentum:  list[RankedStock] = field(default_factory=list)
    highest_volatility:  list[RankedStock] = field(default_factory=list)

    def all_as_flat_list(self) -> list[RankedStock]:
        """Return all rankings as a flat list (useful for persistence)."""
        return (
            self.strongest_trends
            + self.strongest_momentum
            + self.highest_volatility
        )


# ─────────────────────────────────────────────────────────────
#  Ranker
# ─────────────────────────────────────────────────────────────

class StockRanker:
    """
    Ranks a batch of StockRegimeResult objects.

    Parameters
    ----------
    config : StockEngineConfig
    """

    def __init__(self, config: StockEngineConfig) -> None:
        self.cfg = config

    def rank(self, results: list[StockRegimeResult]) -> RankingOutput:
        """
        Produce three ranked lists from a batch of classification results.

        Only valid (non-error, non-UNCERTAIN) results are included.

        Parameters
        ----------
        results :
            All StockRegimeResult objects from a single engine run.

        Returns
        -------
        RankingOutput
        """
        top_n = int(self.cfg.ranking.top_n)

        # Filter out failed and uncertain stocks for ranking purposes.
        # UNCERTAIN stocks have unreliable dimensional scores.
        valid = [
            r for r in results
            if r.is_valid() and r.stock_regime != StockRegime.UNCERTAIN
        ]

        return RankingOutput(
            strongest_trends   = self._rank_by(valid, "trend",      top_n),
            strongest_momentum = self._rank_by(valid, "momentum",   top_n),
            highest_volatility = self._rank_by(valid, "volatility", top_n),
        )

    # ──────────────────────────────────────────────────────────────
    #  Private helpers
    # ──────────────────────────────────────────────────────────────

    @staticmethod
    def _rank_by(
        results: list[StockRegimeResult],
        dimension: str,
        top_n: int,
    ) -> list[RankedStock]:
        """
        Sort results by a single dimensional score and return the top N.

        Parameters
        ----------
        results :
            Valid classification results.
        dimension :
            One of ``"trend"``, ``"momentum"``, ``"volatility"``.
        top_n :
            Maximum number of entries to return.

        Returns
        -------
        list[RankedStock]
            Ordered best-to-worst, 1-indexed.
        """
        def _get_score(r: StockRegimeResult) -> float:
            ds = r.dimensional_scores
            return getattr(ds, dimension, 0.0)

        sorted_results = sorted(results, key=_get_score, reverse=True)

        ranked: list[RankedStock] = []
        for pos, result in enumerate(sorted_results[:top_n], start=1):
            ranked.append(
                RankedStock(
                    rank         = pos,
                    symbol       = result.symbol,
                    market       = result.market,
                    stock_regime = result.stock_regime.value,
                    confidence   = result.confidence,
                    score        = round(_get_score(result), 4),
                    ranking_type = dimension,
                )
            )

        return ranked
