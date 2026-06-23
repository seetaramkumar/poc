"""
stock_regime/ranking/__init__.py
==================================
Public API for the composite opportunity ranking layer.

FIX 2/3: also exports the default market alignment matrix and the matrix
loader so callers (and tests) can inspect or override it without reaching
into composite_scorer.py internals.
"""

from .ranking_engine    import RankingEngine
from .composite_scorer  import (
    DEFAULT_MARKET_ALIGNMENT_MATRIX,
    load_market_alignment_matrix,
    compute_market_alignment,
    compute_rs_scores_batch,
)
from .models import (
    RankedOpportunity,
    CompositeOpportunityScore,
    ScoreWeights,
    ComponentScores,
)

__all__ = [
    "RankingEngine",
    "RankedOpportunity",
    "CompositeOpportunityScore",
    "ScoreWeights",
    "ComponentScores",
    "DEFAULT_MARKET_ALIGNMENT_MATRIX",
    "load_market_alignment_matrix",
    "compute_market_alignment",
    "compute_rs_scores_batch",
]