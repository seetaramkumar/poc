"""
stock_regime/ranking/models.py
================================
Typed contracts for the composite opportunity scoring and ranking layer.

FIX 2 — Strengthened Market Alignment Scoring
-----------------------------------------------
- market_alignment default weight raised from 0.10 to 0.15.
- ScoreWeights.validate() added: warns if any single weight exceeds 0.35
  (prevents one noisy signal from dominating) or if the weight sum is far
  from 1.0.
- CompositeOpportunityScore now stores the NORMALISED weights (summing to
  exactly 1.0), not the raw config weights. This makes every derived
  contribution/penalty calculation mathematically consistent with the
  actual opportunity_score that was computed.

FIX 3 — Relative Strength as a First-Class Component
-------------------------------------------------------
- rs_score default weight raised from 0.00 to 0.10.
- Default weight distribution now matches the recommended profile:
    trend=20%, quality=20%, sector_strength=15%, momentum=15%,
    rs_score=10%, market_alignment=15%, breadth_alignment=3%, stability=2%
  (sums to 100%).
- RankedOpportunity gained `weights_used: ScoreWeights` (the normalised
  weights actually applied) so diagnostics can compute exact per-component
  contributions and the "market alignment penalty" without recomputation.
- Added helper methods on both CompositeOpportunityScore and
  RankedOpportunity: component_contribution(), market_alignment_penalty(),
  rs_contribution() — every number that appears in diagnostics logging is
  produced by one of these, never by ad-hoc arithmetic scattered in
  logging code.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date
from typing import Optional

logger = logging.getLogger(__name__)

# Component names in the order they are evaluated — used by component_contribution()
_COMPONENT_NAMES = [
    "trend", "momentum", "quality", "market_alignment",
    "breadth_alignment", "sector_strength", "stability", "rs_score",
]


# ─────────────────────────────────────────────────────────────────────────────
#  Score weights (fully config-driven)
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ScoreWeights:
    """
    Weights for each component of the composite opportunity score.

    All weights should sum to 1.0. The engine normalises them if they don't,
    but a sum far from 1.0 is logged as a warning via validate().

    Default calibration (FIX 2 + FIX 3):
        trend               20%
        quality              20%
        sector_strength       15%
        momentum             15%
        market_alignment      15%   ← raised from 10% (FIX 2)
        rs_score             10%   ← raised from 0%  (FIX 3)
        breadth_alignment     3%
        stability             2%
                            ────
                            100%

    Tuning guidance:
    - In a range-heavy market, increase stability weight.
    - In a strong trending market, increase trend + momentum weights.
    - Never set any single component above 35% — over-concentration
      creates false precision from a single noisy signal. validate()
      will warn if this happens.
    """
    trend:             float = 0.20
    momentum:          float = 0.15
    quality:           float = 0.20
    sector_strength:   float = 0.15
    breadth_alignment: float = 0.03
    market_alignment:  float = 0.15
    stability:         float = 0.02
    rs_score:          float = 0.10

    def total(self) -> float:
        return (
            self.trend + self.momentum + self.quality +
            self.sector_strength + self.breadth_alignment +
            self.market_alignment + self.stability + self.rs_score
        )

    def normalised(self) -> "ScoreWeights":
        """Return a copy with weights normalised to sum to exactly 1.0."""
        t = self.total()
        if t <= 0:
            raise ValueError("ScoreWeights total is zero — cannot normalise.")
        return ScoreWeights(
            trend             = self.trend             / t,
            momentum          = self.momentum          / t,
            quality           = self.quality           / t,
            sector_strength   = self.sector_strength   / t,
            breadth_alignment = self.breadth_alignment / t,
            market_alignment  = self.market_alignment  / t,
            stability         = self.stability         / t,
            rs_score          = self.rs_score          / t,
        )

    def validate(self, max_single_weight: float = 0.35) -> list[str]:
        """
        Return a list of human-readable warnings about this weight
        configuration. Does not raise — callers decide whether to log,
        abort, or proceed. An empty list means no concerns.
        """
        warnings: list[str] = []
        total = self.total()
        if abs(total - 1.0) > 0.05:
            warnings.append(
                f"ScoreWeights sum to {total:.3f} (expected 1.0) — "
                f"will be normalised automatically."
            )
        for name in _COMPONENT_NAMES:
            w = getattr(self, name)
            if w > max_single_weight:
                warnings.append(
                    f"Component '{name}' has weight {w:.2f}, which exceeds "
                    f"the recommended single-component cap of {max_single_weight:.2f}. "
                    f"A single noisy signal should not be allowed to dominate "
                    f"the composite score."
                )
        return warnings

    @classmethod
    def from_config(cls, config) -> "ScoreWeights":
        """Load weights from the ranking.composite_weights config section."""
        rk = getattr(config, "ranking", None)
        cw = getattr(rk, "composite_weights", None) if rk else None
        if cw is None:
            return cls()   # use defaults
        return cls(
            trend             = float(getattr(cw, "trend",             0.20)),
            momentum          = float(getattr(cw, "momentum",          0.15)),
            quality           = float(getattr(cw, "quality",           0.20)),
            sector_strength   = float(getattr(cw, "sector_strength",   0.15)),
            breadth_alignment = float(getattr(cw, "breadth_alignment", 0.03)),
            market_alignment  = float(getattr(cw, "market_alignment",  0.15)),
            stability         = float(getattr(cw, "stability",         0.02)),
            rs_score          = float(getattr(cw, "rs_score",          0.10)),
        )


# ─────────────────────────────────────────────────────────────────────────────
#  Component scores (all in [0, 1] before final × 100 scaling)
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ComponentScores:
    """
    Individual normalised component scores [0, 1] feeding the composite.

    Every component is explicitly derived and documented so the composite
    score is fully explainable — no black-box aggregation.

    FIX 3: rs_score is now computed from a 70%-level / 30%-trend blend,
    each normalised via percentile rank across the universe (see
    composite_scorer.compute_rs_scores_batch). rs_level_score and
    rs_trend_score are stored separately purely for diagnostics —
    rs_score (the blended value) is what actually feeds the composite.
    """
    trend:     float = 0.0   # from DimensionalScores.trend
    momentum:  float = 0.0   # from DimensionalScores.momentum
    quality:   float = 0.0   # from QualityScore.quality_score

    # 1.0 = regime perfectly aligned with market; 0.0 = fully misaligned
    market_alignment: float = 0.5

    # BreadthSnapshot.regime_breadth_score already [0,1]
    breadth_alignment: float = 0.5

    # Derived from SectorState: LEADING=1.0, STRONG=0.8, ... WEAK=0.0
    sector_strength: float = 0.5

    # Combines smoothed_confidence + regime_age normalisation
    stability: float = 0.0

    # Blended RS score: 0.70 * rs_level_score + 0.30 * rs_trend_score
    rs_score: float = 0.5
    # Diagnostic-only components (not separately weighted — informational)
    rs_level_score: float = 0.5    # percentile rank of rs_3m across universe
    rs_trend_score: float = 0.5    # percentile rank of rs_trend across universe

    def to_dict(self) -> dict:
        return {
            "trend":             round(self.trend,             4),
            "momentum":          round(self.momentum,          4),
            "quality":           round(self.quality,           4),
            "market_alignment":  round(self.market_alignment,  4),
            "breadth_alignment": round(self.breadth_alignment, 4),
            "sector_strength":   round(self.sector_strength,   4),
            "stability":         round(self.stability,         4),
            "rs_score":          round(self.rs_score,          4),
            "rs_level_score":    round(self.rs_level_score,    4),
            "rs_trend_score":    round(self.rs_trend_score,    4),
        }


# ─────────────────────────────────────────────────────────────────────────────
#  Primary scoring output
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class CompositeOpportunityScore:
    """
    Final composite score for one stock on one run date.

    opportunity_score is in [0, 100] — a clean, readable scale.
    component_scores carries the full breakdown for diagnostics.
    weights_used carries the NORMALISED weights actually applied (FIX 2) —
    every contribution/penalty helper below derives directly from these
    two fields, so the numbers always reconcile with opportunity_score.
    """
    symbol:            str
    market:            str
    run_date:          date
    opportunity_score: float          # [0, 100]
    component_scores:  ComponentScores
    weights_used:      ScoreWeights   # normalised — sums to 1.0

    def to_dict(self) -> dict:
        return {
            "symbol":            self.symbol,
            "market":            self.market,
            "run_date":          str(self.run_date),
            "opportunity_score": round(self.opportunity_score, 2),
            **{f"cs_{k}": v for k, v in self.component_scores.to_dict().items()},
        }

    def component_contribution(self, component: str) -> float:
        """
        Points (on the [0, 100] scale) that `component` contributed to the
        final opportunity_score: weight × score × 100.
        """
        weight = getattr(self.weights_used, component, 0.0)
        score  = getattr(self.component_scores, component, 0.0)
        return round(weight * score * 100.0, 3)

    def market_alignment_penalty(self) -> float:
        """
        Points (on the [0, 100] scale) LOST due to imperfect market alignment:
        weight_market_alignment × (1.0 − market_alignment_score) × 100.

        0.0 means the stock's regime is perfectly aligned with the current
        market regime — no penalty. The maximum possible penalty equals
        the full market_alignment weight (e.g. 15 points if weight=0.15).
        """
        weight = self.weights_used.market_alignment
        score  = self.component_scores.market_alignment
        return round(weight * (1.0 - score) * 100.0, 3)

    def rs_contribution(self) -> float:
        """Points contributed by the blended RS score."""
        return self.component_contribution("rs_score")

    def dominant_factor(self) -> str:
        """Return the name of the highest-weighted contributing component."""
        contrib = {name: self.component_contribution(name) for name in _COMPONENT_NAMES}
        return max(contrib, key=contrib.get)


# ─────────────────────────────────────────────────────────────────────────────
#  Ranked output (adds rank, strategy, sector)
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class RankedOpportunity:
    """
    Final ranked output — what gets persisted and consumed downstream.

    This is the record that a signal generator or dashboard reads.

    weights_used (FIX 2/3) is carried through from the CompositeOpportunityScore
    so diagnostics can compute exact contribution/penalty figures per stock
    without needing to recompute or guess at the weighting that produced
    opportunity_score.
    """
    rank:              int
    symbol:            str
    market:            str
    run_date:          date
    opportunity_score: float          # [0, 100]
    strategy:          str            # from StrategyRouter
    sector:            str
    stable_regime:     str
    risk_profile:      str
    psm:               float          # position size multiplier from router
    component_scores:  ComponentScores
    weights_used:      ScoreWeights   # normalised weights actually applied
    dominant_factor:   str

    def to_dict(self) -> dict:
        return {
            "rank":              self.rank,
            "symbol":            self.symbol,
            "market":            self.market,
            "run_date":          str(self.run_date),
            "opportunity_score": round(self.opportunity_score, 2),
            "strategy":          self.strategy,
            "sector":            self.sector,
            "stable_regime":     self.stable_regime,
            "risk_profile":      self.risk_profile,
            "psm":               round(self.psm, 2),
            "dominant_factor":   self.dominant_factor,
            "market_alignment_penalty": self.market_alignment_penalty(),
            "rs_contribution":           self.rs_contribution(),
            **{f"cs_{k}": v for k, v in self.component_scores.to_dict().items()},
        }

    def component_contribution(self, component: str) -> float:
        weight = getattr(self.weights_used, component, 0.0)
        score  = getattr(self.component_scores, component, 0.0)
        return round(weight * score * 100.0, 3)

    def market_alignment_penalty(self) -> float:
        weight = self.weights_used.market_alignment
        score  = self.component_scores.market_alignment
        return round(weight * (1.0 - score) * 100.0, 3)

    def rs_contribution(self) -> float:
        return self.component_contribution("rs_score")