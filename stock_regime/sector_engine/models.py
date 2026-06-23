"""
stock_regime/sector_engine/models.py
======================================
Typed data contracts for the Sector Intelligence Engine.

FIX 1 — Minimum Sector Sample Size Validation
-----------------------------------------------
Added SectorState.INSUFFICIENT_DATA, distinct from UNKNOWN.

  UNKNOWN            : reserved for sectors that exist in the symbol map
                        but have not yet been classified (transient state,
                        rarely seen in persisted output).
  INSUFFICIENT_DATA   : sector has fewer than `min_sector_stocks` classified
                        members. A single strong/weak stock cannot make an
                        entire sector LEADING or WEAK — this state makes
                        that explicit and machine-checkable.

SectorSnapshot gained `excluded_from_ranking: bool`. When True:
  - rank is forced to 0 (no ordinal position — it never appears in a
    "top N" or "bottom N" listing)
  - the sector is omitted from sector_rankings.parquet (the leaderboard file)
  - the sector is omitted from the router/ranking context map via
    is_actionable() returning False

Design notes
------------
- SectorMetrics holds raw computed values (no classification logic here)
- SectorSnapshot is the final classified output (metrics + state)
- SectorState has 6 levels: 5 substantive + INSUFFICIENT_DATA (+ legacy UNKNOWN)
- All fields are explicit — no loose dicts in the public interface
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from enum import Enum
from typing import Optional


class SectorState(str, Enum):
    """
    Sector classification states.

    LEADING            : outperforming, strong breadth, high RS
    STRONG             : healthy but not at the top
    NEUTRAL            : balanced — no clear directional conviction
    WEAKENING          : breadth deteriorating, RS rolling over
    WEAK               : majority bearish, underperforming
    INSUFFICIENT_DATA  : fewer than min_sector_stocks classified members —
                         too small a sample for a reliable signal
    UNKNOWN            : reserved / transient — not normally persisted
    """
    LEADING           = "LEADING"
    STRONG            = "STRONG"
    NEUTRAL           = "NEUTRAL"
    WEAKENING         = "WEAKENING"
    WEAK              = "WEAK"
    INSUFFICIENT_DATA = "INSUFFICIENT_DATA"
    UNKNOWN           = "UNKNOWN"


# States that represent a real, sample-size-validated classification.
# Used by is_actionable() and anywhere we need to distinguish "this sector
# has a trustworthy signal" from "ignore this sector".
_SUBSTANTIVE_STATES = {
    SectorState.LEADING,
    SectorState.STRONG,
    SectorState.NEUTRAL,
    SectorState.WEAKENING,
    SectorState.WEAK,
}


@dataclass
class SectorMetrics:
    """
    Raw computed metrics for one sector on one run date.

    All scores are in [0, 1]. Percentages are in [0, 100].
    Fields are None when insufficient stocks exist for reliable computation.

    Note: metrics are still computed for small sectors (so the diagnostics
    layer can show "TEXTILES stocks=1 ..." with real numbers) — it is the
    CLASSIFIER that decides whether those metrics are trustworthy enough
    to drive a state and a rank.
    """
    sector:      str
    universe:    str
    run_date:    date
    stock_count: int

    # ── Participation ────────────────────────────────────────────────
    pct_bullish:           float   # % stocks in TREND_UP or MOMENTUM
    pct_bearish:           float   # % stocks in TREND_DOWN
    pct_above_ema200:      float   # % stocks with price > EMA200
    momentum_participation: float  # % stocks in MOMENTUM regime
    breakout_participation: float  # % stocks in BREAKOUT_SETUP regime
    advance_decline_ratio: float   # bullish_count / max(bearish_count, 1)

    # ── Composite scores (mean of individual stock scores) ───────────
    avg_trend_score:     float   # mean dimensional_scores.trend
    avg_momentum_score:  float   # mean dimensional_scores.momentum
    avg_quality_score:   float   # mean quality_engine output score

    # ── Relative strength ─────────────────────────────────────────────
    avg_rs_3m:           Optional[float] = None
    avg_rs_trend:        Optional[float] = None   # mean rs_trend (slope)

    # ── Stability ─────────────────────────────────────────────────────
    avg_confidence:      float = 0.0   # mean smoothed_confidence
    pct_oscillating:     float = 0.0   # % stocks flagged as oscillation_detected

    def to_dict(self) -> dict:
        return {
            "sector":                self.sector,
            "universe":              self.universe,
            "run_date":              str(self.run_date),
            "stock_count":           self.stock_count,
            "pct_bullish":           round(self.pct_bullish,            2),
            "pct_bearish":           round(self.pct_bearish,            2),
            "pct_above_ema200":      round(self.pct_above_ema200,       2),
            "momentum_participation":round(self.momentum_participation,  2),
            "breakout_participation":round(self.breakout_participation,  2),
            "advance_decline_ratio": round(self.advance_decline_ratio,   3),
            "avg_trend_score":       round(self.avg_trend_score,         4),
            "avg_momentum_score":    round(self.avg_momentum_score,      4),
            "avg_quality_score":     round(self.avg_quality_score,       4),
            "avg_rs_3m":             round(self.avg_rs_3m, 4) if self.avg_rs_3m is not None else None,
            "avg_rs_trend":          round(self.avg_rs_trend, 6) if self.avg_rs_trend is not None else None,
            "avg_confidence":        round(self.avg_confidence,          4),
            "pct_oscillating":       round(self.pct_oscillating,         2),
        }


@dataclass
class SectorSnapshot:
    """
    Final classified output for one sector — metrics plus state.

    This is what gets persisted and consumed by the strategy router /
    ranking engine.

    excluded_from_ranking : True when stock_count < min_sector_stocks.
    When True, rank is always 0 — the sector occupies no ordinal position
    and is omitted from sector_rankings.parquet and from the router's
    sector context map.
    """
    metrics:               SectorMetrics
    state:                 SectorState
    rank:                  int   = 0          # 1 = strongest; 0 = excluded
    composite_score:       float = 0.0        # weighted score [0, 1]
    excluded_from_ranking: bool  = False

    # Convenience pass-throughs
    @property
    def sector(self) -> str:       return self.metrics.sector
    @property
    def universe(self) -> str:     return self.metrics.universe
    @property
    def run_date(self) -> date:    return self.metrics.run_date
    @property
    def stock_count(self) -> int:  return self.metrics.stock_count

    def to_dict(self) -> dict:
        d = self.metrics.to_dict()
        d["state"]                 = self.state.value
        d["rank"]                  = self.rank
        d["composite_score"]       = round(self.composite_score, 4)
        d["excluded_from_ranking"] = self.excluded_from_ranking
        return d

    def is_actionable(self) -> bool:
        """
        True only when the sector has a substantive, sample-size-validated
        classification. INSUFFICIENT_DATA and UNKNOWN sectors are never
        actionable — they are excluded from the router's sector context map
        and from any ranking/leaderboard output.
        """
        return (
            self.state in _SUBSTANTIVE_STATES and
            not self.excluded_from_ranking
        )