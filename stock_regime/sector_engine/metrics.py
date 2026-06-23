"""
stock_regime/sector_engine/metrics.py
========================================
Pure functions that compute SectorMetrics from a list of classified stocks.

Design notes
------------
- Every function is a pure computation — no logging, no I/O, no side effects.
- This makes the logic trivially testable and backtesting-compatible.
- The orchestrator (sector.py) calls these functions and handles I/O.
- Quality scores are passed in as a pre-built lookup dict so this module
  has no dependency on OpportunityQualityEngine.
"""

from __future__ import annotations

from datetime import date
from typing import Optional

from .models import SectorMetrics

# Regimes that count as "bullish" for breadth purposes
_BULLISH = {"TREND_UP", "MOMENTUM"}
_BEARISH = {"TREND_DOWN"}


def compute_sector_metrics(
    sector:         str,
    universe:       str,
    run_date:       date,
    stable_results: list,                    # list[StableRegimeResult]
    quality_map:    dict[str, float],        # symbol → quality_score
) -> Optional[SectorMetrics]:
    """
    Compute all sector metrics from a batch of classified stocks.

    Parameters
    ----------
    sector :
        Sector name (already normalised by SectorMap).
    universe :
        Universe label (e.g. "NIFTY500").
    run_date :
        Date of this run.
    stable_results :
        All StableRegimeResult objects belonging to this sector.
    quality_map :
        {symbol: quality_score} from OpportunityQualityEngine.
        Symbols missing from this map default to 0.0.

    Returns
    -------
    SectorMetrics or None
        None when stable_results is empty.
    """
    valid = [r for r in stable_results if r.is_valid()]
    n     = len(valid)
    if n == 0:
        return None

    # ── Regime strings ─────────────────────────────────────────────────
    regimes = [_stable_regime(r) for r in valid]

    n_bullish  = sum(1 for rg in regimes if rg in _BULLISH)
    n_bearish  = sum(1 for rg in regimes if rg in _BEARISH)
    n_momentum = sum(1 for rg in regimes if rg == "MOMENTUM")
    n_breakout = sum(1 for rg in regimes if rg == "BREAKOUT_SETUP")

    pct_bullish  = n_bullish  / n * 100
    pct_bearish  = n_bearish  / n * 100
    ad_ratio     = n_bullish  / max(n_bearish, 1)

    # ── EMA200 participation ────────────────────────────────────────────
    n_above_ema200   = sum(1 for r in valid if r.signals.price_above_ema200)
    pct_above_ema200 = n_above_ema200 / n * 100

    # ── Dimensional scores ─────────────────────────────────────────────
    trend_scores = [
        r.dimensional_scores.trend
        for r in valid
        if r.dimensional_scores is not None
    ]
    mom_scores = [
        r.dimensional_scores.momentum
        for r in valid
        if r.dimensional_scores is not None
    ]
    avg_trend    = _safe_mean(trend_scores)
    avg_momentum = _safe_mean(mom_scores)

    # ── Quality scores ─────────────────────────────────────────────────
    quality_vals = [quality_map.get(r.symbol, 0.0) for r in valid]
    avg_quality  = _safe_mean(quality_vals)

    # ── Relative strength ──────────────────────────────────────────────
    rs_vals   = [
        r.indicators.rs_3m
        for r in valid
        if r.indicators.rs_3m is not None
    ]
    rs_trends = [
        r.indicators.rs_trend
        for r in valid
        if r.indicators.rs_trend is not None
    ]
    avg_rs_3m   = _safe_mean(rs_vals)   if rs_vals   else None
    avg_rs_trend= _safe_mean(rs_trends) if rs_trends else None

    # ── Stability ──────────────────────────────────────────────────────
    conf_vals = [
        r.smoothed_confidence if hasattr(r, "smoothed_confidence") else r.confidence
        for r in valid
    ]
    avg_conf = _safe_mean(conf_vals)

    n_oscillating  = sum(
        1 for r in valid
        if hasattr(r, "oscillation_detected") and r.oscillation_detected
    )
    pct_oscillating = n_oscillating / n * 100

    return SectorMetrics(
        sector                  = sector,
        universe                = universe,
        run_date                = run_date,
        stock_count             = n,
        pct_bullish             = round(pct_bullish,             2),
        pct_bearish             = round(pct_bearish,             2),
        pct_above_ema200        = round(pct_above_ema200,        2),
        momentum_participation  = round(n_momentum / n * 100,    2),
        breakout_participation  = round(n_breakout / n * 100,    2),
        advance_decline_ratio   = round(ad_ratio,                3),
        avg_trend_score         = round(avg_trend,               4),
        avg_momentum_score      = round(avg_momentum,            4),
        avg_quality_score       = round(avg_quality,             4),
        avg_rs_3m               = round(avg_rs_3m,   4) if avg_rs_3m   is not None else None,
        avg_rs_trend            = round(avg_rs_trend, 6) if avg_rs_trend is not None else None,
        avg_confidence          = round(avg_conf,                4),
        pct_oscillating         = round(pct_oscillating,         2),
    )


def compute_composite_score(metrics: SectorMetrics) -> float:
    """
    Single composite score [0, 1] that drives sector ranking.

    Weights reflect that breadth (pct_bullish) and EMA200 participation
    are the most reliable forward-looking signals, while RS adds the
    relative outperformance dimension.

    Component weights
    -----------------
    pct_bullish       0.25   — regime breadth (normalised 0–100 → 0–1)
    avg_trend_score   0.20   — dimensional trend score (already [0,1])
    avg_momentum_score 0.15  — dimensional momentum score
    pct_above_ema200  0.20   — structural health
    avg_rs_3m         0.10   — benchmark-relative strength (normalised)
    avg_quality_score 0.10   — opportunity quality
    """
    # Normalise percentage fields to [0, 1]
    bull_norm  = metrics.pct_bullish    / 100.0
    ema_norm   = metrics.pct_above_ema200 / 100.0

    # RS normalisation: RS=0.90→0, RS=1.00→0.5, RS=1.10→1.0
    rs_norm    = 0.5  # neutral default when no benchmark data
    if metrics.avg_rs_3m is not None:
        rs_norm = min(max((metrics.avg_rs_3m - 0.90) / 0.20, 0.0), 1.0)

    score = (
        0.25 * bull_norm               +
        0.20 * metrics.avg_trend_score  +
        0.15 * metrics.avg_momentum_score +
        0.20 * ema_norm                +
        0.10 * rs_norm                 +
        0.10 * metrics.avg_quality_score
    )
    return round(min(max(score, 0.0), 1.0), 4)


# ─────────────────────────────────────────────────────────────────────────────
#  Private helpers
# ─────────────────────────────────────────────────────────────────────────────

def _stable_regime(r) -> str:
    """Extract stable_regime string with fallback to raw stock_regime."""
    if hasattr(r, "stable_regime"):
        val = r.stable_regime.value
        return val if val != "UNCERTAIN" else r.stock_regime.value
    return r.stock_regime.value


def _safe_mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0