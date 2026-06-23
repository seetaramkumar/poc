"""
stock_regime/ranking/composite_scorer.py
==========================================
Pure functions that compute composite opportunity scores.

FIX 2 — Strengthened Market Alignment Scoring
-----------------------------------------------
Replaced the ad-hoc alignment rule list with a full, configurable
MARKET_ALIGNMENT_MATRIX: market_regime → {stock_regime: score}. Every
market regime now has an explicit score for every stock regime (not just
a handful of special cases with a single neutral fallback), matching the
example logic from the spec:

    BULLISH_TREND:  TREND_UP=1.0  MOMENTUM=0.9  BREAKOUT_SETUP=1.0  RANGE=0.4  TREND_DOWN=0.0
    BEARISH_TREND:  TREND_UP=0.3  MOMENTUM=0.2  BREAKOUT_SETUP=0.2  RANGE=0.5  TREND_DOWN=1.0
    SIDEWAYS:       RANGE=1.0     TREND_UP=0.5  MOMENTUM=0.4        BREAKOUT_SETUP=0.3
    VOLATILE:       VOLATILE=1.0  TREND_UP=0.4  MOMENTUM=0.3        BREAKOUT_SETUP=0.2

The matrix is loaded from config (ranking.market_alignment_matrix) if
present; otherwise the hardcoded default below is used. This makes
ranking behaviour visibly and immediately responsive to market regime
changes — a TREND_UP stock loses 0.7 of alignment score the moment the
market flips from BULLISH_TREND (1.0) to BEARISH_TREND (0.3).

FIX 3 — Relative Strength as a First-Class Ranking Component
----------------------------------------------------------------
RS is now a two-part, percentile-normalised score computed across the
WHOLE universe batch (not a single narrow-band clip on one stock at a
time, which saturated at 1.0 for every momentum leader regardless of how
extreme its RS actually was — HFCL at RS=2.81 and STLTECH at RS=3.30 were
indistinguishable under the old clip).

  rs_score = 0.70 × percentile_rank(rs_3m   across universe)
           + 0.30 × percentile_rank(rs_trend across universe)

Percentile rank naturally handles long-tailed RS distributions: the
strongest momentum leader in the universe scores close to 1.0, the
weakest scores close to 0.0, regardless of the absolute scale of RS
values. When the batch is too small for a stable percentile estimate
(< min_universe_for_percentile, default 5), a clipped min/max scaling
fallback is used instead.

Normalisation conventions (unchanged)
---------------------------------------
All components are normalised to [0, 1] before weighting. The final
opportunity_score = weighted_sum × 100, giving [0, 100].

Sector state → score mapping (unchanged)
-------------------------------------------
LEADING → 1.00  STRONG → 0.80  NEUTRAL → 0.60
WEAKENING → 0.35  WEAK → 0.15  UNKNOWN/INSUFFICIENT_DATA → 0.50
"""

from __future__ import annotations

import logging
import math
from typing import Optional

from .models import ComponentScores, CompositeOpportunityScore, ScoreWeights

logger = logging.getLogger(__name__)

# ── Sector state → normalised score ─────────────────────────
_SECTOR_STATE_SCORE: dict[str, float] = {
    "LEADING":           1.00,
    "STRONG":            0.80,
    "NEUTRAL":           0.60,
    "WEAKENING":         0.35,
    "WEAK":              0.15,
    "UNKNOWN":           0.50,
    "INSUFFICIENT_DATA": 0.50,   # FIX 1 interop: thin sectors are neutral, not penalised
}

# ── FIX 2: Full configurable market alignment matrix ─────────
# market_regime → {stock_regime: alignment_score [0,1]}
# Every (market, stock_regime) combination used by the engine has an
# explicit entry. Combinations not listed fall back to the market's
# "_default" entry, then to the global _ALIGNMENT_DEFAULT.
DEFAULT_MARKET_ALIGNMENT_MATRIX: dict[str, dict[str, float]] = {
    "BULLISH_TREND": {
        "TREND_UP":       1.00,
        "MOMENTUM":       0.90,
        "BREAKOUT_SETUP": 1.00,
        "RANGE":          0.40,
        "QUIET":          0.30,
        "VOLATILE":       0.20,
        "TREND_DOWN":     0.00,
        "UNCERTAIN":      0.50,
        "_default":       0.50,
    },
    "BEARISH_TREND": {
        "TREND_DOWN":     1.00,
        "RANGE":          0.50,
        "QUIET":          0.40,
        "VOLATILE":       0.30,
        "TREND_UP":       0.30,
        "MOMENTUM":       0.20,
        "BREAKOUT_SETUP": 0.20,
        "UNCERTAIN":      0.50,
        "_default":       0.50,
    },
    "SIDEWAYS": {
        "RANGE":          1.00,
        "QUIET":          0.70,
        "TREND_UP":       0.50,
        "MOMENTUM":       0.40,
        "BREAKOUT_SETUP": 0.30,
        "TREND_DOWN":     0.30,
        "VOLATILE":       0.20,
        "UNCERTAIN":      0.50,
        "_default":       0.50,
    },
    "VOLATILE": {
        "VOLATILE":       1.00,
        "TREND_UP":       0.40,
        "MOMENTUM":       0.30,
        "RANGE":          0.30,
        "TREND_DOWN":     0.40,
        "BREAKOUT_SETUP": 0.20,
        "QUIET":          0.20,
        "UNCERTAIN":      0.50,
        "_default":       0.50,
    },
    "QUIET": {
        "QUIET":          1.00,
        "RANGE":          0.70,
        "TREND_UP":       0.40,
        "MOMENTUM":       0.30,
        "BREAKOUT_SETUP": 0.30,
        "TREND_DOWN":     0.30,
        "VOLATILE":       0.10,
        "UNCERTAIN":      0.50,
        "_default":       0.50,
    },
    "UNCERTAIN": {
        # No clear market signal — neither boost nor penalise any regime.
        "_default": 0.50,
    },
}

_ALIGNMENT_DEFAULT = 0.50  # absolute fallback if market_regime itself is unrecognised


def load_market_alignment_matrix(config) -> dict[str, dict[str, float]]:
    """
    Load the market alignment matrix from config.ranking.market_alignment_matrix
    if present, merging it OVER the hardcoded defaults (so a partial config
    override only needs to specify the cells it wants to change).

    Falls back entirely to DEFAULT_MARKET_ALIGNMENT_MATRIX when no config
    section is present.
    """
    matrix = {
        market: dict(rules) for market, rules in DEFAULT_MARKET_ALIGNMENT_MATRIX.items()
    }
    rk = getattr(config, "ranking", None) if config is not None else None
    cfg_matrix = getattr(rk, "market_alignment_matrix", None) if rk else None
    if cfg_matrix is None:
        return matrix

    # cfg_matrix is a _NS-style nested namespace: market → stock_regime → score
    for market_attr, market_ns in vars(cfg_matrix).items() if hasattr(cfg_matrix, "__dict__") else []:
        market_key = market_attr
        matrix.setdefault(market_key, {"_default": _ALIGNMENT_DEFAULT})
        if hasattr(market_ns, "__dict__"):
            for regime_attr, score in vars(market_ns).items():
                try:
                    matrix[market_key][regime_attr] = float(score)
                except (TypeError, ValueError):
                    continue
    return matrix


def compute_market_alignment(
    market_regime: str,
    stock_regime:  str,
    matrix:        Optional[dict[str, dict[str, float]]] = None,
) -> float:
    """
    Look up the alignment score for (market_regime, stock_regime) from the
    matrix. Falls back to the market's own "_default" entry, then to the
    global _ALIGNMENT_DEFAULT if the market itself is unrecognised.
    """
    matrix = matrix or DEFAULT_MARKET_ALIGNMENT_MATRIX
    rules  = matrix.get(market_regime)
    if rules is None:
        return _ALIGNMENT_DEFAULT
    if stock_regime in rules:
        return rules[stock_regime]
    return rules.get("_default", _ALIGNMENT_DEFAULT)


def compute_component_scores(
    stable_result,                       # StableRegimeResult
    quality_score:    float,             # from OpportunityQualityEngine
    breadth_score:    float,             # BreadthSnapshot.regime_breadth_score [0,1]
    sector_state:     str,               # from SectorEngine
    market_regime:    str,               # from MarketRegimeEngine
    alignment_matrix: Optional[dict[str, dict[str, float]]] = None,
    rs_score_override: Optional[float] = None,
    rs_level_score:    float = 0.5,
    rs_trend_score:    float = 0.5,
) -> ComponentScores:
    """
    Derive all component scores from their respective engine outputs.

    All components land in [0, 1].

    Parameters
    ----------
    stable_result :
        A StableRegimeResult (or compatible object).
    quality_score :
        Composite quality score [0, 1] from OpportunityQualityEngine.
    breadth_score :
        BreadthSnapshot.regime_breadth_score [0, 1].
    sector_state :
        Sector state string: LEADING/STRONG/NEUTRAL/WEAKENING/WEAK/
        INSUFFICIENT_DATA/UNKNOWN.
    market_regime :
        Market regime string: BULLISH_TREND/BEARISH_TREND/SIDEWAYS/etc.
    alignment_matrix :
        FIX 2 — configurable market×regime alignment matrix. Defaults to
        DEFAULT_MARKET_ALIGNMENT_MATRIX when not supplied.
    rs_score_override :
        FIX 3 — when provided (computed by compute_rs_scores_batch across
        the full universe), this value is used directly as rs_score instead
        of the old single-stock narrow-band clip. This is the PRIMARY path
        used by score_batch(). When None, falls back to a clipped min/max
        scaling of rs_3m for standalone/single-stock callers (e.g. tests).
    rs_level_score, rs_trend_score :
        Diagnostic-only sub-components carried through to ComponentScores
        for transparency (not separately weighted).
    """
    ds   = stable_result.dimensional_scores
    snap = stable_result.indicators

    trend    = _clip(float(ds.trend))
    momentum = _clip(float(ds.momentum))
    quality  = _clip(float(quality_score))

    regime_str = (
        stable_result.stable_regime.value
        if hasattr(stable_result, "stable_regime")
        else stable_result.stock_regime.value
    )
    market_alignment = compute_market_alignment(
        market_regime, regime_str, alignment_matrix
    )

    breadth_alignment = _clip(float(breadth_score))
    sector_strength    = _SECTOR_STATE_SCORE.get(sector_state, 0.50)

    conf = float(
        stable_result.smoothed_confidence
        if hasattr(stable_result, "smoothed_confidence")
        else stable_result.confidence
    )
    age = int(
        stable_result.stable_regime_age
        if hasattr(stable_result, "stable_regime_age")
        else 0
    )
    oscillating = bool(getattr(stable_result, "oscillation_detected", False))
    age_bonus   = min(age / 20.0, 1.0) * 0.30
    osc_penalty = 0.20 if oscillating else 0.0
    stability   = _clip(conf * 0.70 + age_bonus - osc_penalty)

    # ── FIX 3: RS score ────────────────────────────────────────────
    if rs_score_override is not None:
        rs_score = _clip(float(rs_score_override))
    else:
        # Fallback for single-stock callers without a universe batch:
        # clipped min/max scaling over a wider band than the old [0.90,1.10]
        # since momentum leaders can legitimately exceed rs_3m=2.0-3.0.
        rs_val = getattr(snap, "rs_3m", None) or getattr(snap, "relative_strength", None)
        rs_score = 0.50
        if rs_val is not None and not _is_nan(rs_val):
            rs_score = _clip((float(rs_val) - 0.85) / (1.30 - 0.85))

    return ComponentScores(
        trend             = round(trend,             4),
        momentum          = round(momentum,          4),
        quality           = round(quality,           4),
        market_alignment  = round(market_alignment,  4),
        breadth_alignment = round(breadth_alignment, 4),
        sector_strength   = round(sector_strength,   4),
        stability         = round(stability,         4),
        rs_score          = round(rs_score,          4),
        rs_level_score    = round(_clip(rs_level_score), 4),
        rs_trend_score    = round(_clip(rs_trend_score), 4),
    )


def compute_composite(
    symbol:           str,
    market:           str,
    run_date,
    component_scores: ComponentScores,
    weights:          ScoreWeights,
) -> CompositeOpportunityScore:
    """
    Compute the composite opportunity score from component scores and weights.

    Returns CompositeOpportunityScore with opportunity_score in [0, 100].

    FIX 2: weights_used now stores the NORMALISED weights (summing to
    exactly 1.0), not the raw config weights, so every diagnostic
    contribution/penalty figure reconciles exactly with opportunity_score.
    """
    w = weights.normalised()

    raw = (
        w.trend             * component_scores.trend             +
        w.momentum          * component_scores.momentum          +
        w.quality           * component_scores.quality           +
        w.market_alignment  * component_scores.market_alignment  +
        w.breadth_alignment * component_scores.breadth_alignment +
        w.sector_strength   * component_scores.sector_strength   +
        w.stability         * component_scores.stability         +
        w.rs_score          * component_scores.rs_score
    )

    opportunity_score = round(_clip(raw) * 100.0, 2)

    return CompositeOpportunityScore(
        symbol            = symbol,
        market            = market,
        run_date          = run_date,
        opportunity_score = opportunity_score,
        component_scores  = component_scores,
        weights_used      = w,          # normalised, not raw
    )


def compute_rs_scores_batch(
    stable_results:               list,           # list[StableRegimeResult]
    level_weight:                 float = 0.70,
    trend_weight:                 float = 0.30,
    min_universe_for_percentile:  int   = 5,
    fallback_level_min:           float = 0.85,
    fallback_level_max:           float = 1.30,
    fallback_trend_min:           float = -0.005,
    fallback_trend_max:           float = 0.005,
) -> dict[str, tuple[float, float, float]]:
    """
    FIX 3 — Compute percentile-ranked RS scores across the WHOLE universe batch.

    For each stock, returns (rs_score, rs_level_score, rs_trend_score) where:
        rs_level_score = percentile_rank(rs_3m)    across stocks with rs_3m available
        rs_trend_score = percentile_rank(rs_trend) across stocks with rs_trend available
        rs_score       = level_weight * rs_level_score + trend_weight * rs_trend_score

    Percentile rank is used because RS distributions are long-tailed —
    momentum leaders can have rs_3m of 2.0–3.5 while the median stock sits
    near 1.0. A fixed clip band saturates every leader to the same score;
    percentile rank correctly differentiates HFCL(2.81) from STLTECH(3.30).

    Stocks missing rs_3m or rs_trend get a neutral 0.5 for that sub-score
    and are EXCLUDED from the percentile population (so a handful of
    missing-data stocks can't compress everyone else's percentiles).

    When fewer than `min_universe_for_percentile` stocks have valid RS data,
    percentile ranking is statistically unstable (e.g. with only 2 stocks,
    one is always "100th percentile"). In that case this function falls
    back to clipped min/max scaling using the fallback_* bounds.

    Returns
    -------
    dict[symbol, (rs_score, rs_level_score, rs_trend_score)]
        Missing symbols (no valid stable_result) default to (0.5, 0.5, 0.5)
        when looked up by the caller.
    """
    # ── Collect raw values ──────────────────────────────────────────
    level_values: dict[str, float] = {}
    trend_values: dict[str, float] = {}

    for r in stable_results:
        if not r.is_valid():
            continue
        snap = r.indicators
        rs_3m   = getattr(snap, "rs_3m", None)
        rs_tr   = getattr(snap, "rs_trend", None)
        if rs_3m is not None and not _is_nan(rs_3m):
            level_values[r.symbol] = float(rs_3m)
        if rs_tr is not None and not _is_nan(rs_tr):
            trend_values[r.symbol] = float(rs_tr)

    use_percentile_level = len(level_values) >= min_universe_for_percentile
    use_percentile_trend = len(trend_values) >= min_universe_for_percentile

    if not use_percentile_level:
        logger.debug(
            "RS level: only %d stocks have rs_3m (< min_universe_for_percentile=%d) "
            "— using clipped min/max fallback instead of percentile rank.",
            len(level_values), min_universe_for_percentile,
        )
    if not use_percentile_trend:
        logger.debug(
            "RS trend: only %d stocks have rs_trend (< min_universe_for_percentile=%d) "
            "— using clipped min/max fallback instead of percentile rank.",
            len(trend_values), min_universe_for_percentile,
        )

    level_population = list(level_values.values())
    trend_population  = list(trend_values.values())

    result: dict[str, tuple[float, float, float]] = {}
    for r in stable_results:
        if not r.is_valid():
            continue
        sym = r.symbol

        if sym in level_values:
            if use_percentile_level:
                level_score = _percentile_rank(level_values[sym], level_population)
            else:
                level_score = _clip(
                    (level_values[sym] - fallback_level_min) /
                    max(fallback_level_max - fallback_level_min, 1e-9)
                )
        else:
            level_score = 0.50

        if sym in trend_values:
            if use_percentile_trend:
                trend_score = _percentile_rank(trend_values[sym], trend_population)
            else:
                trend_score = _clip(
                    (trend_values[sym] - fallback_trend_min) /
                    max(fallback_trend_max - fallback_trend_min, 1e-9)
                )
        else:
            trend_score = 0.50

        blended = _clip(level_weight * level_score + trend_weight * trend_score)
        result[sym] = (
            round(blended,     4),
            round(level_score, 4),
            round(trend_score, 4),
        )

    return result


def score_batch(
    stable_results:   list,             # list[StableRegimeResult]
    quality_map:      dict[str, float], # symbol → quality_score
    breadth_score:    float,
    sector_state_map: dict[str, str],   # symbol → sector_state
    market_regime:    str,
    weights:          ScoreWeights,
    run_date,
    alignment_matrix: Optional[dict[str, dict[str, float]]] = None,
    rs_level_weight:  float = 0.70,
    rs_trend_weight:  float = 0.30,
) -> list[CompositeOpportunityScore]:
    """
    Score a full batch of classified stocks.

    FIX 3: RS scores are computed ONCE across the whole batch via
    compute_rs_scores_batch() (percentile ranking requires the full
    population), then injected into each stock's component scores via
    rs_score_override.

    FIX 2: alignment_matrix is threaded through to every stock so a
    single config-driven matrix governs the whole batch's market
    alignment scoring.

    Returns scores sorted by opportunity_score descending.
    """
    rs_scores = compute_rs_scores_batch(
        stable_results,
        level_weight = rs_level_weight,
        trend_weight = rs_trend_weight,
    )

    scores: list[CompositeOpportunityScore] = []

    for r in stable_results:
        if not r.is_valid():
            continue
        quality      = quality_map.get(r.symbol, 0.0)
        sector_state = sector_state_map.get(r.symbol, "UNKNOWN")
        rs_blend, rs_level, rs_trend = rs_scores.get(r.symbol, (0.50, 0.50, 0.50))

        cs = compute_component_scores(
            stable_result      = r,
            quality_score      = quality,
            breadth_score      = breadth_score,
            sector_state       = sector_state,
            market_regime      = market_regime,
            alignment_matrix   = alignment_matrix,
            rs_score_override  = rs_blend,
            rs_level_score     = rs_level,
            rs_trend_score     = rs_trend,
        )
        score = compute_composite(
            symbol           = r.symbol,
            market           = r.market,
            run_date         = run_date,
            component_scores = cs,
            weights          = weights,
        )
        scores.append(score)

    scores.sort(key=lambda s: s.opportunity_score, reverse=True)
    return scores


# ─────────────────────────────────────────────────────────────────────────────
#  Private helpers
# ─────────────────────────────────────────────────────────────────────────────

def _clip(v: float) -> float:
    return min(max(v, 0.0), 1.0)


def _is_nan(v) -> bool:
    try:
        return math.isnan(float(v))
    except (TypeError, ValueError):
        return True


def _percentile_rank(value: float, population: list[float]) -> float:
    """
    Fraction of `population` that is <= value, with standard tie handling:
        (count_less + 0.5 * count_equal) / n

    Returns 0.5 for a single-element population (no information to rank against).
    This is the textbook "percentile rank" formula — robust to outliers and
    to the absolute scale of the underlying values, which is exactly what's
    needed for a long-tailed RS distribution.
    """
    n = len(population)
    if n <= 1:
        return 0.50
    count_less  = sum(1 for v in population if v < value)
    count_equal = sum(1 for v in population if v == value)
    return (count_less + 0.5 * count_equal) / n