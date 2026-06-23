"""
stock_regime/ranking/diagnostics.py
=====================================
Structured diagnostic logging for the opportunity ranking layer.

FIX 2 — Market alignment diagnostics
---------------------------------------
log_score_breakdown() and _log_component_contributions() now show:
  - market alignment SCORE       (the raw [0,1] component value)
  - market alignment CONTRIBUTION (weight × score × 100 — points earned)
  - market PENALTY applied        (weight × (1-score) × 100 — points lost)

FIX 3 — Relative strength diagnostics
----------------------------------------
log_ranking_summary() gained a "Relative Strength Leaders" section showing
the top RS stocks in the batch by blended rs_score, with their underlying
rs_level_score and rs_trend_score broken out — this is the section that
should visibly show momentum leaders like HFCL/STLTECH/POLYCAB rising.

log_score_breakdown() now shows:
  - RS Score (blended)
  - RS Level Score / RS Trend Score (the two raw sub-components)
  - RS Contribution (points earned)
  - RS Weight (the configured weight, as %)
"""

from __future__ import annotations

import logging
from datetime import date
from typing import Optional

from .models import RankedOpportunity, ScoreWeights

logger = logging.getLogger(__name__)


def log_ranking_summary(
    ranked:        list[RankedOpportunity],
    universe:      str,
    run_date:      date,
    market_regime: str,
    breadth_state: str,
    top_n:         int = 20,
    weights:       Optional[ScoreWeights] = None,
    rs_leaders_n:  int = 10,
) -> None:
    """
    Log a complete ranking diagnostic summary.

    Logs:
    1. Score distribution (mean, std, percentiles)
    2. Component contribution analysis (incl. market alignment + RS, FIX 2/3)
    3. Strategy distribution across ranked set
    4. Sector concentration in top N
    5. Relative Strength Leaders (FIX 3) — top stocks by blended RS score
    6. Market alignment penalty summary (FIX 2) — who is losing the most
       points to market misalignment right now
    7. Top N opportunities
    """
    if not ranked:
        logger.warning("RankingDiagnostics [%s]: no opportunities to rank.", universe)
        return

    scores   = [r.opportunity_score for r in ranked]
    n        = len(scores)
    mean_s   = sum(scores) / n
    variance = sum((s - mean_s) ** 2 for s in scores) / max(n - 1, 1)
    std_s    = variance ** 0.5
    sorted_s = sorted(scores)
    p25      = sorted_s[int(n * 0.25)]
    p75      = sorted_s[int(n * 0.75)]
    p90      = sorted_s[int(n * 0.90)]

    logger.info("=" * 65)
    logger.info(
        "Opportunity Rankings [%s]  %s  n=%d  market=%s  breadth=%s",
        universe, run_date, n, market_regime, breadth_state,
    )
    logger.info("=" * 65)

    logger.info(
        "Score distribution:  mean=%.1f  std=%.1f  P25=%.1f  P75=%.1f  P90=%.1f",
        mean_s, std_s, p25, p75, p90,
    )

    if weights:
        _log_component_contributions(ranked, weights)

    # ── Strategy distribution ───────────────────────────────────
    strategy_counts: dict[str, int] = {}
    for r in ranked:
        strategy_counts[r.strategy] = strategy_counts.get(r.strategy, 0) + 1
    logger.info("Strategy mix (ranked set):")
    for strat, count in sorted(strategy_counts.items(), key=lambda x: -x[1]):
        pct = count / n * 100
        bar = "▓" * min(int(pct / 5), 20)
        logger.info("  %-18s %3d  (%.0f%%)  %s", strat, count, pct, bar)

    # ── Sector distribution in top N ───────────────────────────
    top      = ranked[:top_n]
    sector_c: dict[str, int] = {}
    for r in top:
        sector_c[r.sector] = sector_c.get(r.sector, 0) + 1
    logger.info("Sector concentration (top %d):", top_n)
    for sector, count in sorted(sector_c.items(), key=lambda x: -x[1])[:8]:
        logger.info("  %-25s %3d", sector, count)

    # ── FIX 3: Relative Strength Leaders ─────────────────────────
    logger.info("─" * 65)
    rs_leaders = sorted(ranked, key=lambda r: r.component_scores.rs_score, reverse=True)
    logger.info("Relative Strength Leaders (top %d by RS score):", rs_leaders_n)
    logger.info(
        "  %-14s %-9s %-9s %-9s %-10s %-6s",
        "Symbol", "RS_Score", "RS_Level", "RS_Trend", "RS_Contrib", "Rank",
    )
    for r in rs_leaders[:rs_leaders_n]:
        cs = r.component_scores
        logger.info(
            "  %-14s %-9.3f %-9.3f %-9.3f %-10.2f #%-5d",
            r.symbol, cs.rs_score, cs.rs_level_score, cs.rs_trend_score,
            r.rs_contribution(), r.rank,
        )

    # ── FIX 2: Market alignment penalty summary ──────────────────
    logger.info("─" * 65)
    misaligned = sorted(ranked, key=lambda r: r.market_alignment_penalty(), reverse=True)
    misaligned_with_penalty = [r for r in misaligned if r.market_alignment_penalty() > 0.5]
    if misaligned_with_penalty:
        logger.info(
            "Largest market-alignment penalties (market=%s):", market_regime
        )
        logger.info(
            "  %-14s %-12s %-9s %-10s",
            "Symbol", "Regime", "Align", "Penalty",
        )
        for r in misaligned_with_penalty[:rs_leaders_n]:
            logger.info(
                "  %-14s %-12s %-9.2f %-10.2f",
                r.symbol, r.stable_regime,
                r.component_scores.market_alignment,
                r.market_alignment_penalty(),
            )
    else:
        logger.info("Market alignment: no significant penalties — ranked set is well-aligned with %s.", market_regime)

    # ── Top N opportunities ─────────────────────────────────────
    logger.info("─" * 65)
    logger.info("Top %d Opportunities:", min(top_n, n))
    logger.info(
        "  %-4s %-14s %-10s %-8s %-18s %-22s %-14s",
        "Rank", "Symbol", "Score", "PSM", "Strategy", "Sector", "DominantFactor",
    )
    logger.info("  " + "─" * 100)
    for r in ranked[:top_n]:
        logger.info(
            "  #%-3d %-14s %5.1f     %-6.2f  %-18s %-22s %-14s",
            r.rank,
            r.symbol,
            r.opportunity_score,
            r.psm,
            r.strategy,
            r.sector[:22],
            r.dominant_factor,
        )
    logger.info("─" * 65)


def log_score_breakdown(opportunity: RankedOpportunity) -> None:
    """
    Log the full component breakdown for a single opportunity.

    FIX 2/3: now shows market alignment penalty and RS sub-components
    explicitly, with both raw scores and point contributions.
    """
    cs = opportunity.component_scores
    w  = opportunity.weights_used

    logger.info(
        "[%s] Score=%.1f  Rank=#%d  Strategy=%s  Sector=%s  Regime=%s",
        opportunity.symbol,
        opportunity.opportunity_score,
        opportunity.rank,
        opportunity.strategy,
        opportunity.sector,
        opportunity.stable_regime,
    )
    logger.info(
        "  Components:  trend=%.2f  mom=%.2f  quality=%.2f  "
        "mkt_align=%.2f  breadth=%.2f  sector=%.2f  stability=%.2f",
        cs.trend, cs.momentum, cs.quality,
        cs.market_alignment, cs.breadth_alignment,
        cs.sector_strength, cs.stability,
    )
    logger.info(
        "  Market Alignment:  score=%.2f  weight=%.0f%%  "
        "contribution=%.2f pts  PENALTY=%.2f pts",
        cs.market_alignment, w.market_alignment * 100,
        opportunity.component_contribution("market_alignment"),
        opportunity.market_alignment_penalty(),
    )
    logger.info(
        "  Relative Strength: rs_score=%.2f (level=%.2f, trend=%.2f)  "
        "weight=%.0f%%  contribution=%.2f pts",
        cs.rs_score, cs.rs_level_score, cs.rs_trend_score,
        w.rs_score * 100, opportunity.rs_contribution(),
    )
    logger.info("  Dominant factor: %s", opportunity.dominant_factor)


def _log_component_contributions(
    ranked:  list[RankedOpportunity],
    weights: ScoreWeights,
) -> None:
    """
    Log which components contribute most to average scores, including
    FIX 2 (market alignment) and FIX 3 (RS) which are now first-class
    components rather than afterthoughts.
    """
    if not ranked:
        return
    w = weights.normalised()
    n = len(ranked)

    mean_components = {
        "trend":             sum(r.component_scores.trend             for r in ranked) / n,
        "momentum":          sum(r.component_scores.momentum          for r in ranked) / n,
        "quality":           sum(r.component_scores.quality           for r in ranked) / n,
        "market_alignment":  sum(r.component_scores.market_alignment  for r in ranked) / n,
        "breadth_alignment": sum(r.component_scores.breadth_alignment for r in ranked) / n,
        "sector_strength":   sum(r.component_scores.sector_strength   for r in ranked) / n,
        "stability":         sum(r.component_scores.stability         for r in ranked) / n,
        "rs_score":          sum(r.component_scores.rs_score          for r in ranked) / n,
    }
    weight_map = {
        "trend":             w.trend,
        "momentum":          w.momentum,
        "quality":           w.quality,
        "market_alignment":  w.market_alignment,
        "breadth_alignment": w.breadth_alignment,
        "sector_strength":   w.sector_strength,
        "stability":         w.stability,
        "rs_score":          w.rs_score,
    }

    logger.info("Component contributions (mean score × weight, on 0-100 scale):")
    contrib_items = sorted(
        [(k, mean_components[k] * weight_map[k] * 100) for k in mean_components],
        key=lambda x: -x[1],
    )
    for comp, contrib in contrib_items:
        mean_c = mean_components[comp]
        weight = weight_map[comp]
        bar    = "▓" * min(int(contrib / 2), 20)
        logger.info(
            "  %-18s  score=%.2f  weight=%4.1f%%  contrib=%5.2f pts  %s",
            comp, mean_c, weight * 100, contrib, bar,
        )

    # Average market alignment penalty across the batch — FIX 2 visibility.
    avg_penalty = sum(r.market_alignment_penalty() for r in ranked) / n
    logger.info(
        "  → Average market-alignment penalty across ranked set: %.2f pts "
        "(max possible: %.2f pts at weight=%.0f%%)",
        avg_penalty, w.market_alignment * 100, w.market_alignment * 100,
    )