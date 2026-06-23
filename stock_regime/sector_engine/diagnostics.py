"""
stock_regime/sector_engine/diagnostics.py
===========================================
Structured diagnostic logging for sector intelligence outputs.

FIX 1 — Minimum Sector Sample Size Validation
-----------------------------------------------
log_sector_summary() now:
  - Computes Top N / Bottom N ONLY from rankable sectors
    (excluded_from_ranking == False). A 1-stock sector can never appear
    in either list.
  - Adds a dedicated "Excluded (insufficient data)" section listing every
    sector below min_sector_stocks, with its actual stock_count, e.g.:
        TEXTILES        stocks=1  state=INSUFFICIENT_DATA
        CONSTRUCTION    stocks=2  state=INSUFFICIENT_DATA
  - State distribution counts INSUFFICIENT_DATA separately from the 5
    substantive states so the two concepts are never visually conflated.

build_sector_context_map() already filtered via is_actionable(), which
now also excludes INSUFFICIENT_DATA sectors — no change needed there,
but the docstring is updated for clarity.
"""

from __future__ import annotations

import logging
from datetime import date
from typing import Optional

from .models import SectorSnapshot, SectorState

logger = logging.getLogger(__name__)

_SUBSTANTIVE_STATES_ORDER = [
    SectorState.LEADING, SectorState.STRONG, SectorState.NEUTRAL,
    SectorState.WEAKENING, SectorState.WEAK,
]


def log_sector_summary(
    snapshots:     list[SectorSnapshot],
    universe:      str,
    run_date:      date,
    top_n:         int = 5,
    bottom_n:      int = 5,
) -> None:
    """
    Log a full sector intelligence summary.

    Logs:
    - Sector state distribution (substantive states + insufficient-data count)
    - Excluded sectors (insufficient data) — explicit listing
    - Top N sectors (rankable only)
    - Bottom N sectors (rankable only)
    - Participation summary (rankable only)
    - Sectors with deteriorating RS (rankable only)
    """
    if not snapshots:
        logger.warning("SectorDiagnostics [%s]: no sector data to log.", universe)
        return

    rankable = sorted(
        [s for s in snapshots if not s.excluded_from_ranking],
        key=lambda s: s.composite_score, reverse=True,
    )
    excluded = sorted(
        [s for s in snapshots if s.excluded_from_ranking],
        key=lambda s: s.sector,
    )
    total = len(snapshots)

    logger.info("=" * 60)
    logger.info(
        "Sector Intelligence [%s]  %s  (%d sectors, %d rankable, %d excluded)",
        universe, run_date, total, len(rankable), len(excluded),
    )
    logger.info("=" * 60)

    # ── State distribution (substantive states only) ───────────────
    state_counts: dict[str, int] = {}
    for snap in rankable:
        state_counts[snap.state.value] = state_counts.get(snap.state.value, 0) + 1

    logger.info("State distribution (rankable sectors):")
    for state in _SUBSTANTIVE_STATES_ORDER:
        count = state_counts.get(state.value, 0)
        if count > 0:
            logger.info("  %-12s  %2d  (%s)", state.value, count,
                        "▓" * count + "░" * max(0, 10 - count))

    # ── Excluded (insufficient data) ────────────────────────────────
    if excluded:
        logger.info("─" * 60)
        logger.info(
            "Excluded — insufficient data (< min_sector_stocks):"
        )
        for snap in excluded:
            logger.info(
                "  %-22s  stocks=%-3d  state=%s",
                snap.sector, snap.stock_count, snap.state.value,
            )

    if not rankable:
        logger.warning(
            "SectorDiagnostics [%s]: no sectors met the minimum sample "
            "size — nothing to rank.", universe,
        )
        logger.info("=" * 60)
        return

    # ── Top N (rankable only) ───────────────────────────────────────
    logger.info("─" * 60)
    logger.info("Top %d sectors (rankable):", min(top_n, len(rankable)))
    for snap in rankable[:top_n]:
        m = snap.metrics
        logger.info(
            "  #%-2d %-22s [%-9s]  composite=%.3f  bull=%.0f%%  "
            "ema200=%.0f%%  rs=%.3f  stocks=%d",
            snap.rank,
            snap.sector,
            snap.state.value,
            snap.composite_score,
            m.pct_bullish,
            m.pct_above_ema200,
            m.avg_rs_3m or 0.0,
            m.stock_count,
        )

    # ── Bottom N (rankable only) ─────────────────────────────────────
    logger.info("─" * 60)
    logger.info("Bottom %d sectors (rankable):", min(bottom_n, len(rankable)))
    for snap in rankable[-bottom_n:]:
        m = snap.metrics
        logger.info(
            "  #%-2d %-22s [%-9s]  composite=%.3f  bull=%.0f%%  "
            "bear=%.0f%%  stocks=%d",
            snap.rank,
            snap.sector,
            snap.state.value,
            snap.composite_score,
            m.pct_bullish,
            m.pct_bearish,
            m.stock_count,
        )

    # ── Participation summary (rankable only) ────────────────────────
    logger.info("─" * 60)
    total_stocks = sum(s.metrics.stock_count for s in rankable)
    excluded_stocks = sum(s.metrics.stock_count for s in excluded)
    leading_secs = [s for s in rankable if s.state == SectorState.LEADING]
    weak_secs    = [s for s in rankable if s.state in (SectorState.WEAK, SectorState.WEAKENING)]

    logger.info(
        "Participation: rankable_stocks=%d  excluded_stocks=%d  "
        "leading_sectors=%d  weak/weakening=%d",
        total_stocks, excluded_stocks, len(leading_secs), len(weak_secs),
    )

    # ── Momentum concentration (rankable only) ────────────────────────
    top_mom = sorted(
        [s for s in rankable if s.is_actionable()],
        key=lambda s: s.metrics.momentum_participation,
        reverse=True,
    )[:3]
    if top_mom:
        logger.info("Highest momentum participation:")
        for s in top_mom:
            logger.info(
                "  %-22s  momentum_pct=%.1f%%  breakout_pct=%.1f%%",
                s.sector,
                s.metrics.momentum_participation,
                s.metrics.breakout_participation,
            )

    # ── Weakening RS alert (rankable only) ────────────────────────────
    weakening_rs = [
        s for s in rankable
        if s.metrics.avg_rs_trend is not None and s.metrics.avg_rs_trend < -0.001
        and s.state in (SectorState.LEADING, SectorState.STRONG)
    ]
    if weakening_rs:
        logger.warning(
            "⚠ RS weakening in previously-strong sectors: %s",
            [s.sector for s in weakening_rs],
        )

    logger.info("=" * 60)


def build_sector_context_map(
    snapshots: list[SectorSnapshot],
) -> dict[str, dict]:
    """
    Build a lookup dict for the strategy router / ranking engine:
      {sector_name: {state, composite_score, rank, pct_bullish, avg_rs_3m}}

    FIX 1: is_actionable() now also excludes INSUFFICIENT_DATA sectors, so
    a thin sector is automatically invisible to the router and ranking
    engine — symbols mapped to it fall back to the router's NEUTRAL default
    rather than being marked LEADING or WEAK on a one-stock sample.
    """
    return {
        snap.sector: {
            "state":              snap.state.value,
            "composite_score":    snap.composite_score,
            "rank":               snap.rank,
            "pct_bullish":        snap.metrics.pct_bullish,
            "pct_above_ema200":   snap.metrics.pct_above_ema200,
            "avg_rs_3m":          snap.metrics.avg_rs_3m,
            "avg_momentum_score": snap.metrics.avg_momentum_score,
            "stock_count":        snap.metrics.stock_count,
        }
        for snap in snapshots
        if snap.is_actionable()
    }