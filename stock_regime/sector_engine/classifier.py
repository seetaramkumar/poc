"""
stock_regime/sector_engine/classifier.py
==========================================
Deterministic sector state classification with minimum sample size validation.

FIX 1 — Minimum Sector Sample Size Validation
-----------------------------------------------
A sector with fewer than `min_sector_stocks` classified members is
classified as INSUFFICIENT_DATA regardless of its composite score. A
single strong or weak stock cannot represent a whole sector.

INSUFFICIENT_DATA sectors:
  - get excluded_from_ranking = True
  - get rank = 0 (never occupy an ordinal position)
  - are excluded from classify_batch()'s rank assignment loop entirely —
    ranks 1..N are assigned only among sectors that pass the sample size
    check, so a thin sector can never "push down" a real sector's rank.

State definitions (substantive states require stock_count >= min_sector_stocks)
---------------------------------------------------------------------------------
LEADING            composite >= leading_threshold   AND pct_bullish >= leading_bull_pct
STRONG             composite >= strong_threshold     (below LEADING cutoff)
NEUTRAL            composite >= neutral_threshold    (below STRONG cutoff)
WEAKENING          composite >= weakening_threshold  (below NEUTRAL cutoff)
WEAK               composite <  weakening_threshold
INSUFFICIENT_DATA  stock_count < min_sector_stocks   (sample size gate — checked FIRST)
"""

from __future__ import annotations

import logging

from .models import SectorMetrics, SectorSnapshot, SectorState

logger = logging.getLogger(__name__)


class SectorClassifier:
    """
    Classifies sector state from SectorMetrics using configurable thresholds.

    Parameters
    ----------
    leading_threshold : float
        Composite score above which a sector may be LEADING.
    leading_bull_pct : float
        Minimum pct_bullish required for LEADING.
    strong_threshold : float
        Composite score above which sector is STRONG.
    neutral_threshold : float
        Composite score above which sector is NEUTRAL.
    weakening_threshold : float
        Composite score above which sector is WEAKENING. Below → WEAK.
    min_sector_stocks : int
        Minimum classified stocks required for a substantive classification.
        Sectors below this threshold are classified INSUFFICIENT_DATA and
        excluded from ranking. Default: 5.
    """

    def __init__(
        self,
        leading_threshold:   float = 0.65,
        leading_bull_pct:    float = 55.0,
        strong_threshold:    float = 0.50,
        neutral_threshold:   float = 0.38,
        weakening_threshold: float = 0.25,
        min_sector_stocks:   int   = 5,
    ) -> None:
        self.leading_threshold   = leading_threshold
        self.leading_bull_pct    = leading_bull_pct
        self.strong_threshold    = strong_threshold
        self.neutral_threshold   = neutral_threshold
        self.weakening_threshold = weakening_threshold
        self.min_sector_stocks   = min_sector_stocks

    @classmethod
    def from_config(cls, config) -> "SectorClassifier":
        """
        Build from the sector_engine: section of config.yaml.

        Reads `min_sector_stocks` (new name). Falls back to the legacy
        `min_stocks` key for backward compatibility with existing config
        files, then to the hardcoded default of 5.
        """
        sc = getattr(config, "sector_engine", None)
        if sc is None:
            return cls()

        legacy_min = getattr(sc, "min_stocks", None)          # old key, if present
        min_sector_stocks = getattr(sc, "min_sector_stocks", None)
        if min_sector_stocks is None:
            min_sector_stocks = legacy_min if legacy_min is not None else 5

        return cls(
            leading_threshold   = float(getattr(sc, "leading_threshold",   0.65)),
            leading_bull_pct    = float(getattr(sc, "leading_bull_pct",    55.0)),
            strong_threshold    = float(getattr(sc, "strong_threshold",    0.50)),
            neutral_threshold   = float(getattr(sc, "neutral_threshold",   0.38)),
            weakening_threshold = float(getattr(sc, "weakening_threshold", 0.25)),
            min_sector_stocks   = int(min_sector_stocks),
        )

    def classify(
        self,
        metrics:         SectorMetrics,
        composite_score: float,
    ) -> SectorSnapshot:
        """
        Assign a SectorState to the sector given its metrics and composite score.

        The minimum sample size check is applied FIRST and short-circuits
        every other rule — a sector below `min_sector_stocks` is always
        INSUFFICIENT_DATA regardless of how extreme its composite score is.

        Returns
        -------
        SectorSnapshot with state, composite_score, excluded_from_ranking,
        and rank=0 (rank is assigned by classify_batch() for sectors that
        pass the sample size check).
        """
        insufficient = metrics.stock_count < self.min_sector_stocks

        if insufficient:
            state = SectorState.INSUFFICIENT_DATA
            logger.info(
                "Sector %-20s  stocks=%-3d  state=%s  (min required: %d) "
                "— excluded from rankings",
                metrics.sector, metrics.stock_count,
                state.value, self.min_sector_stocks,
            )
        else:
            state = self._classify_state(metrics, composite_score)
            logger.debug(
                "Sector %-20s  state=%-10s  composite=%.3f  "
                "bull=%.0f%%  ema200=%.0f%%  rs=%.3f  stocks=%d",
                metrics.sector, state.value, composite_score,
                metrics.pct_bullish, metrics.pct_above_ema200,
                metrics.avg_rs_3m or 0.0,
                metrics.stock_count,
            )

        return SectorSnapshot(
            metrics               = metrics,
            state                 = state,
            rank                  = 0,            # assigned by classify_batch()
            composite_score       = composite_score,
            excluded_from_ranking = insufficient,
        )

    def classify_batch(
        self,
        metrics_list:     list[SectorMetrics],
        composite_scores: dict[str, float],   # sector → composite_score
    ) -> list[SectorSnapshot]:
        """
        Classify all sectors and assign ordinal ranks.

        FIX 1: Ranks 1..N are assigned ONLY among sectors that pass the
        minimum sample size check (excluded_from_ranking == False).
        INSUFFICIENT_DATA sectors always get rank = 0 and are sorted to
        the end of the returned list — they never occupy or displace a
        real ordinal position.

        Returns
        -------
        list[SectorSnapshot]
            Ranked sectors first (by composite_score descending, rank 1..N),
            followed by excluded (INSUFFICIENT_DATA) sectors with rank=0.
        """
        snapshots: list[SectorSnapshot] = []
        for m in metrics_list:
            score = composite_scores.get(m.sector, 0.0)
            snap  = self.classify(m, score)
            snapshots.append(snap)

        rankable  = [s for s in snapshots if not s.excluded_from_ranking]
        excluded  = [s for s in snapshots if s.excluded_from_ranking]

        rankable.sort(key=lambda s: s.composite_score, reverse=True)
        for i, snap in enumerate(rankable):
            snap.rank = i + 1

        # Excluded sectors keep rank=0 (set in classify()) and are appended
        # after the rankable list — visible in full output but never ranked.
        excluded.sort(key=lambda s: s.sector)   # deterministic, alphabetical

        if excluded:
            logger.info(
                "SectorClassifier: %d sector(s) excluded from ranking "
                "(below min_sector_stocks=%d): %s",
                len(excluded), self.min_sector_stocks,
                [f"{s.sector}({s.stock_count})" for s in excluded],
            )

        return rankable + excluded

    # ──────────────────────────────────────────────────────────────
    #  State classification rules (only reached when sample size passes)
    # ──────────────────────────────────────────────────────────────

    def _classify_state(
        self,
        metrics:         SectorMetrics,
        composite_score: float,
    ) -> SectorState:
        # NOTE: the min_sector_stocks gate is checked in classify() BEFORE
        # this method is called — by the time we get here, stock_count is
        # guaranteed >= min_sector_stocks.

        if (composite_score >= self.leading_threshold and
                metrics.pct_bullish >= self.leading_bull_pct):
            return SectorState.LEADING

        if composite_score >= self.strong_threshold:
            return SectorState.STRONG

        if composite_score >= self.neutral_threshold:
            return SectorState.NEUTRAL

        if composite_score >= self.weakening_threshold:
            return SectorState.WEAKENING

        return SectorState.WEAK