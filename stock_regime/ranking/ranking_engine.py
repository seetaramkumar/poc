"""
stock_regime/ranking/ranking_engine.py
========================================
RankingEngine — orchestrates composite scoring, ranking, and persistence.

FIX 2 — Strengthened Market Alignment Scoring
-----------------------------------------------
RankingEngine now loads a configurable market_alignment_matrix via
composite_scorer.load_market_alignment_matrix() and threads it through
every score_batch() call, so the matrix in config.yaml is the single
source of truth for how strongly market regime shifts move rankings.

FIX 3 — Relative Strength as a First-Class Ranking Component
----------------------------------------------------------------
rs_level_weight / rs_trend_weight (default 0.70 / 0.30) are now
configurable and passed through to score_batch(), which computes
percentile-ranked RS scores across the whole universe before scoring
individual stocks.

Validation
----------
On construction, ScoreWeights.validate() is called and any warnings
(weight sum far from 1.0, a single component above 35%) are logged
immediately — config mistakes are caught at startup, not silently
baked into every ranking run.

Interface inputs (unchanged contract — all passed from pipeline, no
direct engine coupling here):
  stable_results      list[StableRegimeResult]
  quality_scores      list[QualityScore]
  routing_decisions   list[RoutingDecision]  — only ALLOWED ones enter ranking
  breadth_snapshot    BreadthSnapshot | None
  symbol_sector_map   dict[symbol → sector_state]   5-level string
  symbol_sector_name  dict[symbol → sector_name]    display name
  market_regime       str
  universe            str
"""

from __future__ import annotations

import logging
from datetime import date
from pathlib import Path
from typing import Optional

import pandas as pd

from .composite_scorer import (
    DEFAULT_MARKET_ALIGNMENT_MATRIX,
    load_market_alignment_matrix,
    score_batch,
)
from .diagnostics import log_ranking_summary, log_score_breakdown
from .models      import (
    CompositeOpportunityScore,
    RankedOpportunity,
    ScoreWeights,
)

logger = logging.getLogger(__name__)


class RankingEngine:
    """
    Composite opportunity ranking engine.

    Parameters
    ----------
    weights : ScoreWeights
        Config-driven weights for each scoring component.
    top_n : int
        How many top opportunities to show in diagnostic logs.
    min_score_to_log : float
        Stocks above this score [0-100] get per-stock breakdown logged.
    alignment_matrix : dict | None
        FIX 2 — configurable market×regime alignment matrix. If None,
        DEFAULT_MARKET_ALIGNMENT_MATRIX is used.
    rs_level_weight, rs_trend_weight : float
        FIX 3 — blend weights for the two RS sub-components. Default 0.70/0.30
        per the "70% RS level, 30% RS trend" specification.
    rs_leaders_n : int
        How many top-RS stocks to show in the "Relative Strength Leaders"
        diagnostic section.
    """

    def __init__(
        self,
        weights:          Optional[ScoreWeights] = None,
        top_n:            int   = 20,
        min_score_to_log: float = 70.0,
        alignment_matrix: Optional[dict] = None,
        rs_level_weight:  float = 0.70,
        rs_trend_weight:  float = 0.30,
        rs_leaders_n:     int   = 10,
    ) -> None:
        self.weights          = weights or ScoreWeights()
        self.top_n            = top_n
        self.min_score_to_log = min_score_to_log
        self.alignment_matrix = alignment_matrix or DEFAULT_MARKET_ALIGNMENT_MATRIX
        self.rs_level_weight  = rs_level_weight
        self.rs_trend_weight  = rs_trend_weight
        self.rs_leaders_n     = rs_leaders_n

        for warning in self.weights.validate():
            logger.warning("ScoreWeights validation: %s", warning)

    @classmethod
    def from_config(cls, config) -> "RankingEngine":
        """Build from the ranking: section of config.yaml."""
        weights = ScoreWeights.from_config(config)
        rk      = getattr(config, "ranking", None)
        matrix  = load_market_alignment_matrix(config)
        return cls(
            weights          = weights,
            top_n            = int(  getattr(rk, "top_n",             20)   if rk else 20),
            min_score_to_log = float(getattr(rk, "min_score_to_log",  70.0) if rk else 70.0),
            alignment_matrix = matrix,
            rs_level_weight  = float(getattr(rk, "rs_level_weight",   0.70) if rk else 0.70),
            rs_trend_weight  = float(getattr(rk, "rs_trend_weight",   0.30) if rk else 0.30),
            rs_leaders_n     = int(  getattr(rk, "rs_leaders_n",      10)   if rk else 10),
        )

    # ──────────────────────────────────────────────────────────────
    #  Primary API
    # ──────────────────────────────────────────────────────────────

    def rank(
        self,
        stable_results:     list,
        quality_scores:     list,
        routing_decisions:  list,
        breadth_snapshot,
        symbol_sector_map:  dict[str, str],
        symbol_sector_name: dict[str, str],
        market_regime:      str,
        universe:           str,
        run_date:           Optional[date] = None,
    ) -> list[RankedOpportunity]:
        """
        Score and rank all stocks that have an allowed routing decision.

        Only ALLOWED routing decisions are ranked — NO_TRADE stocks are
        excluded because ranking non-actionable stocks is misleading and
        would distort score distributions.

        Returns list[RankedOpportunity] sorted by opportunity_score desc.
        Rank 1 = highest composite score = best current opportunity.
        """
        run_date = run_date or date.today()

        quality_map: dict[str, float] = {q.symbol: q.quality_score for q in quality_scores}
        routing_map: dict[str, object] = {
            d.symbol: d for d in routing_decisions if d.allowed
        }
        breadth_score = float(getattr(breadth_snapshot, "regime_breadth_score", 0.50)
                              if breadth_snapshot else 0.50)
        breadth_state = str(getattr(breadth_snapshot,   "breadth_state", "NEUTRAL")
                            if breadth_snapshot else "NEUTRAL")

        allowed_results = [
            r for r in stable_results
            if r.is_valid() and r.symbol in routing_map
        ]

        total_valid = sum(1 for r in stable_results if r.is_valid())
        logger.info(
            "RankingEngine [%s]: %d classified → %d allowed (%.0f%% pass rate)  "
            "market=%s",
            universe, total_valid, len(allowed_results),
            len(allowed_results) / max(total_valid, 1) * 100,
            market_regime,
        )

        if not allowed_results:
            logger.warning("RankingEngine [%s]: no allowed opportunities to rank.", universe)
            return []

        composite_scores = score_batch(
            stable_results    = allowed_results,
            quality_map       = quality_map,
            breadth_score     = breadth_score,
            sector_state_map  = symbol_sector_map,
            market_regime     = market_regime,
            weights           = self.weights,
            run_date          = run_date,
            alignment_matrix  = self.alignment_matrix,
            rs_level_weight   = self.rs_level_weight,
            rs_trend_weight   = self.rs_trend_weight,
        )

        ranked = self._build_ranked(composite_scores, routing_map, symbol_sector_name)

        log_ranking_summary(
            ranked=ranked, universe=universe, run_date=run_date,
            market_regime=market_regime, breadth_state=breadth_state,
            top_n=self.top_n, weights=self.weights,
            rs_leaders_n=self.rs_leaders_n,
        )
        for r in ranked:
            if r.opportunity_score >= self.min_score_to_log:
                log_score_breakdown(r)

        return ranked

    # ──────────────────────────────────────────────────────────────
    #  Persistence
    # ──────────────────────────────────────────────────────────────

    def persist(
        self,
        ranked:     list[RankedOpportunity],
        output_dir: str | Path,
        universe:   str,
        append:     bool = True,
    ) -> dict[str, Path]:
        """
        Write two parquet files to output/rankings/:
          opportunity_rankings.parquet  — full ranked list (incl. FIX 2/3 fields:
                                           market_alignment_penalty, rs_contribution)
          opportunity_scores.parquet    — component score breakdown

        Both files support append mode with deduplication on
        (symbol, universe, run_date) so daily runs accumulate history.

        Returns dict with keys "rankings" and "scores".
        """
        if not ranked:
            return {}

        out = Path(output_dir) / "rankings"
        out.mkdir(parents=True, exist_ok=True)
        saved: dict[str, Path] = {}

        # ── opportunity_rankings.parquet ───────────────────────────
        rows = [r.to_dict() for r in ranked]
        for row in rows:
            row["universe"] = universe

        p_rankings = out / "opportunity_rankings.parquet"
        saved["rankings"] = self._write_parquet(
            pd.DataFrame(rows), p_rankings,
            dedup_cols=["symbol", "universe", "run_date"], append=append,
        )
        logger.info("opportunity_rankings [%s]: %d rows → '%s'.", universe, len(rows), p_rankings)

        # ── opportunity_scores.parquet ─────────────────────────────
        score_rows = []
        for r in ranked:
            cs = r.component_scores
            score_rows.append({
                "symbol":                   r.symbol,
                "universe":                 universe,
                "run_date":                 str(r.run_date),
                "rank":                     r.rank,
                "opportunity_score":        r.opportunity_score,
                "strategy":                 r.strategy,
                "sector":                   r.sector,
                "stable_regime":            r.stable_regime,
                "risk_profile":             r.risk_profile,
                "psm":                      r.psm,
                "dominant_factor":          r.dominant_factor,
                "cs_trend":                 cs.trend,
                "cs_momentum":              cs.momentum,
                "cs_quality":               cs.quality,
                "cs_market_align":          cs.market_alignment,
                "cs_breadth":               cs.breadth_alignment,
                "cs_sector":                cs.sector_strength,
                "cs_stability":             cs.stability,
                "cs_rs":                    cs.rs_score,
                "cs_rs_level":              cs.rs_level_score,
                "cs_rs_trend":              cs.rs_trend_score,
                "market_alignment_penalty": r.market_alignment_penalty(),
                "rs_contribution":          r.rs_contribution(),
            })

        p_scores = out / "opportunity_scores.parquet"
        saved["scores"] = self._write_parquet(
            pd.DataFrame(score_rows), p_scores,
            dedup_cols=["symbol", "universe", "run_date"], append=append,
        )
        logger.info("opportunity_scores [%s]: %d rows → '%s'.", universe, len(score_rows), p_scores)

        return saved

    # ──────────────────────────────────────────────────────────────
    #  Private helpers
    # ──────────────────────────────────────────────────────────────

    @staticmethod
    def _build_ranked(
        composite_scores:   list[CompositeOpportunityScore],
        routing_map:        dict,
        symbol_sector_name: dict[str, str],
    ) -> list[RankedOpportunity]:
        ranked: list[RankedOpportunity] = []
        for i, cs in enumerate(composite_scores):
            routing = routing_map.get(cs.symbol)
            if routing is None:
                continue
            ranked.append(RankedOpportunity(
                rank              = i + 1,
                symbol            = cs.symbol,
                market            = cs.market,
                run_date          = cs.run_date,
                opportunity_score = cs.opportunity_score,
                strategy          = routing.strategy,
                sector            = symbol_sector_name.get(cs.symbol, "UNKNOWN"),
                stable_regime     = getattr(routing, "regime_context", "UNKNOWN"),
                risk_profile      = routing.risk_profile,
                psm               = routing.position_size_multiplier,
                component_scores  = cs.component_scores,
                weights_used      = cs.weights_used,
                dominant_factor   = cs.dominant_factor(),
            ))
        return ranked

    @staticmethod
    def _write_parquet(
        df: pd.DataFrame, path: Path, dedup_cols: list[str], append: bool,
    ) -> Path:
        if append and path.exists():
            existing = pd.read_parquet(path)
            for col in df.columns:
                if col not in existing.columns:
                    existing[col] = None
            combined = pd.concat([existing, df], ignore_index=True)
            combined.drop_duplicates(subset=dedup_cols, keep="last", inplace=True)
            combined.to_parquet(path, engine="pyarrow", compression="snappy", index=False)
        else:
            df.to_parquet(path, engine="pyarrow", compression="snappy", index=False)
        return path