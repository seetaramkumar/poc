"""
stock_regime/sector_engine/sector.py
======================================
SectorEngine — orchestrates the full sector intelligence pipeline.

FIX 1 — Minimum Sector Sample Size Validation
-----------------------------------------------
persist() now writes sector_rankings.parquet (the LEADERBOARD file) using
ONLY rankable sectors (excluded_from_ranking == False). sector_metrics.parquet
and sector_states.parquet still record EVERY sector — including
INSUFFICIENT_DATA ones — for audit/debugging purposes, but they carry the
excluded_from_ranking flag and INSUFFICIENT_DATA state explicitly so no
downstream consumer can mistake them for a real ranking signal.

Pipeline per run
----------------
1. SectorMap.get_sector()           → group stocks by sector
2. metrics.compute_sector_metrics() → compute raw metrics per sector
3. metrics.compute_composite_score()→ single [0,1] ranking score
4. SectorClassifier.classify_batch()→ assign SectorState + ranks
   (sectors below min_sector_stocks become INSUFFICIENT_DATA, rank=0)
5. diagnostics.log_sector_summary() → structured logging
6. persist()                        → three parquet files

Public interface (unchanged)
-----------------------------
SectorEngine.compute()    → list[SectorSnapshot]
SectorEngine.persist()    → dict[str, Path]
SectorEngine.get_sector() → str (for strategy router / pipeline)
SectorEngine.get_sector_context_map() → dict
SectorEngine.get_symbol_sector_state() → str
"""

from __future__ import annotations

import logging
from datetime import date
from pathlib import Path
from typing import Optional

import pandas as pd

from .classifier  import SectorClassifier
from .diagnostics import build_sector_context_map, log_sector_summary
from .metrics     import compute_composite_score, compute_sector_metrics
from .models      import SectorSnapshot, SectorState
from .sector_map  import SectorMap

logger = logging.getLogger(__name__)


class SectorEngine:
    """
    Full sector intelligence engine.

    Parameters
    ----------
    sector_map :
        Pre-built SectorMap instance.
    classifier :
        Pre-built SectorClassifier instance (carries min_sector_stocks).
    top_n_log :
        How many top sectors to log.
    bottom_n_log :
        How many bottom sectors to log.
    """

    def __init__(
        self,
        sector_map:   SectorMap,
        classifier:   SectorClassifier,
        top_n_log:    int = 5,
        bottom_n_log: int = 5,
    ) -> None:
        self._map        = sector_map
        self._classifier = classifier
        self.top_n_log   = top_n_log
        self.bottom_n_log= bottom_n_log
        self._last_snapshots: list[SectorSnapshot] = []

        cov = sector_map.coverage()
        logger.info(
            "SectorEngine ready: %d symbols mapped → %d sectors. "
            "min_sector_stocks=%d.",
            cov["total_mapped"], cov["unique_sectors"],
            classifier.min_sector_stocks,
        )

    # ──────────────────────────────────────────────────────────────
    #  Factory
    # ──────────────────────────────────────────────────────────────

    @classmethod
    def from_config(
        cls,
        config,
        universe:     str,
        project_root: Path,
    ) -> "SectorEngine":
        """Build a SectorEngine from config + CSV at data/sectors/<universe>_sectors.csv."""
        sector_map  = SectorMap.from_project_root(project_root, universe)
        classifier  = SectorClassifier.from_config(config)
        sc          = getattr(config, "sector_engine", None)
        return cls(
            sector_map   = sector_map,
            classifier   = classifier,
            top_n_log    = int(getattr(sc, "top_n_log",    5) if sc else 5),
            bottom_n_log = int(getattr(sc, "bottom_n_log", 5) if sc else 5),
        )

    # ──────────────────────────────────────────────────────────────
    #  Primary API
    # ──────────────────────────────────────────────────────────────

    def compute(
        self,
        stable_results: list,               # list[StableRegimeResult]
        quality_scores: list,               # list[QualityScore]
        universe:       str,
        run_date:       Optional[date] = None,
    ) -> list[SectorSnapshot]:
        """
        Compute full sector intelligence from today's classified stocks.

        Returns
        -------
        list[SectorSnapshot]
            Rankable sectors first (rank 1..N, sorted by composite_score),
            followed by INSUFFICIENT_DATA sectors (rank=0, alphabetical).
        """
        run_date = run_date or date.today()
        valid    = [r for r in stable_results if r.is_valid()]

        quality_map: dict[str, float] = {
            q.symbol: q.quality_score for q in quality_scores
        }

        sector_groups: dict[str, list] = {}
        unmapped_count = 0
        for r in valid:
            sector = self._map.get_sector(r.symbol)
            if sector == "UNKNOWN":
                unmapped_count += 1
            sector_groups.setdefault(sector, []).append(r)

        if unmapped_count > 0:
            logger.debug(
                "%d stocks mapped to UNKNOWN sector in '%s'. "
                "Add them to data/sectors/%s_sectors.csv to improve coverage.",
                unmapped_count, universe, universe.lower(),
            )

        metrics_list = []
        composite_map: dict[str, float] = {}
        for sector, members in sector_groups.items():
            m = compute_sector_metrics(
                sector         = sector,
                universe       = universe,
                run_date       = run_date,
                stable_results = members,
                quality_map    = quality_map,
            )
            if m is not None:
                score = compute_composite_score(m)
                metrics_list.append(m)
                composite_map[sector] = score

        # classify_batch() applies the min_sector_stocks gate and assigns
        # ranks ONLY among sectors that pass it.
        snapshots = self._classifier.classify_batch(metrics_list, composite_map)

        log_sector_summary(
            snapshots  = snapshots,
            universe   = universe,
            run_date   = run_date,
            top_n      = self.top_n_log,
            bottom_n   = self.bottom_n_log,
        )

        self._last_snapshots = snapshots
        return snapshots

    def get_sector(self, symbol: str) -> str:
        """Return sector for a symbol (UNKNOWN if not mapped)."""
        return self._map.get_sector(symbol)

    def get_sector_context_map(self) -> dict[str, dict]:
        """
        Return a router/ranking-ready dict of sector context.
        Only includes sectors with is_actionable() == True, i.e. sectors
        that are NOT INSUFFICIENT_DATA and NOT excluded_from_ranking.
        """
        return build_sector_context_map(self._last_snapshots)

    def get_symbol_sector_state(self, symbol: str) -> str:
        """
        Return the sector state for a given symbol.

        Symbols belonging to an INSUFFICIENT_DATA sector are not present
        in the context map (it was filtered by is_actionable()), so this
        naturally falls back to NEUTRAL — the sector neither boosts nor
        penalises routing/ranking decisions when its sample size is too
        small to trust.
        """
        sector = self._map.get_sector(symbol)
        ctx    = self.get_sector_context_map()
        return ctx.get(sector, {}).get("state", SectorState.NEUTRAL.value)

    def get_excluded_sectors(self) -> list[SectorSnapshot]:
        """
        Return all sectors currently excluded from ranking due to
        insufficient sample size. Useful for dashboards / alerts that
        want to surface "N sectors have too few stocks mapped".
        """
        return [s for s in self._last_snapshots if s.excluded_from_ranking]

    # ──────────────────────────────────────────────────────────────
    #  Persistence
    # ──────────────────────────────────────────────────────────────

    def persist(
        self,
        snapshots:  list[SectorSnapshot],
        output_dir: str | Path,
        append:     bool = True,
    ) -> dict[str, Path]:
        """
        Persist three parquet files to output/sectors/:

        sector_metrics.parquet
            EVERY sector, including INSUFFICIENT_DATA ones — full audit trail.
        sector_states.parquet
            EVERY sector's state + rank + composite_score + excluded flag.
        sector_rankings.parquet
            LEADERBOARD — ONLY rankable sectors (excluded_from_ranking=False).
            This is the file a dashboard or "top sectors" view should read.

        Returns
        -------
        dict[str, Path]
            Keys: "metrics", "states", "rankings"
        """
        if not snapshots:
            return {}

        out = Path(output_dir) / "sectors"
        out.mkdir(parents=True, exist_ok=True)
        saved: dict[str, Path] = {}

        universe = snapshots[0].universe

        # ── sector_metrics.parquet — full audit trail, all sectors ──
        metrics_rows = [s.metrics.to_dict() for s in snapshots]
        saved["metrics"] = self._write_parquet(
            pd.DataFrame(metrics_rows),
            out / "sector_metrics.parquet",
            dedup_cols=["sector", "universe", "run_date"],
            append=append,
        )

        # ── sector_states.parquet — full audit trail, all sectors ───
        state_rows = [
            {
                "sector":                s.sector,
                "universe":              s.universe,
                "run_date":              str(s.run_date),
                "state":                 s.state.value,
                "rank":                  s.rank,
                "composite_score":       s.composite_score,
                "stock_count":           s.stock_count,
                "excluded_from_ranking": s.excluded_from_ranking,
            }
            for s in snapshots
        ]
        saved["states"] = self._write_parquet(
            pd.DataFrame(state_rows),
            out / "sector_states.parquet",
            dedup_cols=["sector", "universe", "run_date"],
            append=append,
        )

        # ── sector_rankings.parquet — LEADERBOARD: rankable only ────
        rankable = [s for s in snapshots if not s.excluded_from_ranking]
        ranking_rows = [
            {
                "sector":                  s.sector,
                "universe":                s.universe,
                "run_date":                str(s.run_date),
                "rank":                    s.rank,
                "state":                   s.state.value,
                "composite_score":         s.composite_score,
                "pct_bullish":             s.metrics.pct_bullish,
                "pct_above_ema200":        s.metrics.pct_above_ema200,
                "avg_trend_score":         s.metrics.avg_trend_score,
                "avg_momentum_score":      s.metrics.avg_momentum_score,
                "avg_quality_score":       s.metrics.avg_quality_score,
                "avg_rs_3m":               s.metrics.avg_rs_3m,
                "momentum_participation":  s.metrics.momentum_participation,
                "stock_count":             s.stock_count,
            }
            for s in sorted(rankable, key=lambda x: x.rank)
        ]
        saved["rankings"] = self._write_parquet(
            pd.DataFrame(ranking_rows),
            out / "sector_rankings.parquet",
            dedup_cols=["sector", "universe", "run_date"],
            append=append,
        )

        excluded_count = len(snapshots) - len(rankable)
        logger.info(
            "Sector Intelligence [%s]: %d sectors persisted "
            "(%d rankable in leaderboard, %d excluded as INSUFFICIENT_DATA) → '%s'.",
            universe, len(snapshots), len(rankable), excluded_count, out,
        )
        return saved

    # ──────────────────────────────────────────────────────────────
    #  Private helpers
    # ──────────────────────────────────────────────────────────────

    @staticmethod
    def _write_parquet(
        new_df:    pd.DataFrame,
        path:      Path,
        dedup_cols: list[str],
        append:    bool,
    ) -> Path:
        if append and path.exists():
            existing = pd.read_parquet(path)
            for col in new_df.columns:
                if col not in existing.columns:
                    existing[col] = None
            combined = pd.concat([existing, new_df], ignore_index=True)
            combined.drop_duplicates(subset=dedup_cols, keep="last", inplace=True)
            combined.to_parquet(path, engine="pyarrow", compression="snappy", index=False)
        else:
            new_df.to_parquet(path, engine="pyarrow", compression="snappy", index=False)
        return path