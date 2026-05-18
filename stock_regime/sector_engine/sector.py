"""
stock_regime/sector_engine/sector.py
======================================
Aggregates stock classifications by sector to produce sector-level
trend, momentum, breadth, and relative strength metrics.

Sector mapping is loaded from a CSV file at:
  data/sectors/<universe>_sectors.csv

Expected CSV schema:
  symbol, sector, industry

When no CSV is found, the engine falls back to an UNKNOWN sector
for all symbols (graceful degradation — never fails the pipeline).

Output per sector
-----------------
SectorSnapshot:
  sector              : sector name
  stock_count         : number of classified stocks in sector
  trend_strength      : mean trend dimensional score
  momentum_strength   : mean momentum dimensional score
  pct_bullish         : % stocks TREND_UP or MOMENTUM
  pct_bearish         : % stocks TREND_DOWN
  avg_confidence      : mean smoothed_confidence
  regime_distribution : {regime: count}
  sector_state        : LEADING | NEUTRAL | LAGGING
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)

_BULLISH_REGIMES = {"TREND_UP", "MOMENTUM"}
_BEARISH_REGIMES = {"TREND_DOWN"}


@dataclass
class SectorSnapshot:
    sector:             str
    universe:           str
    run_date:           date
    stock_count:        int
    trend_strength:     float
    momentum_strength:  float
    pct_bullish:        float
    pct_bearish:        float
    avg_confidence:     float
    regime_distribution: dict = field(default_factory=dict)
    sector_state:       str = "NEUTRAL"   # LEADING | NEUTRAL | LAGGING

    def to_dict(self) -> dict:
        return {
            "sector":            self.sector,
            "universe":          self.universe,
            "run_date":          str(self.run_date),
            "stock_count":       self.stock_count,
            "trend_strength":    round(self.trend_strength,    4),
            "momentum_strength": round(self.momentum_strength, 4),
            "pct_bullish":       round(self.pct_bullish,       2),
            "pct_bearish":       round(self.pct_bearish,       2),
            "avg_confidence":    round(self.avg_confidence,    4),
            "sector_state":      self.sector_state,
        }


class SectorEngine:
    """
    Computes sector-level intelligence from stock regime classifications.

    Parameters
    ----------
    sector_map_path : str | Path | None
        Path to a CSV with columns: symbol, sector [, industry].
        When None or missing, all symbols map to "UNKNOWN".
    leading_threshold : float
        pct_bullish above this → sector is LEADING.
    lagging_threshold : float
        pct_bearish above this → sector is LAGGING.
    """

    def __init__(
        self,
        sector_map_path:    Optional[str | Path] = None,
        leading_threshold:  float = 60.0,
        lagging_threshold:  float = 40.0,
    ) -> None:
        self.leading_thr  = leading_threshold
        self.lagging_thr  = lagging_threshold
        self._sector_map: dict[str, str] = {}   # symbol → sector
        if sector_map_path:
            self._load_map(Path(sector_map_path))

    @classmethod
    def from_config(cls, config, universe: str, project_root: Path) -> "SectorEngine":
        """Build from config, looking for data/sectors/<universe>_sectors.csv."""
        map_path = project_root / "data" / "sectors" / f"{universe.lower()}_sectors.csv"
        sc = getattr(config, "sector_engine", None)
        return cls(
            sector_map_path   = map_path if map_path.exists() else None,
            leading_threshold = float(getattr(sc, "leading_threshold",  60.0) if sc else 60.0),
            lagging_threshold = float(getattr(sc, "lagging_threshold",  40.0) if sc else 40.0),
        )

    def compute(
        self,
        stable_results: list,       # list[StableRegimeResult]
        universe:       str,
        run_date:       Optional[date] = None,
    ) -> list[SectorSnapshot]:
        """Compute one SectorSnapshot per sector found in stable_results."""
        run_date = run_date or date.today()
        valid    = [r for r in stable_results if r.is_valid()]

        # Group by sector
        groups: dict[str, list] = {}
        for r in valid:
            sector = self._sector_map.get(r.symbol, "UNKNOWN")
            groups.setdefault(sector, []).append(r)

        snapshots = []
        for sector, results in sorted(groups.items()):
            snap = self._compute_sector(sector, universe, run_date, results)
            snapshots.append(snap)

        # Log sector summary
        logger.info("SectorEngine [%s]: %d sectors computed", universe, len(snapshots))
        for s in sorted(snapshots, key=lambda x: -x.trend_strength)[:5]:
            logger.info(
                "  %-20s  state=%-8s  trend=%.2f  mom=%.2f  bull=%.0f%%",
                s.sector, s.sector_state,
                s.trend_strength, s.momentum_strength, s.pct_bullish,
            )

        return snapshots

    def persist(
        self,
        snapshots:  list[SectorSnapshot],
        output_dir: str | Path,
        append:     bool = True,
    ) -> Optional[Path]:
        if not snapshots:
            return None
        out = Path(output_dir) / "sectors"
        out.mkdir(parents=True, exist_ok=True)
        universe = snapshots[0].universe
        path     = out / f"{universe.lower()}_sectors.parquet"

        new_df = pd.DataFrame([s.to_dict() for s in snapshots])
        if append and path.exists():
            existing = pd.read_parquet(path)
            combined = pd.concat([existing, new_df], ignore_index=True)
            combined.drop_duplicates(
                subset=["sector", "universe", "run_date"], keep="last", inplace=True
            )
            combined.to_parquet(path, engine="pyarrow", compression="snappy", index=False)
        else:
            new_df.to_parquet(path, engine="pyarrow", compression="snappy", index=False)

        logger.info("Sectors persisted [%s] → '%s'.", universe, path)
        return path

    def get_sector(self, symbol: str) -> str:
        """Return the sector for a symbol (UNKNOWN if not mapped)."""
        return self._sector_map.get(symbol, "UNKNOWN")

    # ──────────────────────────────────────────────────────────────
    #  Private helpers
    # ──────────────────────────────────────────────────────────────

    def _compute_sector(
        self, sector: str, universe: str, run_date: date, results: list
    ) -> SectorSnapshot:
        n     = max(len(results), 1)
        regimes = [
            r.stable_regime.value if hasattr(r, "stable_regime")
            else r.stock_regime.value
            for r in results
        ]
        n_bull = sum(1 for rg in regimes if rg in _BULLISH_REGIMES)
        n_bear = sum(1 for rg in regimes if rg in _BEARISH_REGIMES)

        pct_bull = n_bull / n * 100
        pct_bear = n_bear / n * 100

        # Mean dimensional scores
        trend_scores = [
            r.dimensional_scores.trend for r in results
            if r.dimensional_scores is not None
        ]
        mom_scores = [
            r.dimensional_scores.momentum for r in results
            if r.dimensional_scores is not None
        ]
        confs = [
            r.smoothed_confidence if hasattr(r, "smoothed_confidence")
            else r.confidence
            for r in results
        ]

        trend_strength    = sum(trend_scores) / len(trend_scores) if trend_scores else 0.0
        momentum_strength = sum(mom_scores)   / len(mom_scores)   if mom_scores   else 0.0
        avg_conf          = sum(confs)         / len(confs)        if confs        else 0.0

        # Regime distribution
        dist: dict[str, int] = {}
        for rg in regimes:
            dist[rg] = dist.get(rg, 0) + 1

        # State
        if pct_bull > self.leading_thr:
            state = "LEADING"
        elif pct_bear > self.lagging_thr:
            state = "LAGGING"
        else:
            state = "NEUTRAL"

        return SectorSnapshot(
            sector             = sector,
            universe           = universe,
            run_date           = run_date,
            stock_count        = len(results),
            trend_strength     = round(trend_strength,    4),
            momentum_strength  = round(momentum_strength, 4),
            pct_bullish        = round(pct_bull,          2),
            pct_bearish        = round(pct_bear,          2),
            avg_confidence     = round(avg_conf,          4),
            regime_distribution= dist,
            sector_state       = state,
        )

    def _load_map(self, path: Path) -> None:
        if not path.exists():
            logger.debug("Sector map not found at '%s' — using UNKNOWN.", path)
            return
        try:
            df = pd.read_csv(path)
            df.columns = [c.lower().strip() for c in df.columns]
            if "symbol" not in df.columns or "sector" not in df.columns:
                logger.warning("Sector CSV missing 'symbol' or 'sector' column.")
                return
            self._sector_map = dict(zip(df["symbol"], df["sector"]))
            logger.info("Loaded %d sector mappings from '%s'.", len(self._sector_map), path)
        except Exception as exc:
            logger.warning("Could not load sector map '%s': %s", path, exc)