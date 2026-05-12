"""
stock_regime/src/persistence.py
================================
Saves all engine outputs to partitioned parquet files under the
``output/`` directory.

Directory layout
----------------
output/
  indicators/
    YYYY-MM-DD/
      indicators.parquet
  signals/
    YYYY-MM-DD/
      signals.parquet
  classifications/
    YYYY-MM-DD/
      classifications.parquet
  rankings/
    YYYY-MM-DD/
      trend_ranking.parquet
      momentum_ranking.parquet
      volatility_ranking.parquet

Parquet was chosen because it:
  • preserves dtypes (no float → string surprises)
  • is ~5× smaller than CSV for columnar time-series data
  • reads 10–50× faster than CSV for large symbol universes
  • is natively supported by the rest of the trading platform

Design constraints
------------------
• Each ``save_*`` method is independent — callers can persist subsets.
• Files are partitioned by run date so historical runs are never overwritten.
• All methods silently succeed on empty input (no-op with a debug log).
"""

from __future__ import annotations

import logging
from datetime import date, datetime
from pathlib import Path
from typing import Optional

import pandas as pd

from .models import StockRegimeResult
from .ranker import RankingOutput, RankedStock

logger = logging.getLogger(__name__)


class OutputPersistence:
    """
    Writes Stock Regime Engine outputs to partitioned parquet files.

    Parameters
    ----------
    output_dir :
        Root directory for all outputs.  Created if it does not exist.
    """

    def __init__(self, output_dir: str | Path = "output") -> None:
        self._root = Path(output_dir)
        self._root.mkdir(parents=True, exist_ok=True)
        logger.debug("OutputPersistence root: '%s'.", self._root.resolve())

    # ──────────────────────────────────────────────────────────────
    #  Public save methods
    # ──────────────────────────────────────────────────────────────

    def save_indicators(
        self,
        results: list[StockRegimeResult],
        run_date: Optional[date] = None,
    ) -> Optional[Path]:
        """
        Persist indicator snapshots for all stocks.

        Schema
        ------
        symbol | run_date | close | ema20 | ema50 | ema200 |
        ema20_slope | ema50_slope | adx | atr | atr_ma |
        volume | volume_ma | relative_strength | high_52w

        Parameters
        ----------
        results :
            All StockRegimeResult objects from one engine run.
        run_date :
            Date label for partitioning.  Defaults to today.

        Returns
        -------
        Path or None
            Path to the saved file, or None if nothing to save.
        """
        if not results:
            logger.debug("save_indicators: nothing to save.")
            return None

        run_date = run_date or date.today()
        rows = []

        for r in results:
            snap = r.indicators
            row = {"symbol": r.symbol, "run_date": run_date}
            row.update(snap.to_dict())
            rows.append(row)

        path = self._make_path("indicators", run_date, "indicators.parquet")
        return self._write(pd.DataFrame(rows), path)

    def save_signals(
        self,
        results: list[StockRegimeResult],
        run_date: Optional[date] = None,
    ) -> Optional[Path]:
        """
        Persist boolean signals for all stocks.

        Schema
        ------
        symbol | run_date | price_above_ema200 | price_below_ema200 |
        ema20_above_ema50 | ema20_below_ema50 | ema20_flat | ema50_flat |
        adx_strong | adx_weak | atr_high | atr_low | atr_compressed |
        atr_expanding | volume_confirmed | rs_positive | rs_negative |
        rs_strong | price_near_52w_high
        """
        if not results:
            logger.debug("save_signals: nothing to save.")
            return None

        run_date = run_date or date.today()
        rows = []

        for r in results:
            row = {"symbol": r.symbol, "run_date": run_date}
            row.update(r.signals.to_dict())
            rows.append(row)

        path = self._make_path("signals", run_date, "signals.parquet")
        return self._write(pd.DataFrame(rows), path)

    def save_classifications(
        self,
        results: list[StockRegimeResult],
        run_date: Optional[date] = None,
    ) -> Optional[Path]:
        """
        Persist classification results for all stocks.

        Schema
        ------
        symbol | market | stock_regime | confidence |
        trend_score | momentum_score | volatility_score |
        regime_score_TREND_UP | regime_score_TREND_DOWN | ... |
        is_valid | error | run_date
        """
        if not results:
            logger.debug("save_classifications: nothing to save.")
            return None

        run_date = run_date or date.today()
        rows = []

        for r in results:
            ds = r.dimensional_scores
            row: dict = {
                "symbol":           r.symbol,
                "market":           r.market,
                "stock_regime":     r.stock_regime.value,
                "confidence":       r.confidence,
                "trend_score":      ds.trend,
                "momentum_score":   ds.momentum,
                "volatility_score": ds.volatility,
                "is_valid":         r.is_valid(),
                "error":            r.error or "",
                "run_date":         run_date,
            }
            # Flatten individual regime scores for queryability
            for regime_name, score in r.regime_scores.items():
                row[f"rscore_{regime_name.lower()}"] = score
            rows.append(row)

        path = self._make_path("classifications", run_date, "classifications.parquet")
        return self._write(pd.DataFrame(rows), path)

    def save_rankings(
        self,
        rankings: RankingOutput,
        run_date: Optional[date] = None,
    ) -> dict[str, Optional[Path]]:
        """
        Persist all three ranking lists to separate parquet files.

        Schema (each file)
        ------------------
        rank | symbol | market | stock_regime | confidence | score |
        ranking_type | run_date

        Returns
        -------
        dict[str, Optional[Path]]
            Keys: "trend", "momentum", "volatility".
        """
        run_date = run_date or date.today()
        saved: dict[str, Optional[Path]] = {}

        specs = [
            ("trend",      rankings.strongest_trends),
            ("momentum",   rankings.strongest_momentum),
            ("volatility", rankings.highest_volatility),
        ]

        for key, entries in specs:
            if not entries:
                saved[key] = None
                logger.debug("save_rankings: %s ranking is empty.", key)
                continue

            rows = []
            for entry in entries:
                row = entry.to_dict()
                row["run_date"] = run_date
                rows.append(row)

            path = self._make_path("rankings", run_date, f"{key}_ranking.parquet")
            saved[key] = self._write(pd.DataFrame(rows), path)

        return saved

    # ──────────────────────────────────────────────────────────────
    #  Private helpers
    # ──────────────────────────────────────────────────────────────

    def _make_path(self, category: str, run_date: date, filename: str) -> Path:
        """Build a partitioned output path."""
        date_str = run_date.strftime("%Y-%m-%d")
        path = self._root / category / date_str / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    @staticmethod
    def _write(df: pd.DataFrame, path: Path) -> Optional[Path]:
        """Write DataFrame to parquet; log and return the path."""
        if df.empty:
            logger.debug("Skipping empty DataFrame write to '%s'.", path)
            return None
        try:
            df.to_parquet(path, engine="pyarrow", compression="snappy", index=False)
            logger.info("Saved %d rows → '%s'.", len(df), path)
            return path
        except Exception as exc:  # noqa: BLE001
            logger.error("Failed to write '%s': %s", path, exc)
            return None
