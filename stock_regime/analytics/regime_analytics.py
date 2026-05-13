"""
stock_regime/analytics/regime_analytics.py
===========================================
Computes historical regime statistics from the persisted
regime_history.parquet file produced by RegimeStabiliser.

Analytics produced
------------------
1. Regime duration stats   — how long each regime typically lasts
2. Transition matrix       — what regime most commonly follows each regime
3. Current episodes        — stocks currently in each regime with age
4. Recent transitions      — regime changes in the last N days

These analytics require no ML — they are simple counting operations
over the historical regime record.  They are the foundation for
building a strategy router and for validating classification quality.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date, timedelta
from pathlib import Path
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)

# All possible stable regime values (matches StockRegime enum)
_ALL_REGIMES = [
    "TREND_UP", "TREND_DOWN", "RANGE", "MOMENTUM",
    "BREAKOUT_SETUP", "VOLATILE", "QUIET", "UNCERTAIN",
]


# ─────────────────────────────────────────────────────────────────────────────
#  Output dataclasses
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class RegimeDurationStats:
    """Duration statistics for one regime across all stocks and episodes."""
    regime:        str
    universe:      str
    mean_bars:     float
    median_bars:   float
    p25_bars:      float
    p75_bars:      float
    p95_bars:      float
    min_bars:      int
    max_bars:      int
    sample_count:  int    # number of completed episodes

    def to_dict(self) -> dict:
        return {
            "regime":       self.regime,
            "universe":     self.universe,
            "mean_bars":    round(self.mean_bars,   2),
            "median_bars":  round(self.median_bars, 2),
            "p25_bars":     round(self.p25_bars,    2),
            "p75_bars":     round(self.p75_bars,    2),
            "p95_bars":     round(self.p95_bars,    2),
            "min_bars":     self.min_bars,
            "max_bars":     self.max_bars,
            "sample_count": self.sample_count,
        }


@dataclass
class TransitionMatrix:
    """
    Probability matrix: P(next_regime | current_regime).

    rows = current stable regime
    cols = next stable regime (after a transition)
    """
    universe:    str
    matrix:      pd.DataFrame    # index=from_regime, columns=to_regime
    counts:      pd.DataFrame    # raw counts (same shape)
    total_transitions: int

    def to_dict(self) -> dict:
        return {
            "universe":          self.universe,
            "total_transitions": self.total_transitions,
            "matrix":            self.matrix.to_dict(),
        }


@dataclass
class CurrentEpisode:
    """A stock currently in a stable regime, with age info."""
    symbol:        str
    universe:      str
    stable_regime: str
    regime_age:    int      # bars in current stable regime
    confidence:    float
    last_date:     date

    def to_dict(self) -> dict:
        return {
            "symbol":        self.symbol,
            "universe":      self.universe,
            "stable_regime": self.stable_regime,
            "regime_age":    self.regime_age,
            "confidence":    round(self.confidence, 4),
            "last_date":     str(self.last_date),
        }


@dataclass
class RegimeAnalyticsReport:
    """Full analytics output for one run."""
    universe:          str
    computed_at:       date
    duration_stats:    list[RegimeDurationStats] = field(default_factory=list)
    transition_matrix: Optional[TransitionMatrix] = None
    current_episodes:  list[CurrentEpisode]       = field(default_factory=list)
    recent_changes:    list[CurrentEpisode]        = field(default_factory=list)


# ─────────────────────────────────────────────────────────────────────────────
#  Analytics engine
# ─────────────────────────────────────────────────────────────────────────────

class RegimeAnalytics:
    """
    Computes regime analytics from the regime_history.parquet file.

    Parameters
    ----------
    history_path :
        Path to regime_history.parquet (written by RegimeStabiliser).
    min_episode_bars :
        Minimum bars for an episode to count in duration stats.
        Filters out noise from very short visits to a regime.
    recent_change_days :
        How many calendar days back to look for "recent" changes.
    """

    def __init__(
        self,
        history_path:      str | Path,
        min_episode_bars:  int = 2,
        recent_change_days:int = 5,
    ) -> None:
        self._path              = Path(history_path)
        self.min_episode_bars   = min_episode_bars
        self.recent_change_days = recent_change_days

    # ──────────────────────────────────────────────────────────────────────────
    #  Public API
    # ──────────────────────────────────────────────────────────────────────────

    def compute(
        self,
        universe:   str,
        as_of_date: Optional[date] = None,
    ) -> RegimeAnalyticsReport:
        """
        Compute all analytics for *universe* using data up to *as_of_date*.

        Parameters
        ----------
        universe :
            Universe label to filter (e.g. "NIFTY500").
        as_of_date :
            Only use history rows where run_date <= as_of_date.
            Defaults to today.  Used for backtesting reproducibility.
        """
        as_of = as_of_date or date.today()
        df    = self._load(universe, as_of)

        if df.empty:
            logger.warning("No history data for universe='%s' as_of=%s.", universe, as_of)
            return RegimeAnalyticsReport(universe=universe, computed_at=as_of)

        episodes  = self._build_episodes(df)
        report    = RegimeAnalyticsReport(universe=universe, computed_at=as_of)

        report.duration_stats    = self._duration_stats(episodes, universe)
        report.transition_matrix = self._transition_matrix(df, universe)
        report.current_episodes  = self._current_episodes(df, universe)
        report.recent_changes    = self._recent_changes(df, universe, as_of)

        logger.info(
            "RegimeAnalytics [%s]: %d episodes | %d duration stats | "
            "%d current | %d recent changes",
            universe,
            len(episodes),
            len(report.duration_stats),
            len(report.current_episodes),
            len(report.recent_changes),
        )
        return report

    def persist(self, report: RegimeAnalyticsReport, output_dir: str | Path) -> dict[str, Path]:
        """
        Save all analytics to parquet files.

        Returns a dict of {category: path} for each file written.
        """
        out_dir = Path(output_dir) / "analytics" / str(report.computed_at)
        out_dir.mkdir(parents=True, exist_ok=True)
        saved: dict[str, Path] = {}

        if report.duration_stats:
            p = out_dir / "regime_duration_stats.parquet"
            pd.DataFrame([s.to_dict() for s in report.duration_stats]).to_parquet(
                p, engine="pyarrow", compression="snappy", index=False,
            )
            saved["duration_stats"] = p

        if report.transition_matrix is not None:
            p = out_dir / "transition_matrix.parquet"
            report.transition_matrix.matrix.to_parquet(
                p, engine="pyarrow", compression="snappy",
            )
            saved["transition_matrix"] = p

        if report.current_episodes:
            p = out_dir / "current_episodes.parquet"
            pd.DataFrame([e.to_dict() for e in report.current_episodes]).to_parquet(
                p, engine="pyarrow", compression="snappy", index=False,
            )
            saved["current_episodes"] = p

        if report.recent_changes:
            p = out_dir / "recent_transitions.parquet"
            pd.DataFrame([e.to_dict() for e in report.recent_changes]).to_parquet(
                p, engine="pyarrow", compression="snappy", index=False,
            )
            saved["recent_transitions"] = p

        logger.info("Analytics persisted to '%s': %s", out_dir, list(saved.keys()))
        return saved

    # ──────────────────────────────────────────────────────────────────────────
    #  Private computation helpers
    # ──────────────────────────────────────────────────────────────────────────

    def _load(self, universe: str, as_of: date) -> pd.DataFrame:
        if not self._path.exists():
            return pd.DataFrame()
        df = pd.read_parquet(self._path)
        df["run_date"] = pd.to_datetime(df["run_date"]).dt.date
        df = df[
            (df["market"] == universe) &
            (df["run_date"] <= as_of)
        ].copy()
        df.sort_values(["symbol", "run_date"], inplace=True)
        return df

    def _build_episodes(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Convert bar-level regime history into completed regime episodes.

        An episode is one continuous run of the same stable_regime for
        one symbol.  Only completed episodes (ended before the last bar)
        are included.
        """
        records = []
        for symbol, sym_df in df.groupby("symbol"):
            sym_df  = sym_df.sort_values("run_date").reset_index(drop=True)
            prev_regime = None
            ep_start    = None
            ep_bars     = 0

            for _, row in sym_df.iterrows():
                regime = row["stable_regime"]
                if regime != prev_regime:
                    if prev_regime is not None and ep_bars >= self.min_episode_bars:
                        records.append({
                            "symbol":       symbol,
                            "regime":       prev_regime,
                            "start_date":   ep_start,
                            "end_date":     row["run_date"],
                            "duration_bars":ep_bars,
                        })
                    ep_start    = row["run_date"]
                    ep_bars     = 1
                    prev_regime = regime
                else:
                    ep_bars += 1

        return pd.DataFrame(records)

    def _duration_stats(
        self, episodes: pd.DataFrame, universe: str
    ) -> list[RegimeDurationStats]:
        if episodes.empty:
            return []
        stats: list[RegimeDurationStats] = []
        for regime in _ALL_REGIMES:
            grp = episodes[episodes["regime"] == regime]["duration_bars"]
            if len(grp) < 3:
                continue
            stats.append(RegimeDurationStats(
                regime       = regime,
                universe     = universe,
                mean_bars    = float(grp.mean()),
                median_bars  = float(grp.median()),
                p25_bars     = float(grp.quantile(0.25)),
                p75_bars     = float(grp.quantile(0.75)),
                p95_bars     = float(grp.quantile(0.95)),
                min_bars     = int(grp.min()),
                max_bars     = int(grp.max()),
                sample_count = int(len(grp)),
            ))
        return stats

    def _transition_matrix(
        self, df: pd.DataFrame, universe: str
    ) -> Optional[TransitionMatrix]:
        """
        Build a probability matrix of regime transitions.

        For each symbol, find consecutive pairs of stable_regime where
        the regime changed (regime_changed_today == True).
        """
        transitions = []
        for _, sym_df in df.groupby("symbol"):
            sym_df    = sym_df.sort_values("run_date")
            # Only rows where a regime change happened
            changes   = sym_df[sym_df["regime_changed_today"] == True]
            if len(changes) < 2:
                continue
            regimes   = changes["stable_regime"].tolist()
            for from_r, to_r in zip(regimes[:-1], regimes[1:]):
                transitions.append({"from": from_r, "to": to_r})

        if not transitions:
            return None

        trans_df = pd.DataFrame(transitions)
        counts   = (
            trans_df.groupby(["from", "to"])
            .size()
            .unstack(fill_value=0)
            .reindex(index=_ALL_REGIMES, columns=_ALL_REGIMES, fill_value=0)
        )
        # Normalise each row to get probabilities
        row_sums = counts.sum(axis=1).replace(0, 1)
        probs    = counts.div(row_sums, axis=0).round(4)

        return TransitionMatrix(
            universe          = universe,
            matrix            = probs,
            counts            = counts,
            total_transitions = len(transitions),
        )

    def _current_episodes(
        self, df: pd.DataFrame, universe: str
    ) -> list[CurrentEpisode]:
        """Return each symbol's current stable regime and age (from latest bar)."""
        latest = df.groupby("symbol").last().reset_index()
        episodes = []
        for _, row in latest.iterrows():
            episodes.append(CurrentEpisode(
                symbol        = row["symbol"],
                universe      = universe,
                stable_regime = row["stable_regime"],
                regime_age    = int(row.get("stable_regime_age", 0)),
                confidence    = float(row.get("confidence", 0.0)),
                last_date     = row["run_date"],
            ))
        # Sort by regime age descending — longest-running first
        episodes.sort(key=lambda e: e.regime_age, reverse=True)
        return episodes

    def _recent_changes(
        self, df: pd.DataFrame, universe: str, as_of: date
    ) -> list[CurrentEpisode]:
        """Return symbols that changed stable_regime in the last recent_change_days."""
        cutoff   = as_of - timedelta(days=self.recent_change_days)
        changed  = df[
            (df["regime_changed_today"] == True) &
            (df["run_date"] >= cutoff)
        ].copy()
        if changed.empty:
            return []
        # Keep the most recent change per symbol
        changed = changed.sort_values("run_date").groupby("symbol").last().reset_index()
        episodes = []
        for _, row in changed.iterrows():
            episodes.append(CurrentEpisode(
                symbol        = row["symbol"],
                universe      = universe,
                stable_regime = row["stable_regime"],
                regime_age    = int(row.get("stable_regime_age", 0)),
                confidence    = float(row.get("confidence", 0.0)),
                last_date     = row["run_date"],
            ))
        episodes.sort(key=lambda e: e.last_date, reverse=True)
        return episodes