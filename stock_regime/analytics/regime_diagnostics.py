"""
stock_regime/analytics/regime_diagnostics.py
==============================================
Oscillation and regime stability diagnostics.

Analyzes:
1. Oscillation patterns (back-and-forth regime changes)
2. Regime transition matrices (which regimes fight each other)
3. Regime stability statistics (duration distribution)
4. Root cause analysis of oscillation (which transitions dominate)

Produces 4 analytics outputs:
- oscillation_analysis.parquet        (per-symbol oscillation metrics)
- top_oscillators.parquet             (top 50 most oscillating stocks)
- regime_transition_matrix.parquet    (all regime transitions with counts)
- regime_stability.parquet            (stability stats per regime)
- oscillation_summary.parquet         (aggregate oscillation report)

NO changes to existing regime logic — purely diagnostic.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date, timedelta
from pathlib import Path
from typing import Optional

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

# All possible stable regime values
_ALL_REGIMES = [
    "TREND_UP", "TREND_DOWN", "RANGE", "MOMENTUM",
    "BREAKOUT_SETUP", "VOLATILE", "QUIET", "UNCERTAIN",
]


# ─────────────────────────────────────────────────────────────────────────────
#  Output Dataclasses
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class OscillationMetrics:
    """Per-symbol oscillation analysis."""
    symbol:                       str
    current_regime:               str
    stable_regime:                str
    prior_regime:                 str
    oscillation_detected:         bool
    regime_changes_30d:           int
    unique_regimes_seen_30d:      int
    oscillation_count_30d:        int
    avg_confidence_30d:           float
    avg_smoothed_confidence_30d:  float
    current_quality_score:        Optional[float] = None
    sector:                       Optional[str]    = None
    universe:                     Optional[str]    = None

    def to_dict(self) -> dict:
        return {
            "symbol":                        self.symbol,
            "current_regime":                self.current_regime,
            "stable_regime":                 self.stable_regime,
            "prior_regime":                  self.prior_regime,
            "oscillation_detected":          self.oscillation_detected,
            "regime_changes_30d":            self.regime_changes_30d,
            "unique_regimes_seen_30d":       self.unique_regimes_seen_30d,
            "oscillation_count_30d":         self.oscillation_count_30d,
            "avg_confidence_30d":            round(self.avg_confidence_30d, 4),
            "avg_smoothed_confidence_30d":   round(self.avg_smoothed_confidence_30d, 4),
            "current_quality_score":         round(self.current_quality_score, 4) if self.current_quality_score else None,
            "sector":                        self.sector,
            "universe":                      self.universe,
        }


@dataclass
class RegimeTransition:
    """Single regime transition with count."""
    from_regime:      str
    to_regime:        str
    transition_count: int
    transition_pct:   float = 0.0

    def to_dict(self) -> dict:
        return {
            "from_regime":      self.from_regime,
            "to_regime":        self.to_regime,
            "transition_count": self.transition_count,
            "transition_pct":   round(self.transition_pct, 4),
        }


@dataclass
class RegimeStabilityStats:
    """Stability statistics for one regime."""
    regime:              str
    avg_duration_bars:   float
    median_duration_bars: float
    p25_duration_bars:   float
    p75_duration_bars:   float
    max_duration_bars:   int
    min_duration_bars:   int
    episode_count:       int
    universe:            Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "regime":               self.regime,
            "avg_duration_bars":    round(self.avg_duration_bars, 2),
            "median_duration_bars": round(self.median_duration_bars, 2),
            "p25_duration_bars":    round(self.p25_duration_bars, 2),
            "p75_duration_bars":    round(self.p75_duration_bars, 2),
            "max_duration_bars":    self.max_duration_bars,
            "min_duration_bars":    self.min_duration_bars,
            "episode_count":        self.episode_count,
            "universe":             self.universe,
        }


@dataclass
class OscillationSummary:
    """Aggregate oscillation diagnostics."""
    universe:                   str
    total_symbols:              int
    oscillating_symbols:        int
    oscillation_pct:            float
    top_transition_pair:        str
    top_transition_pair_count:  int
    avg_regime_changes:         float
    avg_unique_regimes:         float
    avg_oscillation_count:      float
    most_stable_regime:         str
    most_stable_duration:       float
    most_unstable_regime:       str
    most_unstable_duration:     float

    def to_dict(self) -> dict:
        return {
            "universe":                    self.universe,
            "total_symbols":               self.total_symbols,
            "oscillating_symbols":         self.oscillating_symbols,
            "oscillation_pct":             round(self.oscillation_pct, 2),
            "top_transition_pair":         self.top_transition_pair,
            "top_transition_pair_count":   self.top_transition_pair_count,
            "avg_regime_changes":          round(self.avg_regime_changes, 2),
            "avg_unique_regimes":          round(self.avg_unique_regimes, 2),
            "avg_oscillation_count":       round(self.avg_oscillation_count, 2),
            "most_stable_regime":          self.most_stable_regime,
            "most_stable_duration":        round(self.most_stable_duration, 2),
            "most_unstable_regime":        self.most_unstable_regime,
            "most_unstable_duration":      round(self.most_unstable_duration, 2),
        }


# ─────────────────────────────────────────────────────────────────────────────
#  Regime Diagnostics Engine
# ─────────────────────────────────────────────────────────────────────────────

class RegimeDiagnosticsEngine:
    """
    Analyze regime oscillation and stability from a reconstructed
    historical regime series.

    All analysis is read-only and diagnostic — no changes to regime logic.
    """

    def __init__(self, history_path: Optional[str] = None) -> None:
        """
        Parameters
        ----------
        history_path : str, optional
            Path to regime_history.parquet. If None, will be located
            automatically based on output_dir.
        """
        self._history_path = history_path
        self._history_df:  Optional[pd.DataFrame] = None
        self._quality_map: dict[str, float] = {}

    def load_history(self, path: str) -> None:
        """Load reconstructed historical regime series from parquet."""
        if not Path(path).exists():
            logger.warning("History file not found: %s", path)
            self._history_df = None
            return

        try:
            df = pd.read_parquet(path)
            if df.empty:
                self._history_df = df.copy()
                return

            df = df.copy()
            if 'date' in df.columns:
                df['date'] = pd.to_datetime(df['date'])
            elif df.index.name == 'date':
                df = df.reset_index()
                df['date'] = pd.to_datetime(df['date'])
            else:
                df = df.reset_index()
                if 'date' in df.columns:
                    df['date'] = pd.to_datetime(df['date'])

            if 'regime' in df.columns:
                df['effective_regime'] = df['regime']
            elif 'stable_regime' in df.columns:
                df['effective_regime'] = df['stable_regime']
            elif 'raw_regime' in df.columns:
                df['effective_regime'] = df['raw_regime']
            else:
                df['effective_regime'] = None

            if 'confidence' not in df.columns:
                df['confidence'] = 0.0

            df = df.sort_values(['symbol', 'date'])
            self._history_df = df
            logger.info("Loaded historical regime series: %d rows from %s", len(self._history_df), path)
        except Exception as exc:
            logger.error("Failed to load history: %s", exc)
            self._history_df = None

    def set_quality_scores(self, quality_scores: list) -> None:
        """
        Add current quality scores (from quality_engine output).
        
        Parameters
        ----------
        quality_scores : list
            List of QualityScore objects with .symbol and .quality_score attributes.
        """
        self._quality_map = {
            qs.symbol: qs.quality_score 
            for qs in quality_scores if hasattr(qs, 'symbol') and hasattr(qs, 'quality_score')
        }

    # ─────────────────────────────────────────────────────────────────────────
    #  Analysis 1: Per-symbol oscillation metrics (30-day window)
    # ─────────────────────────────────────────────────────────────────────────

    def compute_oscillation_metrics(
        self,
        stable_results: list,
        universe: str,
        run_date: Optional[date] = None,
        window_days: int = 30,
    ) -> list[OscillationMetrics]:
        """
        Compute oscillation metrics for all symbols.
        
        Parameters
        ----------
        stable_results : list
            List of StableRegimeResult objects with .symbol, .stable_regime, etc.
        universe : str
            Universe name (e.g., "NIFTY500").
        run_date : date, optional
            Reference date (defaults to today).
        window_days : int
            Lookback window for computing changes (default 30).
        
        Returns
        -------
        list[OscillationMetrics]
            Per-symbol metrics.
        """
        run_date = run_date or date.today()
        if self._history_df is None or self._history_df.empty:
            logger.warning("No history available for oscillation analysis")
            return []

        metrics_list = []
        symbols = {getattr(r, 'symbol', None) for r in stable_results if getattr(r, 'symbol', None)}
        symbols.update(set(self._history_df.get('symbol', pd.Series(dtype=str)).dropna().astype(str)))

        for symbol in sorted(symbols):
            sym_history = self._get_symbol_history(symbol, run_date, days_back=window_days)
            if sym_history.empty:
                current_regime = "UNCERTAIN"
                prior_regime = "UNCERTAIN"
                regime_changes = 0
                unique_regimes = 1
                oscillations = 0
                avg_conf = 0.0
            else:
                current_regime = str(sym_history.iloc[-1]['effective_regime']) if 'effective_regime' in sym_history.columns else "UNCERTAIN"
                prior_regime = str(sym_history.iloc[-2]['effective_regime']) if len(sym_history) > 1 and 'effective_regime' in sym_history.columns else "UNCERTAIN"
                regime_changes = self._count_regime_changes(sym_history)
                unique_regimes = self._count_unique_regimes(sym_history)
                oscillations = self._count_oscillations(sym_history)
                avg_conf = float(sym_history['confidence'].mean()) if 'confidence' in sym_history.columns else 0.0

            metrics = OscillationMetrics(
                symbol=symbol,
                current_regime=current_regime,
                stable_regime=current_regime,
                prior_regime=prior_regime,
                oscillation_detected=oscillations > 0,
                regime_changes_30d=regime_changes,
                unique_regimes_seen_30d=unique_regimes,
                oscillation_count_30d=oscillations,
                avg_confidence_30d=avg_conf,
                avg_smoothed_confidence_30d=avg_conf,
                current_quality_score=self._quality_map.get(symbol),
                universe=universe,
            )
            metrics_list.append(metrics)

        return metrics_list

    # ─────────────────────────────────────────────────────────────────────────
    #  Analysis 2: Regime transition matrix (all history)
    # ─────────────────────────────────────────────────────────────────────────

    def compute_transition_matrix(self) -> list[RegimeTransition]:
        """
        Count all regime transitions across all history and symbols.
        
        Returns
        -------
        list[RegimeTransition]
            Transitions sorted descending by count.
        """
        if self._history_df is None or self._history_df.empty:
            logger.warning("No history available for transition matrix")
            return []

        transitions: dict[tuple[str, str], int] = {}

        for symbol, group in self._history_df.groupby('symbol'):
            if 'date' in group.columns:
                group = group.sort_values('date')

            regimes = group['effective_regime'].dropna().astype(str).tolist()
            for i in range(len(regimes) - 1):
                from_r = str(regimes[i])
                to_r = str(regimes[i + 1])
                if from_r == to_r:
                    continue
                key = (from_r, to_r)
                transitions[key] = transitions.get(key, 0) + 1

        # Convert to list and compute percentages
        total = sum(transitions.values())
        result = [
            RegimeTransition(
                from_regime=fr,
                to_regime=tr,
                transition_count=count,
                transition_pct=100.0 * count / total if total > 0 else 0.0,
            )
            for (fr, tr), count in transitions.items()
        ]
        
        # Sort descending
        result.sort(key=lambda x: x.transition_count, reverse=True)
        return result

    # ─────────────────────────────────────────────────────────────────────────
    #  Analysis 3: Regime stability statistics
    # ─────────────────────────────────────────────────────────────────────────

    def compute_regime_stability(self) -> list[RegimeStabilityStats]:
        """
        Compute duration statistics for each regime across all history.
        
        Returns
        -------
        list[RegimeStabilityStats]
            Stability stats for each regime.
        """
        if self._history_df is None or self._history_df.empty:
            logger.warning("No history available for stability analysis")
            return []

        regime_durations: dict[str, list[int]] = {r: [] for r in _ALL_REGIMES}

        for _, group in self._history_df.groupby('symbol'):
            if 'date' in group.columns:
                group = group.sort_values('date')

            regimes = group['effective_regime'].dropna().astype(str).tolist()
            if not regimes:
                continue

            current_regime = regimes[0]
            current_duration = 1
            for regime in regimes[1:]:
                if regime == current_regime:
                    current_duration += 1
                else:
                    if current_regime in regime_durations:
                        regime_durations[current_regime].append(current_duration)
                    current_regime = regime
                    current_duration = 1

            if current_regime in regime_durations:
                regime_durations[current_regime].append(current_duration)

        # Compute statistics
        result = []
        for regime in _ALL_REGIMES:
            durations = regime_durations[regime]
            if not durations:
                continue
            
            stats = RegimeStabilityStats(
                regime=regime,
                avg_duration_bars=float(np.mean(durations)),
                median_duration_bars=float(np.median(durations)),
                p25_duration_bars=float(np.percentile(durations, 25)),
                p75_duration_bars=float(np.percentile(durations, 75)),
                max_duration_bars=int(np.max(durations)),
                min_duration_bars=int(np.min(durations)),
                episode_count=len(durations),
            )
            result.append(stats)
        
        # Sort by avg duration descending
        result.sort(key=lambda x: x.avg_duration_bars, reverse=True)
        return result

    # ─────────────────────────────────────────────────────────────────────────
    #  Analysis 4: Oscillation root cause summary
    # ─────────────────────────────────────────────────────────────────────────

    def compute_oscillation_summary(
        self,
        oscillation_metrics: list[OscillationMetrics],
        transition_matrix: list[RegimeTransition],
        stability_stats: list[RegimeStabilityStats],
        universe: str,
    ) -> OscillationSummary:
        """
        Aggregate oscillation analysis to identify root causes.
        
        Parameters
        ----------
        oscillation_metrics : list[OscillationMetrics]
            Per-symbol metrics.
        transition_matrix : list[RegimeTransition]
            Regime transitions.
        stability_stats : list[RegimeStabilityStats]
            Regime stability.
        universe : str
            Universe name.
        
        Returns
        -------
        OscillationSummary
            Aggregate report.
        """
        total_symbols = len(oscillation_metrics)
        oscillating = sum(1 for m in oscillation_metrics if m.oscillation_detected)
        oscillation_pct = 100.0 * oscillating / total_symbols if total_symbols > 0 else 0.0
        
        top_transition = transition_matrix[0] if transition_matrix else None
        top_pair = f"{top_transition.from_regime} -> {top_transition.to_regime}" if top_transition else "N/A"
        top_count = top_transition.transition_count if top_transition else 0
        
        avg_changes = np.mean([m.regime_changes_30d for m in oscillation_metrics]) if oscillation_metrics else 0.0
        avg_unique = np.mean([m.unique_regimes_seen_30d for m in oscillation_metrics]) if oscillation_metrics else 0.0
        avg_oscillations = np.mean([m.oscillation_count_30d for m in oscillation_metrics]) if oscillation_metrics else 0.0
        
        # Find most stable and unstable regimes
        most_stable = stability_stats[0] if stability_stats else None
        most_unstable = stability_stats[-1] if stability_stats else None
        
        summary = OscillationSummary(
            universe=universe,
            total_symbols=total_symbols,
            oscillating_symbols=oscillating,
            oscillation_pct=oscillation_pct,
            top_transition_pair=top_pair,
            top_transition_pair_count=top_count,
            avg_regime_changes=float(avg_changes),
            avg_unique_regimes=float(avg_unique),
            avg_oscillation_count=float(avg_oscillations),
            most_stable_regime=most_stable.regime if most_stable else "N/A",
            most_stable_duration=most_stable.avg_duration_bars if most_stable else 0.0,
            most_unstable_regime=most_unstable.regime if most_unstable else "N/A",
            most_unstable_duration=most_unstable.avg_duration_bars if most_unstable else 0.0,
        )
        
        return summary

    # ─────────────────────────────────────────────────────────────────────────
    #  Persistence
    # ─────────────────────────────────────────────────────────────────────────

    def persist(
        self,
        oscillation_metrics: list[OscillationMetrics],
        top_oscillators: list[OscillationMetrics],
        transition_matrix: list[RegimeTransition],
        stability_stats: list[RegimeStabilityStats],
        oscillation_summary: OscillationSummary,
        output_root: str,
        run_date: Optional[date] = None,
    ) -> None:
        """
        Persist all diagnostics to parquet files.
        
        Parameters
        ----------
        oscillation_metrics : list[OscillationMetrics]
            Per-symbol metrics.
        top_oscillators : list[OscillationMetrics]
            Top 50 most oscillating stocks.
        transition_matrix : list[RegimeTransition]
            Regime transitions.
        stability_stats : list[RegimeStabilityStats]
            Regime stability.
        oscillation_summary : OscillationSummary
            Aggregate report.
        output_root : str
            Output directory root.
        run_date : date, optional
            Run date (defaults to today).
        """
        run_date = run_date or date.today()
        output_dir = Path(output_root) / "analytics" / run_date.strftime("%Y-%m-%d")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Oscillation analysis
        if oscillation_metrics:
            osc_df = pd.DataFrame([m.to_dict() for m in oscillation_metrics])
            osc_path = output_dir / "oscillation_analysis.parquet"
            osc_df.to_parquet(osc_path, engine="pyarrow", compression="snappy", index=False)
            logger.info(f"Persisted oscillation_analysis: {len(osc_df)} symbols → {osc_path}")
        
        # Top oscillators
        if top_oscillators:
            top_df = pd.DataFrame([m.to_dict() for m in top_oscillators])
            top_path = output_dir / "top_oscillators.parquet"
            top_df.to_parquet(top_path, engine="pyarrow", compression="snappy", index=False)
            logger.info(f"Persisted top_oscillators: {len(top_df)} symbols → {top_path}")
        
        # Transition matrix
        if transition_matrix:
            trans_df = pd.DataFrame([t.to_dict() for t in transition_matrix])
            trans_path = output_dir / "regime_transition_matrix.parquet"
            trans_df.to_parquet(trans_path, engine="pyarrow", compression="snappy", index=False)
            logger.info(f"Persisted transition_matrix: {len(trans_df)} transitions → {trans_path}")
        
        # Stability stats
        if stability_stats:
            stab_df = pd.DataFrame([s.to_dict() for s in stability_stats])
            stab_path = output_dir / "regime_stability.parquet"
            stab_df.to_parquet(stab_path, engine="pyarrow", compression="snappy", index=False)
            logger.info(f"Persisted regime_stability: {len(stab_df)} regimes → {stab_path}")
        
        # Oscillation summary
        summary_df = pd.DataFrame([oscillation_summary.to_dict()])
        summary_path = output_dir / "oscillation_summary.parquet"
        summary_df.to_parquet(summary_path, engine="pyarrow", compression="snappy", index=False)
        logger.info(f"Persisted oscillation_summary → {summary_path}")

    # ─────────────────────────────────────────────────────────────────────────
    #  Logging Summary
    # ─────────────────────────────────────────────────────────────────────────

    @staticmethod
    def log_summary(
        summary: OscillationSummary,
        top_oscillators: list[OscillationMetrics],
        stability_stats: list[RegimeStabilityStats],
    ) -> None:
        """Log console summary of regime diagnostics."""
        logger.info("=" * 70)
        logger.info("REGIME DIAGNOSTICS SUMMARY")
        logger.info("=" * 70)
        logger.info(f"Universe:                    {summary.universe}")
        logger.info(f"Total Symbols:               {summary.total_symbols}")
        logger.info(f"Oscillating Symbols:         {summary.oscillating_symbols}")
        logger.info(f"Oscillation Rate:            {summary.oscillation_pct:.1f}%")
        logger.info(f"Avg Regime Changes (30d):    {summary.avg_regime_changes:.2f}")
        logger.info(f"Avg Unique Regimes (30d):    {summary.avg_unique_regimes:.2f}")
        logger.info(f"Avg Oscillations (30d):      {summary.avg_oscillation_count:.2f}")
        logger.info("")
        logger.info(f"Most Common Transition:      {summary.top_transition_pair}")
        logger.info(f"  Count:                     {summary.top_transition_pair_count}")
        logger.info("")
        logger.info(f"Most Stable Regime:          {summary.most_stable_regime}")
        logger.info(f"  Avg Duration:              {summary.most_stable_duration:.1f} bars")
        logger.info("")
        logger.info(f"Most Unstable Regime:        {summary.most_unstable_regime}")
        logger.info(f"  Avg Duration:              {summary.most_unstable_duration:.1f} bars")
        logger.info("")
        logger.info("Top 5 Most Oscillating Stocks:")
        for i, stock in enumerate(top_oscillators[:5], 1):
            logger.info(
                f"  {i}. {stock.symbol:12} | "
                f"Oscillations={stock.oscillation_count_30d:2d} | "
                f"Changes={stock.regime_changes_30d:2d} | "
                f"Regime={stock.stable_regime}"
            )
        logger.info("=" * 70)

    # ─────────────────────────────────────────────────────────────────────────
    #  Private helpers
    # ─────────────────────────────────────────────────────────────────────────

    def _get_symbol_history(
        self,
        symbol: str,
        run_date: date,
        days_back: int = 30,
    ) -> pd.DataFrame:
        """Get history for a single symbol within a lookback window."""
        if self._history_df is None or self._history_df.empty:
            return pd.DataFrame()
        
        cutoff = run_date - timedelta(days=days_back)
        sym_hist = self._history_df[self._history_df['symbol'] == symbol].copy()
        
        if 'date' in sym_hist.columns:
            sym_hist['date'] = pd.to_datetime(sym_hist['date'])
            sym_hist = sym_hist[(sym_hist['date'] >= cutoff) & (sym_hist['date'] <= run_date)]
        
        return sym_hist

    @staticmethod
    def _count_regime_changes(history: pd.DataFrame) -> int:
        """Count regime transitions in a history."""
        if history.empty or len(history) < 2:
            return 0

        regime_col = 'effective_regime'
        if regime_col not in history.columns:
            regime_col = 'stable_regime' if 'stable_regime' in history.columns else 'regime'
        if regime_col not in history.columns:
            return 0

        regimes = history[regime_col].dropna().astype(str).values
        changes = sum(1 for i in range(len(regimes) - 1) if regimes[i] != regimes[i+1])
        return changes

    @staticmethod
    def _count_unique_regimes(history: pd.DataFrame) -> int:
        """Count unique regimes in history."""
        if history.empty:
            return 0
        
        regime_col = 'effective_regime'
        if regime_col not in history.columns:
            regime_col = 'stable_regime' if 'stable_regime' in history.columns else 'regime'
        if regime_col not in history.columns:
            return 0
        
        return int(history[regime_col].dropna().astype(str).nunique())

    @staticmethod
    def _count_oscillations(history: pd.DataFrame) -> int:
        """
        Count back-and-forth transitions (oscillations).

        Example: A -> B -> A = 1 oscillation
        """
        if history.empty or len(history) < 3:
            return 0

        regime_col = 'effective_regime'
        if regime_col not in history.columns:
            regime_col = 'stable_regime' if 'stable_regime' in history.columns else 'regime'
        if regime_col not in history.columns:
            return 0

        regimes = history[regime_col].dropna().astype(str).values
        oscillations = 0

        for i in range(len(regimes) - 2):
            if regimes[i] != regimes[i + 1] and regimes[i] == regimes[i + 2]:
                oscillations += 1

        return oscillations
