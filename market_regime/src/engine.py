"""
engine.py — Market Regime Engine (orchestrator)
================================================
The single public-facing class that wires together all modules.
Callers import ONLY this file; the internal modules are an
implementation detail.

Typical usage
-------------
    from src.engine import MarketRegimeEngine
    import pandas as pd

    engine = MarketRegimeEngine()
    df     = pd.read_csv("nifty_daily.csv", parse_dates=["date"], index_col="date")
    result = engine.analyze(df)

    import json
    print(json.dumps(result.to_dict(), indent=2))
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import pandas as pd

from .config_loader import EngineConfig
from .indicators import IndicatorCalculator
from .signals import SignalExtractor
from .scorer import RegimeScorer
from .classifier import RegimeClassifier
from .models import RegimeResult


class MarketRegimeEngine:
    """
    End-to-end market regime classifier.

    Wiring (in order)
    -----------------
    1. IndicatorCalculator  → IndicatorSnapshot  (numeric values)
    2. SignalExtractor       → RegimeSignals      (boolean flags)
    3. RegimeScorer          → dict[regime, score] (weighted scores)
    4. RegimeClassifier      → RegimeResult       (final answer + JSON)

    Parameters
    ----------
    config_path : Path | None
        Path to a custom config.yaml.  If None, uses the default at
        config/config.yaml (or the REGIME_CONFIG env-var).
    """

    def __init__(self, config_path: Optional[Path] = None) -> None:
        self.config     = EngineConfig(config_path)
        self._calc      = IndicatorCalculator(self.config)
        self._extractor = SignalExtractor(self.config)
        self._scorer    = RegimeScorer(self.config)
        self._clf       = RegimeClassifier(self.config)

    # ──────────────────────────────────────────────────────────────
    #  Public API
    # ──────────────────────────────────────────────────────────────

    def analyze(self, df: pd.DataFrame, row_index: int = -1) -> RegimeResult:
        """
        Classify the market regime for a single bar in `df`.

        Parameters
        ----------
        df : pd.DataFrame
            Daily OHLCV data.  Expected columns (case-insensitive):
            open, high, low, close, volume.
        row_index : int
            Which bar to classify.  Default -1 → most recent bar.

        Returns
        -------
        RegimeResult
            Contains regime name, confidence score, signals, and
            per-regime raw scores.  Call .to_dict() for JSON output.
        """
        # Step 1 — compute numeric indicators
        snapshot = self._calc.compute(df, row_index)

        # Step 2 — derive boolean signals from the snapshot
        signals  = self._extractor.extract(snapshot)

        # Step 3 — score each candidate regime
        scores   = self._scorer.score_all(signals)

        # Step 4 — classify and build the final result
        result   = self._clf.classify(scores, signals, snapshot)

        return result

    def analyze_to_json(self, df: pd.DataFrame, row_index: int = -1) -> str:
        """
        Convenience wrapper that returns the result as a JSON string.

        Parameters
        ----------
        df : pd.DataFrame
        row_index : int

        Returns
        -------
        str  — pretty-printed JSON
        """
        result = self.analyze(df, row_index)
        return json.dumps(result.to_dict(), indent=2)

    def analyze_rolling(self, df: pd.DataFrame) -> list[RegimeResult]:
        """
        Classify every bar in df (useful for backtests / charting).

        Bars that lack sufficient history to compute all indicators
        will return UNCERTAIN with confidence 0.

        Parameters
        ----------
        df : pd.DataFrame

        Returns
        -------
        list[RegimeResult]  — one entry per row, in chronological order.
        """
        results = []
        for i in range(len(df)):
            try:
                result = self.analyze(df, row_index=i)
            except Exception:
                from .models import MarketRegime, RegimeSignals
                result = RegimeResult(
                    regime=MarketRegime.UNCERTAIN,
                    confidence=0.0,
                    signals=RegimeSignals(),
                    scores={},
                )
            results.append(result)
        return results
