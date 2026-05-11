"""
stock_regime/src/engine.py
============================
StockRegimeEngine — the single public-facing class that wires together
all internal modules and exposes a clean batch-processing API.

Typical usage
-------------
    from stock_regime.src import StockRegimeEngine
    from stock_regime.src.models import MarketRegimeInput

    engine = StockRegimeEngine()

    market_ctx = MarketRegimeInput.from_dict({
        "regime": "BULLISH_TREND",
        "confidence": 0.82,
    })

    results = engine.analyze_universe(
        stock_data   = {"INFY": df_infy, "RELIANCE": df_rel, "AAPL": df_aapl},
        market_regime = market_ctx,
        benchmark_data = df_nifty,
        market_label  = "NIFTY500",
        persist       = True,
    )

    for r in results:
        print(r.to_dict())

Pipeline (per stock)
---------------------
1. StockIndicatorCalculator  → StockIndicatorSnapshot  (numeric values)
2. StockSignalExtractor       → StockSignals            (boolean flags)
3. StockRegimeScorer          → regime_scores + dimensional_scores
4. StockRegimeClassifier      → StockRegimeResult       (final output)

Post-batch
----------
5. StockRanker                → RankingOutput
6. OutputPersistence          → parquet files
"""

from __future__ import annotations

import logging
from datetime import date
from pathlib import Path
from typing import Optional

import pandas as pd

from .classifier import StockRegimeClassifier
from .config_loader import StockEngineConfig
from .indicators import StockIndicatorCalculator
from .models import (
    DimensionalScores,
    MarketRegimeInput,
    StockRegime,
    StockRegimeResult,
    StockSignals,
    StockIndicatorSnapshot,
)
from .persistence import OutputPersistence
from .ranker import RankingOutput, StockRanker
from .scorer import StockRegimeScorer
from .signals import StockSignalExtractor

logger = logging.getLogger(__name__)


class StockRegimeEngine:
    """
    End-to-end stock regime classifier for a universe of symbols.

    Parameters
    ----------
    config_path :
        Path to a custom config.yaml.  When ``None``, uses
        ``config/config.yaml`` relative to this package (or the
        ``STOCK_REGIME_CONFIG`` environment variable).
    output_dir :
        Root directory for all persisted outputs.
        Defaults to ``"output"`` relative to the caller's working directory.
    """

    def __init__(
        self,
        config_path: Optional[Path] = None,
        output_dir: str | Path = "output",
    ) -> None:
        self.config   = StockEngineConfig(config_path)

        # Wire all pipeline layers
        self._calc    = StockIndicatorCalculator(self.config)
        self._extractor = StockSignalExtractor(self.config)
        self._scorer  = StockRegimeScorer(self.config)
        self._clf     = StockRegimeClassifier(self.config)
        self._ranker  = StockRanker(self.config)
        self._persist = OutputPersistence(output_dir)

        logger.info("StockRegimeEngine initialised.")

    # ──────────────────────────────────────────────────────────────
    #  Primary public API
    # ──────────────────────────────────────────────────────────────

    def analyze_universe(
        self,
        stock_data: dict[str, pd.DataFrame],
        market_regime: MarketRegimeInput,
        benchmark_data: Optional[pd.DataFrame] = None,
        market_label: str = "UNKNOWN",
        run_date: Optional[date] = None,
        persist: bool = True,
    ) -> list[StockRegimeResult]:
        """
        Classify every stock in ``stock_data`` and return ranked results.

        Parameters
        ----------
        stock_data :
            ``{ symbol: ohlcv_df }`` mapping.  Each DataFrame must have
            lowercase columns: ``open, high, low, close, volume`` and
            a ``DatetimeIndex``.  This is the normalised format produced
            by the ``trading_data`` layer.
        market_regime :
            Current market regime context from the Market Regime Engine.
        benchmark_data :
            Optional benchmark OHLCV DataFrame (same schema) used to
            compute relative strength for all stocks.  When ``None``,
            RS-based signals are silently disabled.
        market_label :
            Universe identifier stored in each result (e.g. ``"NIFTY500"``).
        run_date :
            Date label used for output partitioning.  Defaults to today.
        persist :
            When ``True``, writes all outputs to parquet.

        Returns
        -------
        list[StockRegimeResult]
            One result per input symbol, in input order.
            Failed symbols are included with ``error`` set.
        """
        run_date = run_date or date.today()

        logger.info(
            "analyze_universe: %d symbols | market=%s | regime=%s (conf=%.2f)",
            len(stock_data),
            market_label,
            market_regime.regime,
            market_regime.confidence,
        )

        results: list[StockRegimeResult] = []

        for symbol, df in stock_data.items():
            result = self._analyze_single(
                symbol=symbol,
                df=df,
                market_regime=market_regime,
                benchmark_data=benchmark_data,
                market_label=market_label,
            )
            results.append(result)

        # Summarise in log
        valid_count = sum(1 for r in results if r.is_valid())
        error_count = len(results) - valid_count
        logger.info(
            "Classification complete: %d/%d valid | %d errors.",
            valid_count, len(results), error_count,
        )
        self._log_regime_summary(results)

        # Rank
        rankings = self._ranker.rank(results)

        # Persist
        if persist:
            self._persist_all(results, rankings, run_date)

        return results

    def analyze_single(
        self,
        symbol: str,
        df: pd.DataFrame,
        market_regime: MarketRegimeInput,
        benchmark_data: Optional[pd.DataFrame] = None,
        market_label: str = "UNKNOWN",
    ) -> StockRegimeResult:
        """
        Classify a single stock.  No persistence or ranking.

        Parameters
        ----------
        symbol :
            Stock ticker / identifier.
        df :
            Normalised OHLCV DataFrame.
        market_regime :
            Market regime context.
        benchmark_data :
            Optional benchmark for RS computation.
        market_label :
            Universe label.

        Returns
        -------
        StockRegimeResult
        """
        return self._analyze_single(
            symbol=symbol,
            df=df,
            market_regime=market_regime,
            benchmark_data=benchmark_data,
            market_label=market_label,
        )

    def get_rankings(
        self,
        results: list[StockRegimeResult],
    ) -> RankingOutput:
        """
        Re-rank an existing result list without re-running the pipeline.

        Useful for custom filtering before ranking.

        Parameters
        ----------
        results :
            StockRegimeResult objects from a prior call to analyze_universe.

        Returns
        -------
        RankingOutput
        """
        return self._ranker.rank(results)

    # ──────────────────────────────────────────────────────────────
    #  Internal per-stock pipeline
    # ──────────────────────────────────────────────────────────────

    def _analyze_single(
        self,
        symbol: str,
        df: pd.DataFrame,
        market_regime: MarketRegimeInput,
        benchmark_data: Optional[pd.DataFrame],
        market_label: str,
    ) -> StockRegimeResult:
        """
        Run the full pipeline for one stock.  Errors are caught and
        returned as a failed StockRegimeResult — one bad stock never
        aborts the batch.
        """
        try:
            # Step 1 — compute numeric indicators
            snap = self._calc.compute(df, row_index=-1, benchmark_df=benchmark_data)

            # Step 2 — derive boolean signals
            signals = self._extractor.extract(snap)

            # Step 3 — score all regimes + dimensional scores
            regime_scores = self._scorer.score_regimes(signals)
            dim_scores    = self._scorer.score_dimensions(signals, snap)

            # Step 4 — classify, apply context, build result
            result = self._clf.classify(
                symbol=symbol,
                market=market_label,
                regime_scores=regime_scores,
                signals=signals,
                dimensional_scores=dim_scores,
                market_regime=market_regime,
                snap=snap,
            )

            logger.debug(
                "%-15s → %-16s conf=%.2f  trend=%.2f  momentum=%.2f",
                symbol,
                result.stock_regime.value,
                result.confidence,
                dim_scores.trend,
                dim_scores.momentum,
            )

        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to classify '%s': %s", symbol, exc, exc_info=True)
            result = StockRegimeResult(
                symbol=symbol,
                market=market_label,
                stock_regime=StockRegime.UNCERTAIN,
                confidence=0.0,
                dimensional_scores=DimensionalScores(),
                signals=StockSignals(),
                indicators=StockIndicatorSnapshot(),
                error=str(exc),
            )

        return result

    # ──────────────────────────────────────────────────────────────
    #  Persistence helpers
    # ──────────────────────────────────────────────────────────────

    def _persist_all(
        self,
        results: list[StockRegimeResult],
        rankings: RankingOutput,
        run_date: date,
    ) -> None:
        """Persist all four output categories."""
        self._persist.save_indicators(results, run_date)
        self._persist.save_signals(results, run_date)
        self._persist.save_classifications(results, run_date)
        self._persist.save_rankings(rankings, run_date)

    # ──────────────────────────────────────────────────────────────
    #  Logging helpers
    # ──────────────────────────────────────────────────────────────

    @staticmethod
    def _log_regime_summary(results: list[StockRegimeResult]) -> None:
        """Log a regime distribution summary (mirrors market_regime main.py format)."""
        counts: dict[str, int] = {}
        for r in results:
            key = r.stock_regime.value
            counts[key] = counts.get(key, 0) + 1

        total = max(len(results), 1)
        logger.info("Regime distribution:")
        for regime, count in sorted(counts.items(), key=lambda x: -x[1]):
            pct = count / total * 100
            logger.info("  %-16s  %4d  (%5.1f %%)", regime, count, pct)

    def __repr__(self) -> str:
        return "StockRegimeEngine()"
