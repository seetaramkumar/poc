"""
runner/pipeline.py
==================
AlgoTradingPipeline — the single entry point that wires together:

    trading_data      → fetch OHLCV data for benchmark + universe stocks
    market_regime     → classify the market-level regime from benchmark data
    stock_regime      → classify every stock using market context + stock data

Responsibilities
----------------
This module owns the *integration* layer only.  It does not contain any
indicator logic, scoring logic, or classification logic.  Those stay in
their respective engines.

Call flow
---------
1. Load pipeline config (pipeline.yaml)
2. Build DataManager (trading_data)
3. For each universe:
   a. Fetch benchmark OHLCV (DataManager)
   b. Run MarketRegimeEngine on benchmark → MarketRegimeInput
   c. Fetch all stock OHLCV in batch (DataManager)
   d. Run StockRegimeEngine on the batch → list[StockRegimeResult]
   e. Persist outputs + return results

Usage
-----
    from runner.pipeline import AlgoTradingPipeline

    pipeline = AlgoTradingPipeline()
    run_output = pipeline.run()

    # Access results per universe
    nifty_results = run_output.stock_results["NIFTY500"]
    nifty_market  = run_output.market_results["NIFTY500"]
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Optional

import pandas as pd
import yaml

logger = logging.getLogger(__name__)

# ── Default config path (relative to this file) ─────────────────────────────
_DEFAULT_PIPELINE_CONFIG = Path(__file__).parent / "config" / "pipeline.yaml"


# ─────────────────────────────────────────────────────────────────────────────
#  Typed output container
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class UniverseRunResult:
    """
    All outputs for a single universe (e.g. NIFTY500) from one pipeline run.

    Attributes
    ----------
    universe :
        Universe label (e.g. ``"NIFTY500"``).
    market_regime :
        The market regime result for the benchmark.
    stock_results :
        List of StockRegimeResult, one per successfully classified stock.
    failed_symbols :
        Symbols that could not be fetched or classified, with error messages.
    benchmark_df :
        The benchmark OHLCV DataFrame used for market regime + RS computation.
    run_date :
        Date this universe was processed.
    """
    universe:      str
    market_regime: dict                          # RegimeResult.to_dict()
    stock_results: list                          # list[StockRegimeResult]
    failed_symbols: dict[str, str]               # {symbol: error_message}
    benchmark_df:  Optional[pd.DataFrame] = None
    run_date:      date = field(default_factory=date.today)


@dataclass
class PipelineRunOutput:
    """
    Full output of one AlgoTradingPipeline.run() call.

    Attributes
    ----------
    market_results :
        { universe_label: RegimeResult.to_dict() }
    stock_results :
        { universe_label: list[StockRegimeResult] }
    universe_details :
        { universe_label: UniverseRunResult }
    run_date :
        Date of the run.
    elapsed_seconds :
        Wall-clock time for the full run.
    """
    market_results:    dict[str, dict]               = field(default_factory=dict)
    stock_results:     dict[str, list]               = field(default_factory=dict)
    universe_details:  dict[str, UniverseRunResult]  = field(default_factory=dict)
    run_date:          date                          = field(default_factory=date.today)
    elapsed_seconds:   float                         = 0.0


# ─────────────────────────────────────────────────────────────────────────────
#  Pipeline
# ─────────────────────────────────────────────────────────────────────────────

class AlgoTradingPipeline:
    """
    Orchestrates the full data → market regime → stock regime pipeline.

    Parameters
    ----------
    config_path :
        Path to ``pipeline.yaml``.  Defaults to ``runner/config/pipeline.yaml``.
    project_root :
        Absolute path to the project root (where trading_data, market_regime,
        stock_regime packages live).  Defaults to the parent of this file.
    """

    def __init__(
        self,
        config_path: Optional[Path] = None,
        project_root: Optional[Path] = None,
    ) -> None:
        self._root       = project_root or Path(__file__).parent.parent
        self._cfg_path   = config_path or _DEFAULT_PIPELINE_CONFIG
        self._cfg        = self._load_config(self._cfg_path)
        self._run_date   = date.today()

        # Ensure project root is importable
        import sys
        root_str = str(self._root)
        if root_str not in sys.path:
            sys.path.insert(0, root_str)

        self._setup_logging()
        self._build_engines()

    # ──────────────────────────────────────────────────────────────────────────
    #  Public API
    # ──────────────────────────────────────────────────────────────────────────

    def run(
        self,
        universes: Optional[list[str]] = None,
        run_date: Optional[date] = None,
        persist: Optional[bool] = None,
    ) -> PipelineRunOutput:
        """
        Execute the full pipeline for all configured universes.

        Parameters
        ----------
        universes :
            Subset of universe names to run.  Defaults to all configured.
        run_date :
            Override the run date used for output partitioning.
        persist :
            Override the ``output.persist`` setting from config.

        Returns
        -------
        PipelineRunOutput
        """
        import time
        t0 = time.time()
        self._run_date = run_date or date.today()
        do_persist     = persist if persist is not None else self._cfg["output"]["persist"]

        available_universes = list(self._cfg["universes"].keys())
        target_universes    = universes or available_universes

        # Validate requested universes
        unknown = set(target_universes) - set(available_universes)
        if unknown:
            raise ValueError(f"Unknown universe(s): {unknown}. Available: {available_universes}")

        output = PipelineRunOutput(run_date=self._run_date)

        logger.info("=" * 60)
        logger.info("Pipeline run starting — date=%s  universes=%s", self._run_date, target_universes)
        logger.info("=" * 60)

        for universe_name in target_universes:
            universe_cfg = self._cfg["universes"][universe_name]
            result = self._run_universe(universe_name, universe_cfg, do_persist)

            output.market_results[universe_name]   = result.market_regime
            output.stock_results[universe_name]    = result.stock_results
            output.universe_details[universe_name] = result

        output.elapsed_seconds = round(time.time() - t0, 2)

        logger.info("=" * 60)
        logger.info("Pipeline complete in %.1fs", output.elapsed_seconds)
        self._log_summary(output)
        logger.info("=" * 60)

        return output

    def run_single_universe(
        self,
        universe_name: str,
        run_date: Optional[date] = None,
        persist: bool = True,
    ) -> UniverseRunResult:
        """
        Run the pipeline for a single universe.

        Convenience wrapper around ``run()`` when you only need one universe.

        Parameters
        ----------
        universe_name :
            Must match a key in ``pipeline.yaml`` universes section.
        run_date :
            Override for output partitioning.
        persist :
            Write outputs to parquet.

        Returns
        -------
        UniverseRunResult
        """
        output = self.run(universes=[universe_name], run_date=run_date, persist=persist)
        return output.universe_details[universe_name]

    # ──────────────────────────────────────────────────────────────────────────
    #  Per-universe pipeline
    # ──────────────────────────────────────────────────────────────────────────

    def _run_universe(
        self,
        universe_name: str,
        universe_cfg: dict,
        persist: bool,
    ) -> UniverseRunResult:
        """
        Full pipeline execution for one universe.

        Steps
        -----
        1. Fetch benchmark OHLCV
        2. Classify market regime
        3. Batch-fetch all stock OHLCV
        4. Classify all stocks
        5. Persist outputs
        """
        benchmark_symbol = universe_cfg["benchmark"]
        stock_symbols    = universe_cfg["symbols"]
        data_cfg         = self._cfg["data"]

        start = data_cfg["start_date"]
        end   = "today" if data_cfg["end_date"] == "today" else data_cfg["end_date"]
        if end == "today":
            end = self._run_date.strftime("%Y-%m-%d")

        logger.info("─" * 50)
        logger.info("Universe: %s  |  benchmark=%s  |  %d symbols",
                    universe_name, benchmark_symbol, len(stock_symbols))
        logger.info("─" * 50)

        failed_symbols: dict[str, str] = {}

        # ── Step 1: Fetch benchmark ──────────────────────────────────────
        benchmark_df = self._fetch_benchmark(benchmark_symbol, start, end)

        # ── Step 2: Market regime ────────────────────────────────────────
        market_result = self._classify_market(benchmark_df, benchmark_symbol)
        logger.info(
            "Market regime: %s  (confidence=%.2f)",
            market_result["regime"], market_result["confidence"],
        )

        # ── Step 3: Batch-fetch stock OHLCV ─────────────────────────────
        stock_data, fetch_errors = self._fetch_stocks(stock_symbols, start, end)
        failed_symbols.update(fetch_errors)

        if not stock_data:
            logger.warning("No stock data fetched for universe '%s'. Skipping.", universe_name)
            return UniverseRunResult(
                universe=universe_name,
                market_regime=market_result,
                stock_results=[],
                failed_symbols=failed_symbols,
                benchmark_df=benchmark_df,
                run_date=self._run_date,
            )

        # ── Step 4: Classify stocks ──────────────────────────────────────
        from stock_regime.src.models import MarketRegimeInput
        market_ctx = MarketRegimeInput.from_dict(market_result)

        stock_results = self._stock_engine.analyze_universe(
            stock_data      = stock_data,
            market_regime   = market_ctx,
            benchmark_data  = benchmark_df,
            market_label    = universe_name,
            run_date        = self._run_date,
            persist         = persist,
        )

        # Collect any engine-level failures (stocks with error set)
        for r in stock_results:
            if not r.is_valid():
                failed_symbols[r.symbol] = r.error or "unknown error"

        if failed_symbols:
            logger.warning(
                "%d symbols failed in '%s': %s",
                len(failed_symbols), universe_name, list(failed_symbols.keys()),
            )

        return UniverseRunResult(
            universe=universe_name,
            market_regime=market_result,
            stock_results=stock_results,
            failed_symbols=failed_symbols,
            benchmark_df=benchmark_df,
            run_date=self._run_date,
        )

    # ──────────────────────────────────────────────────────────────────────────
    #  Data fetching helpers
    # ──────────────────────────────────────────────────────────────────────────

    def _fetch_benchmark(self, symbol: str, start: str, end: str) -> pd.DataFrame:
        """Fetch benchmark OHLCV; raise clearly on failure."""
        logger.info("Fetching benchmark: %s  (%s → %s)", symbol, start, end)
        df = self._data_manager.get_daily_data(symbol, start=start, end=end)
        logger.info("  Benchmark %s: %d bars  (%s → %s)",
                    symbol, len(df), df.index[0].date(), df.index[-1].date())
        return df

    def _fetch_stocks(
        self,
        symbols: list[str],
        start: str,
        end: str,
    ) -> tuple[dict[str, pd.DataFrame], dict[str, str]]:
        """
        Batch-fetch OHLCV for all symbols.

        Returns
        -------
        tuple[dict, dict]
            (successful_data, failed_errors)
            successful_data: {symbol: DataFrame}
            failed_errors:   {symbol: error_message}
        """
        logger.info("Batch-fetching %d stocks  (%s → %s) …", len(symbols), start, end)
        results = self._data_manager.fetch_multiple_symbols(symbols, start=start, end=end)

        successful: dict[str, pd.DataFrame] = {}
        failed:     dict[str, str]          = {}

        for sym, result in results.items():
            if result.success and not result.data.empty:
                successful[sym] = result.data
            else:
                failed[sym] = result.error or "empty data returned"

        source_counts = {"cache": 0, "live": 0}
        for result in results.values():
            if result.success:
                source_counts["cache" if result.from_cache else "live"] += 1

        logger.info(
            "  Fetched: %d OK  |  %d failed  |  from_cache=%d  live=%d",
            len(successful), len(failed),
            source_counts["cache"], source_counts["live"],
        )
        if failed:
            logger.warning("  Failed symbols: %s", list(failed.keys()))

        return successful, failed

    def _classify_market(self, benchmark_df: pd.DataFrame, symbol: str) -> dict:
        """Run MarketRegimeEngine on the benchmark; return to_dict() output."""
        result = self._market_engine.analyze(benchmark_df)
        logger.debug(
            "  %s market regime: %s (conf=%.2f)  ADX=%.1f",
            symbol, result.regime.value, result.confidence,
            result.indicator_snapshot.adx if result.indicator_snapshot else 0,
        )
        return result.to_dict()

    # ──────────────────────────────────────────────────────────────────────────
    #  Initialisation helpers
    # ──────────────────────────────────────────────────────────────────────────

    def _build_engines(self) -> None:
        """Instantiate all three engines once at startup."""
        import sys
        sys.path.insert(0, str(self._root))

        from trading_data import DataManager, DataManagerConfig
        from market_regime.src import MarketRegimeEngine
        from stock_regime.src import StockRegimeEngine

        data_cfg = self._cfg["data"]
        output_cfg = self._cfg["output"]

        # DataManager
        dm_config = DataManagerConfig(
            cache_enabled      = True,
            cache_dir          = data_cfg["cache_dir"],
            cache_max_age_days = data_cfg["cache_max_age_days"],
            retry_attempts     = data_cfg["retry_attempts"],
        )
        self._data_manager = DataManager(config=dm_config)
        logger.info("DataManager: provider=%s  cache_dir=%s",
                    "yahoo", data_cfg["cache_dir"])

        # MarketRegimeEngine
        mr_config_path = self._cfg.get("market_regime_config")
        self._market_engine = MarketRegimeEngine(
            config_path=Path(mr_config_path) if mr_config_path else None
        )
        logger.info("MarketRegimeEngine ready.")

        # StockRegimeEngine
        sr_config_path = self._cfg.get("stock_regime_config")
        self._stock_engine = StockRegimeEngine(
            config_path=Path(sr_config_path) if sr_config_path else None,
            output_dir=output_cfg["root_dir"],
        )
        logger.info("StockRegimeEngine ready.  output_dir=%s", output_cfg["root_dir"])

    def _setup_logging(self) -> None:
        """Configure structured logging to console and rotating file."""
        log_cfg   = self._cfg.get("output", {})
        log_dir   = log_cfg.get("log_dir", "output/logs")
        log_level = getattr(logging, log_cfg.get("log_level", "INFO").upper(), logging.INFO)

        from stock_regime.src.logging_config import configure_logging
        configure_logging(log_dir=log_dir, console_level=log_level)

        # Also configure the pipeline's own logger
        logging.getLogger("runner").setLevel(log_level)
        logging.getLogger("market_regime").setLevel(log_level)
        logging.getLogger("trading_data").setLevel(log_level)

    @staticmethod
    def _load_config(path: Path) -> dict:
        if not path.exists():
            raise FileNotFoundError(f"Pipeline config not found: {path}")
        with open(path) as fh:
            return yaml.safe_load(fh)

    # ──────────────────────────────────────────────────────────────────────────
    #  Summary logging
    # ──────────────────────────────────────────────────────────────────────────

    @staticmethod
    def _log_summary(output: PipelineRunOutput) -> None:
        for universe, results in output.stock_results.items():
            market = output.market_results.get(universe, {})
            logger.info(
                "%-12s  market=%-14s (%.2f)  stocks=%d",
                universe,
                market.get("regime", "N/A"),
                market.get("confidence", 0.0),
                len(results),
            )
            counts: dict[str, int] = {}
            for r in results:
                k = r.stock_regime.value
                counts[k] = counts.get(k, 0) + 1
            for regime, count in sorted(counts.items(), key=lambda x: -x[1]):
                logger.info("  %-18s %3d", regime, count)
