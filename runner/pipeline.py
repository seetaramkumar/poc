"""
runner/pipeline.py
==================
AlgoTradingPipeline — complete pipeline with all phases wired.

Pipeline steps per universe
---------------------------
1.  Load symbol list
2.  Fetch benchmark OHLCV
3.  Classify market regime
4.  Batch-fetch all stock OHLCV
5.  Data quality validation
6.  Universe filter
7.  Stock regime classification  (with Phase 3 market-context scoring)
8.  Regime stability             (smoothing + hysteresis)
9.  Breadth engine               (Phase 4)
10. Sector engine                (Phase 5)
11. Opportunity quality scoring
12. Score diagnostics persist
13. Strategy router              (Phase 7)
14. Persist all outputs
15. Regime analytics
"""

from __future__ import annotations

import logging
import sys
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Optional

import pandas as pd
import yaml

logger = logging.getLogger(__name__)

_DEFAULT_PIPELINE_CONFIG = Path(__file__).parent / "config" / "pipeline.yaml"


# ─────────────────────────────────────────────────────────────────────────────
#  Output containers
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class UniverseRunResult:
    universe:         str
    market_regime:    dict
    stock_results:    list
    quality_scores:   list
    routing_decisions:list
    breadth_snapshot: object          # BreadthSnapshot | None
    sector_snapshots: list
    failed_symbols:   dict[str, str]
    symbols_loaded:   int   = 0
    accepted_count:   int   = 0
    excluded_count:   int   = 0
    rejected_count:   int   = 0
    benchmark_df:     Optional[pd.DataFrame] = None
    run_date:         date  = field(default_factory=date.today)


@dataclass
class PipelineRunOutput:
    market_results:    dict[str, dict]              = field(default_factory=dict)
    stock_results:     dict[str, list]              = field(default_factory=dict)
    quality_scores:    dict[str, list]              = field(default_factory=dict)
    routing_decisions: dict[str, list]              = field(default_factory=dict)
    breadth:           dict[str, object]            = field(default_factory=dict)
    sectors:           dict[str, list]              = field(default_factory=dict)
    universe_details:  dict[str, UniverseRunResult] = field(default_factory=dict)
    run_date:          date                         = field(default_factory=date.today)
    elapsed_seconds:   float                        = 0.0


# ─────────────────────────────────────────────────────────────────────────────
#  Symbol file loader
# ─────────────────────────────────────────────────────────────────────────────

class SymbolFileLoader:
    def __init__(self, project_root: Path) -> None:
        self._root = project_root

    def load(self, symbol_source: str, max_symbols: Optional[int] = None,
             allow_missing: bool = False) -> list[str]:
        path = Path(symbol_source)
        if not path.is_absolute():
            path = self._root / path
        if not path.exists():
            msg = (f"Symbol file not found: '{path}'\n"
                   f"  python scripts/build_nifty500.py\n"
                   f"  python scripts/build_sp500.py")
            if allow_missing:
                logger.warning(msg); return []
            raise FileNotFoundError(msg)
        tickers = [
            l.strip() for l in path.read_text(encoding="utf-8").splitlines()
            if l.strip() and not l.strip().startswith("#")
        ]
        if not tickers:
            logger.warning("Symbol file '%s' is empty.", path); return []
        if max_symbols is not None:
            original = len(tickers)
            tickers  = tickers[:max_symbols]
            logger.info("max_symbols=%d: using %d of %d.", max_symbols, len(tickers), original)
        else:
            logger.info("Loaded %d tickers from '%s'.", len(tickers), path.name)
        return tickers


# ─────────────────────────────────────────────────────────────────────────────
#  Pipeline
# ─────────────────────────────────────────────────────────────────────────────

class AlgoTradingPipeline:
    def __init__(self, config_path: Optional[Path] = None,
                 project_root: Optional[Path] = None) -> None:
        self._root     = project_root or Path(__file__).parent.parent
        self._cfg_path = config_path  or _DEFAULT_PIPELINE_CONFIG
        self._cfg      = self._load_config(self._cfg_path)
        self._run_date = date.today()

        root_str = str(self._root)
        if root_str not in sys.path:
            sys.path.insert(0, root_str)

        self._setup_logging()
        self._build_engines()
        self._symbol_loader = SymbolFileLoader(project_root=self._root)

    # ──────────────────────────────────────────────────────────────────────────
    #  Public API
    # ──────────────────────────────────────────────────────────────────────────

    def run(self, universes: Optional[list[str]] = None,
            run_date: Optional[date] = None,
            persist: Optional[bool] = None,
            max_symbols: Optional[int] = None) -> PipelineRunOutput:

        import time
        t0 = time.time()

        self._run_date = run_date or date.today()
        do_persist     = persist if persist is not None else self._cfg["output"]["persist"]
        sym_cfg        = self._cfg.get("symbol_loading", {})
        eff_max        = max_symbols if max_symbols is not None else sym_cfg.get("max_symbols")

        available = list(self._cfg["universes"].keys())
        targets   = universes or available
        unknown   = set(targets) - set(available)
        if unknown:
            raise ValueError(f"Unknown universe(s): {unknown}. Available: {available}")

        output = PipelineRunOutput(run_date=self._run_date)

        logger.info("=" * 60)
        logger.info("Pipeline run  date=%s  universes=%s  max=%s",
                    self._run_date, targets, eff_max or "all")
        logger.info("=" * 60)

        for universe_name in targets:
            result = self._run_universe(
                universe_name, self._cfg["universes"][universe_name],
                do_persist, eff_max, sym_cfg,
            )
            output.market_results[universe_name]    = result.market_regime
            output.stock_results[universe_name]     = result.stock_results
            output.quality_scores[universe_name]    = result.quality_scores
            output.routing_decisions[universe_name] = result.routing_decisions
            output.breadth[universe_name]           = result.breadth_snapshot
            output.sectors[universe_name]           = result.sector_snapshots
            output.universe_details[universe_name]  = result

        output.elapsed_seconds = round(time.time() - t0, 2)
        logger.info("=" * 60)
        logger.info("Pipeline complete in %.1fs", output.elapsed_seconds)
        self._log_summary(output)
        logger.info("=" * 60)
        return output

    def run_single_universe(self, universe_name: str,
                            run_date: Optional[date] = None,
                            persist: bool = True,
                            max_symbols: Optional[int] = None) -> UniverseRunResult:
        output = self.run(universes=[universe_name], run_date=run_date,
                          persist=persist, max_symbols=max_symbols)
        return output.universe_details[universe_name]

    # ──────────────────────────────────────────────────────────────────────────
    #  Per-universe pipeline
    # ──────────────────────────────────────────────────────────────────────────

    def _run_universe(self, universe_name, universe_cfg, persist, max_symbols, sym_cfg):
        from stock_regime.filters        import UniverseFilter
        from stock_regime.quality        import DataQualityValidator
        from stock_regime.stability      import RegimeStabiliser
        from stock_regime.quality_engine import OpportunityQualityEngine
        from stock_regime.analytics      import RegimeAnalytics
        from stock_regime.breadth_engine import BreadthEngine
        from stock_regime.sector_engine  import SectorEngine
        from stock_regime.strategy_router import StrategyRouter
        from stock_regime.src.models     import MarketRegimeInput

        benchmark_symbol = universe_cfg["benchmark"]
        symbol_source    = universe_cfg["symbol_source"]
        exchange         = universe_cfg.get("exchange", "NSE")
        allow_missing    = sym_cfg.get("allow_missing_file", False)

        data_cfg = self._cfg["data"]
        start    = data_cfg["start_date"]
        end      = data_cfg["end_date"]
        if end == "today":
            end = self._run_date.strftime("%Y-%m-%d")

        logger.info("─" * 50)
        logger.info("Universe: %s | benchmark=%s | exchange=%s",
                    universe_name, benchmark_symbol, exchange)
        logger.info("─" * 50)

        failed_symbols: dict[str, str] = {}
        output_root = self._cfg["output"]["root_dir"]

        # ── 1. Symbols ────────────────────────────────────────────────
        symbols = self._symbol_loader.load(
            symbol_source, max_symbols=max_symbols, allow_missing=allow_missing,
        )
        logger.info("Symbol source: %d symbols to process", len(symbols))

        # ── 2. Benchmark ──────────────────────────────────────────────
        benchmark_df = self._fetch_benchmark(benchmark_symbol, start, end)

        # ── 3. Market regime ──────────────────────────────────────────
        market_result = self._classify_market(benchmark_df, benchmark_symbol)
        logger.info("Market regime: %s (conf=%.2f)",
                    market_result["regime"], market_result["confidence"])

        if not symbols:
            logger.warning("No symbols for '%s' — skipping.", universe_name)
            return self._empty_result(universe_name, market_result, failed_symbols,
                                      benchmark_df)

        # ── 4. Fetch stocks ───────────────────────────────────────────
        stock_data, fetch_errors = self._fetch_stocks(symbols, start, end)
        failed_symbols.update(fetch_errors)

        # ── 5. Data quality ───────────────────────────────────────────
        q_cfg     = getattr(self._stock_engine.config, "quality", None)
        validator = DataQualityValidator(
            fatal_spike_pct  = float(getattr(q_cfg, "fatal_spike_pct",  0.60)),
            warn_spike_pct   = float(getattr(q_cfg, "warn_spike_pct",   0.20)),
            zero_volume_fill = bool( getattr(q_cfg, "zero_volume_fill", True)),
        )
        q_report = validator.validate(stock_data)
        for sym, fatals in q_report.excluded.items():
            failed_symbols[sym] = f"quality_fatal:{fatals[0].check}"
        if persist:
            self._persist_quality(q_report, universe_name)

        # ── 6. Universe filter ────────────────────────────────────────
        f_cfg = getattr(self._stock_engine.config, "filters", None)
        if f_cfg is not None:
            uni_filter = UniverseFilter.from_config(
                self._stock_engine.config, universe=universe_name, exchange=exchange,
            )
            f_result = uni_filter.apply(q_report.clean, self._run_date)
            for sym, reasons in f_result.rejected.items():
                failed_symbols[sym] = reasons[0].reason if reasons else "filtered"
            if persist:
                self._persist_filter(f_result, universe_name)
            clean_data     = f_result.accepted
            rejected_count = len(f_result.rejected)
        else:
            clean_data     = q_report.clean
            rejected_count = 0

        excluded_count = len(q_report.excluded)
        accepted_count = len(clean_data)
        logger.info("[%s] excluded=%d  rejected=%d  proceeding=%d",
                    universe_name, excluded_count, rejected_count, accepted_count)

        if not clean_data:
            logger.warning("No symbols remain for '%s'.", universe_name)
            return self._empty_result(universe_name, market_result, failed_symbols,
                                      benchmark_df, len(symbols), 0, excluded_count, rejected_count)

        # ── 7. Classify stocks (Phase 3: context-aware scoring) ───────
        market_ctx  = MarketRegimeInput.from_dict(market_result)
        raw_results = self._stock_engine.analyze_universe(
            stock_data=clean_data, market_regime=market_ctx,
            benchmark_data=benchmark_df, market_label=universe_name,
            run_date=self._run_date, persist=persist,
        )

        # ── 8. Regime stability ───────────────────────────────────────
        s_cfg = getattr(self._stock_engine.config, "stability", None)
        stabiliser = RegimeStabiliser(
            confirmation_bars       = int(  getattr(s_cfg, "confirmation_bars",       3)),
            uncertain_propagates    = bool( getattr(s_cfg, "uncertain_propagates",    False)),
            smoothing_enabled       = bool( getattr(s_cfg, "smoothing_enabled",       True)),
            smoothing_alpha         = float(getattr(s_cfg, "smoothing_alpha",         0.30)),
            hysteresis_threshold    = float(getattr(s_cfg, "hysteresis_threshold",    0.05)),
            persistence_window      = int(  getattr(s_cfg, "persistence_window",      5)),
            regime_switch_threshold = float(getattr(s_cfg, "regime_switch_threshold", 0.55)),
        )
        history_path = Path(output_root) / "regime_history" / "regime_history.parquet"
        history      = RegimeStabiliser.load_history(
            str(history_path) if history_path.exists() else None
        )
        stable_results = stabiliser.apply(raw_results, history, self._run_date)
        if persist:
            RegimeStabiliser.save_history(stable_results, str(history_path))

        for r in stable_results:
            if not r.is_valid():
                failed_symbols[r.symbol] = r.error or "unknown"

        # ── 9. Breadth engine (Phase 4) ───────────────────────────────
        breadth_engine  = BreadthEngine.from_config(self._stock_engine.config)
        prior_breadth   = BreadthEngine.load_prior(output_root, universe_name)
        breadth_snap    = breadth_engine.compute(
            stable_results, universe_name, self._run_date, prior_breadth
        )
        if persist:
            breadth_engine.persist(breadth_snap, output_root)

        # ── 10. Sector engine (Phase 5) ───────────────────────────────
        sector_engine   = SectorEngine.from_config(
            self._stock_engine.config, universe_name, self._root
        )
        sector_snaps    = sector_engine.compute(stable_results, universe_name, self._run_date)
        sector_states   = {
            sym: sector_engine.get_sector(sym)
            for sym in clean_data.keys()
        }
        # Map symbol → sector state (LEADING/NEUTRAL/LAGGING)
        sector_state_map = {}
        for snap in sector_snaps:
            sector_state_map[snap.sector] = snap.sector_state
        symbol_sector_state = {
            sym: sector_state_map.get(sector_states.get(sym, "UNKNOWN"), "NEUTRAL")
            for sym in clean_data.keys()
        }
        if persist:
            sector_engine.persist(sector_snaps, output_root)

        # ── 11. Quality scoring ───────────────────────────────────────
        quality_engine = OpportunityQualityEngine.from_config(self._stock_engine.config)
        quality_scores = quality_engine.evaluate_batch(
            stable_results, clean_data, run_date=self._run_date
        )
        if persist:
            quality_engine.persist(quality_scores, output_root, universe=universe_name)

        # ── 12. Score diagnostics ─────────────────────────────────────
        if persist:
            self._persist_score_diagnostics(stable_results, universe_name)
            self._persist_stable(stable_results, universe_name)

        # ── 13. Strategy router (Phase 7) ────────────────────────────
        router   = StrategyRouter.from_config(self._stock_engine.config)
        decisions = router.route_batch(
            stable_results   = stable_results,
            quality_scores   = quality_scores,
            market_regime    = market_result["regime"],
            breadth_state    = breadth_snap.breadth_state,
            sector_states    = symbol_sector_state,
            run_date         = self._run_date,
        )
        if persist:
            router.persist(decisions, output_root, universe=universe_name)

        # ── 14. Analytics ─────────────────────────────────────────────
        if persist:
            self._run_analytics(universe_name, output_root)

        if failed_symbols:
            logger.warning("%d failed in '%s': %s",
                           len(failed_symbols), universe_name,
                           list(failed_symbols.keys())[:10])

        return UniverseRunResult(
            universe          = universe_name,
            market_regime     = market_result,
            stock_results     = stable_results,
            quality_scores    = quality_scores,
            routing_decisions = decisions,
            breadth_snapshot  = breadth_snap,
            sector_snapshots  = sector_snaps,
            failed_symbols    = failed_symbols,
            symbols_loaded    = len(symbols),
            accepted_count    = accepted_count,
            excluded_count    = excluded_count,
            rejected_count    = rejected_count,
            benchmark_df      = benchmark_df,
            run_date          = self._run_date,
        )

    # ──────────────────────────────────────────────────────────────────────────
    #  Data helpers
    # ──────────────────────────────────────────────────────────────────────────

    def _fetch_benchmark(self, symbol: str, start: str, end: str) -> pd.DataFrame:
        logger.info("Fetching benchmark: %s  (%s → %s)", symbol, start, end)
        df = self._data_manager.get_daily_data(symbol, start=start, end=end)
        logger.info("  %s: %d bars (%s → %s)",
                    symbol, len(df), df.index[0].date(), df.index[-1].date())
        return df

    def _fetch_stocks(self, symbols, start, end):
        logger.info("Batch-fetching %d stocks …", len(symbols))
        results    = self._data_manager.fetch_multiple_symbols(symbols, start=start, end=end)
        successful: dict[str, pd.DataFrame] = {}
        failed:     dict[str, str]          = {}
        cached_n   = 0
        for sym, res in results.items():
            if res.success and not res.data.empty:
                successful[sym] = res.data
                if res.from_cache: cached_n += 1
            else:
                failed[sym] = res.error or "empty data"
        logger.info("  OK=%d  failed=%d  from_cache=%d",
                    len(successful), len(failed), cached_n)
        return successful, failed

    def _classify_market(self, df, symbol):
        result = self._market_engine.analyze(df)
        logger.debug("  %s: %s (%.2f)", symbol, result.regime.value, result.confidence)
        return result.to_dict()

    # ──────────────────────────────────────────────────────────────────────────
    #  Persistence helpers
    # ──────────────────────────────────────────────────────────────────────────

    def _output_dir(self, category: str) -> Path:
        p = (Path(self._cfg["output"]["root_dir"]) / category /
             self._run_date.strftime("%Y-%m-%d"))
        p.mkdir(parents=True, exist_ok=True)
        return p

    def _persist_quality(self, report, universe):
        records = report.all_anomalies_as_records()
        if not records: return
        out  = self._output_dir("quality")
        path = out / f"{universe.lower()}_quality.parquet"
        df   = pd.DataFrame(records)
        df["run_date"] = self._run_date
        df["universe"] = universe
        df.to_parquet(path, engine="pyarrow", compression="snappy", index=False)
        logger.info("Quality anomalies [%s]: %d → '%s'.", universe, len(records), path)

    def _persist_filter(self, result, universe):
        out = self._output_dir("filters")
        if result.summary:
            pd.DataFrame([result.summary.to_dict()]).to_parquet(
                out / f"{universe.lower()}_filter_summary.parquet",
                engine="pyarrow", compression="snappy", index=False,
            )
        records = result.rejected_symbols_as_records()
        if records:
            df = pd.DataFrame(records)
            df["run_date"] = self._run_date; df["universe"] = universe
            df.to_parquet(out / f"{universe.lower()}_rejected.parquet",
                          engine="pyarrow", compression="snappy", index=False)

    def _persist_stable(self, stable_results, universe):
        if not stable_results: return
        rows = []
        for r in stable_results:
            ds = r.dimensional_scores
            cs = ds.continuous
            row = {
                "symbol": r.symbol, "market": r.market,
                "run_date": r.run_date,
                "raw_regime": r.stock_regime.value,
                "stable_regime": r.stable_regime.value,
                "prior_stable_regime": r.prior_stable_regime.value,
                "confidence": r.confidence,
                "smoothed_confidence": r.smoothed_confidence,
                "regime_age_bars": r.regime_age_bars,
                "stable_regime_age": r.stable_regime_age,
                "regime_changed_today": r.regime_changed_today,
                "oscillation_detected": r.oscillation_detected,
                "trend_score": ds.trend, "momentum_score": ds.momentum,
                "volatility_score": ds.volatility, "is_valid": r.is_valid(),
            }
            if cs:
                row.update({
                    "adx_score": cs.adx_score,
                    "ema_alignment_score": cs.ema_alignment_score,
                    "instability_score": cs.instability_score,
                    "ranging_score": cs.ranging_score,
                    "rs_score": cs.rs_score,
                    "roc_score": cs.roc_score,
                })
            rows.append(row)
        out  = self._output_dir("stable_classifications")
        path = out / f"{universe.lower()}_stable.parquet"
        pd.DataFrame(rows).to_parquet(
            path, engine="pyarrow", compression="snappy", index=False)
        logger.info("Stable classifications [%s]: %d → '%s'.", universe, len(rows), path)

    def _persist_score_diagnostics(self, stable_results, universe):
        if not stable_results: return
        rows = [{"symbol": r.symbol,
                 "trend_score": r.dimensional_scores.trend,
                 "momentum_score": r.dimensional_scores.momentum,
                 "volatility_score": r.dimensional_scores.volatility,
                 "stable_regime": r.stable_regime.value if hasattr(r, "stable_regime") else r.stock_regime.value,
                 "confidence": r.confidence,
                 "smoothed_conf": getattr(r, "smoothed_confidence", r.confidence),
                 } for r in stable_results]
        df   = pd.DataFrame(rows)
        out  = self._output_dir("scoring")
        path = out / f"{universe.lower()}_score_dist.parquet"
        df.to_parquet(path, engine="pyarrow", compression="snappy", index=False)
        for col in ["trend_score", "momentum_score"]:
            if col in df.columns and len(df) > 0:
                logger.info("ScoreDiag [%s] %s: mean=%.3f  std=%.3f  P90=%.3f",
                            universe, col, df[col].mean(), df[col].std(),
                            df[col].quantile(0.90))

    def _run_analytics(self, universe, output_root):
        from stock_regime.analytics import RegimeAnalytics
        hp = Path(output_root) / "regime_history" / "regime_history.parquet"
        if not hp.exists(): return
        try:
            rpt = RegimeAnalytics(hp, min_episode_bars=2).compute(
                universe, as_of_date=self._run_date
            )
            RegimeAnalytics(hp).persist(rpt, output_root)
        except Exception as exc:
            logger.warning("Analytics failed for '%s': %s", universe, exc)

    def _empty_result(self, universe, market_result, failed_symbols, benchmark_df,
                      symbols_loaded=0, accepted_count=0, excluded_count=0, rejected_count=0):
        return UniverseRunResult(
            universe=universe, market_regime=market_result,
            stock_results=[], quality_scores=[], routing_decisions=[],
            breadth_snapshot=None, sector_snapshots=[],
            failed_symbols=failed_symbols,
            symbols_loaded=symbols_loaded, accepted_count=accepted_count,
            excluded_count=excluded_count, rejected_count=rejected_count,
            benchmark_df=benchmark_df, run_date=self._run_date,
        )

    # ──────────────────────────────────────────────────────────────────────────
    #  Engine init
    # ──────────────────────────────────────────────────────────────────────────

    def _build_engines(self):
        from trading_data import DataManager, DataManagerConfig
        from market_regime.src import MarketRegimeEngine
        from stock_regime.src import StockRegimeEngine

        data_cfg   = self._cfg["data"]
        output_cfg = self._cfg["output"]

        self._data_manager = DataManager(config=DataManagerConfig(
            cache_enabled=True, cache_dir=data_cfg["cache_dir"],
            cache_max_age_days=data_cfg["cache_max_age_days"],
            retry_attempts=data_cfg["retry_attempts"],
        ))
        logger.info("DataManager ready.")

        mr_cfg = self._cfg.get("market_regime_config")
        self._market_engine = MarketRegimeEngine(
            config_path=Path(mr_cfg) if mr_cfg else None)
        logger.info("MarketRegimeEngine ready.")

        sr_cfg = self._cfg.get("stock_regime_config")
        self._stock_engine = StockRegimeEngine(
            config_path=Path(sr_cfg) if sr_cfg else None,
            output_dir=output_cfg["root_dir"],
        )
        logger.info("StockRegimeEngine ready.")

    def _setup_logging(self):
        log_cfg   = self._cfg.get("output", {})
        log_dir   = log_cfg.get("log_dir", "output/logs")
        log_level = getattr(logging, log_cfg.get("log_level", "INFO").upper(), logging.INFO)
        from stock_regime.src.logging_config import configure_logging
        configure_logging(log_dir=log_dir, console_level=log_level)
        for name in ("runner", "market_regime", "trading_data", "stock_regime"):
            logging.getLogger(name).setLevel(log_level)

    @staticmethod
    def _load_config(path):
        if not path.exists():
            raise FileNotFoundError(f"Pipeline config not found: {path}")
        with open(path) as fh:
            return yaml.safe_load(fh)

    @staticmethod
    def _log_summary(output):
        for universe, results in output.stock_results.items():
            mr      = output.market_results.get(universe, {})
            detail  = output.universe_details.get(universe)
            qs      = output.quality_scores.get(universe, [])
            routing = output.routing_decisions.get(universe, [])
            breadth = output.breadth.get(universe)
            avg_q   = sum(q.quality_score for q in qs) / len(qs) if qs else 0.0
            allowed = sum(1 for d in routing if d.allowed)
            logger.info(
                "%-10s  market=%-14s (%.2f)  classified=%d  "
                "quality=%.2f  routed=%d/%d  breadth=%s",
                universe, mr.get("regime", "N/A"), mr.get("confidence", 0.0),
                len(results), avg_q, allowed, len(routing),
                getattr(breadth, "breadth_state", "N/A") if breadth else "N/A",
            )
            counts: dict[str, int] = {}
            for r in results:
                k = r.stable_regime.value if hasattr(r,"stable_regime") else r.stock_regime.value
                counts[k] = counts.get(k, 0) + 1
            for regime, count in sorted(counts.items(), key=lambda x: -x[1]):
                logger.info("  %-18s %3d", regime, count)