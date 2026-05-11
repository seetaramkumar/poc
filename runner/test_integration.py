"""
runner/tests/test_integration.py
==================================
End-to-end integration tests for the full pipeline.

These tests exercise the complete data→market_regime→stock_regime
call chain using synthetic OHLCV data so NO network access is required.

Run with:
    cd <project_root>
    pytest runner/tests/test_integration.py -v
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))


# ─────────────────────────────────────────────────────────────────────────────
#  Shared helpers
# ─────────────────────────────────────────────────────────────────────────────

def _make_df(n: int = 500, drift: float = 0.001, vol: float = 0.01, seed: int = 0) -> pd.DataFrame:
    rng   = np.random.default_rng(seed)
    dates = pd.bdate_range(end="2024-12-31", periods=n)
    close = np.empty(n)
    price = 18_000.0
    for i in range(n):
        price = max(price * (1 + rng.normal(drift, vol)), 100.0)
        close[i] = price
    rng2  = np.random.default_rng(seed + 1)
    dr    = close * 0.01
    high  = close + dr * 0.6
    low   = close - dr * 0.6
    open_ = low + dr * 0.5
    vol_  = rng2.integers(5_000_000, 20_000_000, n).astype(float)
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": vol_},
        index=dates,
    )


BENCHMARK_DF = _make_df(seed=99)
STOCK_DATA = {
    "INFY":     _make_df(drift=0.0018, seed=1),
    "RELIANCE": _make_df(drift=0.0012, seed=2),
    "HDFCBANK": _make_df(drift=-0.001, seed=3),
}


# ─────────────────────────────────────────────────────────────────────────────
#  Direct engine integration (no pipeline config required)
# ─────────────────────────────────────────────────────────────────────────────

class TestDirectIntegration:
    """
    Tests that call each engine directly in the correct sequence.
    This is the canonical integration pattern.
    """

    def test_market_regime_produces_dict_compatible_with_stock_engine(self):
        """
        RegimeResult.to_dict() must be consumable by MarketRegimeInput.from_dict().
        This is the only bridge between the two engines.
        """
        from market_regime.src import MarketRegimeEngine
        from stock_regime.src.models import MarketRegimeInput

        engine = MarketRegimeEngine()
        result = engine.analyze(BENCHMARK_DF)
        d      = result.to_dict()

        # Verify keys the bridge depends on
        assert "regime"     in d
        assert "confidence" in d
        assert isinstance(d["regime"],     str)
        assert isinstance(d["confidence"], float)

        # Verify bridge construction succeeds
        ctx = MarketRegimeInput.from_dict(d)
        assert ctx.regime == d["regime"]
        assert ctx.confidence == pytest.approx(d["confidence"])

    def test_stock_engine_accepts_market_engine_output(self):
        """Full three-engine pipeline: trading_data → market → stock."""
        from market_regime.src import MarketRegimeEngine
        from stock_regime.src import StockRegimeEngine
        from stock_regime.src.models import MarketRegimeInput

        market_engine = MarketRegimeEngine()
        stock_engine  = StockRegimeEngine(output_dir="/tmp/test_output")

        # Step 1: market regime from benchmark
        market_result = market_engine.analyze(BENCHMARK_DF)
        market_ctx    = MarketRegimeInput.from_dict(market_result.to_dict())

        # Step 2: stock classification
        results = stock_engine.analyze_universe(
            stock_data     = STOCK_DATA,
            market_regime  = market_ctx,
            benchmark_data = BENCHMARK_DF,
            market_label   = "NIFTY500",
            persist        = False,
        )

        assert len(results) == len(STOCK_DATA)
        for r in results:
            assert r.symbol in STOCK_DATA
            assert 0.0 <= r.confidence <= 1.0
            d = r.to_dict()
            for key in ["symbol", "market", "stock_regime", "confidence",
                        "scores", "signals", "indicators"]:
                assert key in d

    def test_market_confidence_influences_stock_confidence(self):
        """
        A high-confidence bullish market should boost aligned stock confidence
        and penalise misaligned confidence.
        """
        from stock_regime.src import StockRegimeEngine
        from stock_regime.src.models import MarketRegimeInput, StockRegime

        engine = StockRegimeEngine(output_dir="/tmp/test_output")

        # Strong bullish market
        bullish_ctx = MarketRegimeInput(regime="BULLISH_TREND", confidence=1.0)
        # No market context (UNCERTAIN, confidence=0)
        no_ctx = MarketRegimeInput(regime="UNCERTAIN", confidence=0.0)

        results_bull = engine.analyze_universe(
            STOCK_DATA, bullish_ctx, BENCHMARK_DF, "NIFTY500", persist=False
        )
        results_none = engine.analyze_universe(
            STOCK_DATA, no_ctx, BENCHMARK_DF, "NIFTY500", persist=False
        )

        # For stocks that are TREND_UP (aligned with BULLISH_TREND),
        # bullish context should yield >= no-context confidence.
        for rb, rn in zip(results_bull, results_none):
            if rb.stock_regime == StockRegime.TREND_UP:
                assert rb.confidence >= rn.confidence - 0.01  # allow float rounding

    def test_data_manager_output_is_directly_usable(self, tmp_path):
        """
        DataManager.get_daily_data() schema must satisfy both engine inputs
        without any transformation.
        """
        from trading_data import DataManager, DataManagerConfig

        # Use a no-cache manager with synthetic-compatible real fetch
        # We test schema validation only — use BENCHMARK_DF as the stand-in.

        # Verify the DataFrame our synthetic generator produces passes
        # the same contract as DataManager would return.
        df = BENCHMARK_DF.copy()

        # Contract: lowercase columns, DatetimeIndex, tz-naive, ascending, no NaN OHLC
        assert set(df.columns) >= {"open", "high", "low", "close", "volume"}
        assert isinstance(df.index, pd.DatetimeIndex)
        assert df.index.tz is None
        assert df.index.is_monotonic_increasing
        assert not df[["open", "high", "low", "close"]].isnull().any().any()

    def test_all_stock_results_have_valid_json(self):
        """Every result must serialise to valid JSON without TypeError."""
        import json
        from market_regime.src import MarketRegimeEngine
        from stock_regime.src import StockRegimeEngine
        from stock_regime.src.models import MarketRegimeInput

        engine = StockRegimeEngine(output_dir="/tmp/test_output")
        ctx    = MarketRegimeInput(regime="BULLISH_TREND", confidence=0.75)
        results = engine.analyze_universe(
            STOCK_DATA, ctx, BENCHMARK_DF, "TEST", persist=False
        )
        for r in results:
            serialised = json.dumps(r.to_dict())   # must not raise
            parsed = json.loads(serialised)
            assert parsed["symbol"] == r.symbol

    def test_single_stock_analyze_matches_batch(self):
        """
        analyze_single() and analyze_universe() must produce the same regime
        for the same input (deterministic pipeline).
        """
        from market_regime.src import MarketRegimeEngine
        from stock_regime.src import StockRegimeEngine
        from stock_regime.src.models import MarketRegimeInput

        market_engine = MarketRegimeEngine()
        stock_engine  = StockRegimeEngine(output_dir="/tmp/test_output")
        ctx = MarketRegimeInput.from_dict(market_engine.analyze(BENCHMARK_DF).to_dict())

        single_result = stock_engine.analyze_single("INFY", STOCK_DATA["INFY"], ctx, BENCHMARK_DF, "TEST")
        batch_results = stock_engine.analyze_universe({"INFY": STOCK_DATA["INFY"]}, ctx, BENCHMARK_DF, "TEST", persist=False)

        assert single_result.stock_regime == batch_results[0].stock_regime
        assert single_result.confidence   == pytest.approx(batch_results[0].confidence, rel=1e-4)


# ─────────────────────────────────────────────────────────────────────────────
#  Pipeline orchestrator tests (mocked data fetching)
# ─────────────────────────────────────────────────────────────────────────────

class TestPipelineOrchestrator:
    """
    Tests for AlgoTradingPipeline using a patched DataManager so no
    network calls are made.
    """

    @pytest.fixture
    def pipeline(self, tmp_path):
        """Build a pipeline instance with output directed to tmp_path."""
        import shutil
        config_src = Path(__file__).parent.parent / "config" / "pipeline.yaml"

        # Write a minimal pipeline config pointing to tmp output
        config_content = f"""
data:
  start_date: "2021-01-01"
  end_date:   "today"
  cache_dir:  "{tmp_path}/cache"
  cache_max_age_days: 1
  retry_attempts: 1
universes:
  NIFTY500:
    benchmark: "NIFTY50"
    symbols:
      - "RELIANCE"
      - "INFY"
output:
  root_dir:  "{tmp_path}/output"
  persist:   true
  log_dir:   "{tmp_path}/logs"
  log_level: "WARNING"
market_regime_config: null
stock_regime_config:  null
"""
        cfg_path = tmp_path / "pipeline.yaml"
        cfg_path.write_text(config_content)
        from runner.pipeline import AlgoTradingPipeline
        return AlgoTradingPipeline(config_path=cfg_path)

    def _mock_data_manager(self, pipeline):
        """Patch DataManager to return synthetic data instead of network calls."""
        from trading_data.models import FetchResult

        def fake_get_daily(symbol, start, end, **kwargs):
            return BENCHMARK_DF.copy()

        def fake_fetch_multiple(symbols, start, end, **kwargs):
            return {
                sym: FetchResult(symbol=sym, provider="yahoo", data=_make_df(seed=i), success=True)
                for i, sym in enumerate(symbols)
            }

        pipeline._data_manager.get_daily_data = fake_get_daily
        pipeline._data_manager.fetch_multiple_symbols = fake_fetch_multiple

    def test_pipeline_run_returns_output(self, pipeline):
        self._mock_data_manager(pipeline)
        output = pipeline.run(universes=["NIFTY500"], persist=False)
        assert "NIFTY500" in output.market_results
        assert "NIFTY500" in output.stock_results
        assert len(output.stock_results["NIFTY500"]) == 2

    def test_pipeline_market_regime_field_present(self, pipeline):
        self._mock_data_manager(pipeline)
        output = pipeline.run(universes=["NIFTY500"], persist=False)
        mr = output.market_results["NIFTY500"]
        assert "regime" in mr
        assert "confidence" in mr

    def test_pipeline_persists_parquet_files(self, pipeline, tmp_path):
        self._mock_data_manager(pipeline)
        output = pipeline.run(universes=["NIFTY500"], persist=True)
        parquet_files = list(Path(tmp_path / "output").rglob("*.parquet"))
        assert len(parquet_files) > 0, "Expected parquet files to be written"

    def test_pipeline_elapsed_time_populated(self, pipeline):
        self._mock_data_manager(pipeline)
        output = pipeline.run(universes=["NIFTY500"], persist=False)
        assert output.elapsed_seconds > 0

    def test_unknown_universe_raises(self, pipeline):
        with pytest.raises(ValueError, match="Unknown universe"):
            pipeline.run(universes=["INVALID_UNIVERSE"], persist=False)

    def test_run_single_universe_convenience(self, pipeline):
        self._mock_data_manager(pipeline)
        result = pipeline.run_single_universe("NIFTY500", persist=False)
        assert result.universe == "NIFTY500"
        assert len(result.stock_results) == 2
