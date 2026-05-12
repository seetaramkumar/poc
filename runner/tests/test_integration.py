"""
runner/tests/test_integration.py
==================================
End-to-end integration tests for the full pipeline.
No network access required — synthetic data throughout.

Run with:
    cd algo_platform/
    pytest runner/tests/test_integration.py -v
"""

from __future__ import annotations

import sys
from pathlib import Path
from textwrap import dedent

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))


# ─────────────────────────────────────────────────────────────────────────────
#  Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _make_df(n=500, drift=0.001, vol=0.01, seed=0) -> pd.DataFrame:
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


BENCHMARK = _make_df(seed=99)
STOCKS    = {
    "INFY.NS":     _make_df(drift=0.0018, seed=1),
    "RELIANCE.NS": _make_df(drift=0.0012, seed=2),
    "HDFCBANK.NS": _make_df(drift=-0.001, seed=3),
}


# ─────────────────────────────────────────────────────────────────────────────
#  SymbolFileLoader tests
# ─────────────────────────────────────────────────────────────────────────────

class TestSymbolFileLoader:
    """Test the symbol file reader in isolation."""

    def _loader(self):
        from runner.pipeline import SymbolFileLoader
        return SymbolFileLoader(project_root=ROOT)

    def test_loads_tickers_from_file(self, tmp_path):
        f = tmp_path / "test.txt"
        f.write_text("AAPL\nMSFT\nGOOGL\n")
        loader  = self._loader()
        tickers = loader.load(str(f))
        assert tickers == ["AAPL", "MSFT", "GOOGL"]

    def test_skips_blank_lines_and_comments(self, tmp_path):
        f = tmp_path / "test.txt"
        f.write_text("AAPL\n# this is a comment\n\nMSFT\n")
        tickers = self._loader().load(str(f))
        assert tickers == ["AAPL", "MSFT"]

    def test_max_symbols_applied(self, tmp_path):
        f = tmp_path / "test.txt"
        f.write_text("\n".join(["TICK" + str(i) for i in range(100)]))
        tickers = self._loader().load(str(f), max_symbols=10)
        assert len(tickers) == 10
        assert tickers[0] == "TICK0"

    def test_missing_file_raises_by_default(self):
        loader = self._loader()
        with pytest.raises(FileNotFoundError, match="scripts/build"):
            loader.load("data/universes/nonexistent.txt", allow_missing=False)

    def test_missing_file_allowed_returns_empty(self):
        loader  = self._loader()
        tickers = loader.load("data/universes/nonexistent.txt", allow_missing=True)
        assert tickers == []

    def test_relative_path_resolved_from_project_root(self, tmp_path):
        """Relative paths are resolved against the project root, not cwd."""
        from runner.pipeline import SymbolFileLoader
        loader = SymbolFileLoader(project_root=tmp_path)
        (tmp_path / "data" / "universes").mkdir(parents=True)
        (tmp_path / "data" / "universes" / "test.txt").write_text("INFY.NS\n")
        tickers = loader.load("data/universes/test.txt")
        assert tickers == ["INFY.NS"]

    def test_empty_file_returns_empty_list(self, tmp_path):
        f = tmp_path / "empty.txt"
        f.write_text("")
        assert self._loader().load(str(f)) == []


# ─────────────────────────────────────────────────────────────────────────────
#  Script normalisation tests (no network)
# ─────────────────────────────────────────────────────────────────────────────

class TestBuildScripts:
    """Test the normalisation helpers in each build script."""

    def test_nifty_to_yahoo_standard(self):
        from scripts.build_nifty500 import _to_yahoo_ticker
        assert _to_yahoo_ticker("RELIANCE") == "RELIANCE.NS"
        assert _to_yahoo_ticker("  TCS  ")  == "TCS.NS"
        assert _to_yahoo_ticker("infy")     == "INFY.NS"

    def test_nifty_to_yahoo_override(self):
        from scripts.build_nifty500 import _to_yahoo_ticker
        assert _to_yahoo_ticker("M&M") == "M-M.NS"

    def test_nifty_normalise_dataframe(self):
        from scripts.build_nifty500 import _normalise
        import pandas as pd
        df = pd.DataFrame({"Symbol": ["RELIANCE", "TCS", "INFY", "M&M"]})
        tickers = _normalise(df)
        assert "RELIANCE.NS" in tickers
        assert "TCS.NS"      in tickers
        assert "M-M.NS"      in tickers

    def test_sp500_to_yahoo_dot_replacement(self):
        from scripts.build_sp500 import _to_yahoo_ticker
        assert _to_yahoo_ticker("BRK.B") == "BRK-B"  # override
        assert _to_yahoo_ticker("BF.B")  == "BF-B"   # override
        assert _to_yahoo_ticker("AAPL")  == "AAPL"   # no change

    def test_sp500_normalise_dataframe(self):
        from scripts.build_sp500 import _normalise
        import pandas as pd
        df = pd.DataFrame({"Symbol": ["AAPL", "MSFT", "BRK.B", "BF.B"]})
        tickers = _normalise(df)
        assert "AAPL"  in tickers
        assert "MSFT"  in tickers
        assert "BRK-B" in tickers
        assert "BF-B"  in tickers

    def test_sp500_column_name_variants(self):
        """Normaliser should handle 'Ticker' as well as 'Symbol'."""
        from scripts.build_sp500 import _normalise
        import pandas as pd
        df = pd.DataFrame({"Ticker": ["AAPL", "MSFT"]})
        tickers = _normalise(df)
        assert "AAPL" in tickers

    def test_nifty_dry_run_does_not_write(self, tmp_path):
        from scripts.build_nifty500 import _write
        out = tmp_path / "test.txt"
        _write(["RELIANCE.NS", "TCS.NS"], out, dry_run=True)
        assert not out.exists()

    def test_sp500_dry_run_does_not_write(self, tmp_path):
        from scripts.build_sp500 import _write
        out = tmp_path / "test.txt"
        _write(["AAPL", "MSFT"], out, dry_run=True)
        assert not out.exists()

    def test_write_creates_parent_dirs(self, tmp_path):
        from scripts.build_nifty500 import _write
        out = tmp_path / "deep" / "path" / "nifty500.txt"
        _write(["RELIANCE.NS"], out, dry_run=False)
        assert out.exists()
        assert out.read_text().strip() == "RELIANCE.NS"

    def test_build_from_local_nifty_csv(self, tmp_path):
        """build() works end-to-end from a local CSV (no network needed)."""
        import pandas as pd
        from scripts.build_nifty500 import build
        csv = tmp_path / "ind_nifty500list.csv"
        pd.DataFrame({
            "Symbol": ["RELIANCE", "TCS", "INFY", "HDFCBANK", "WIPRO"],
            "Company Name": ["Reliance", "TCS", "Infosys", "HDFC Bank", "Wipro"],
        }).to_csv(csv, index=False)
        out     = tmp_path / "nifty500.txt"
        tickers = build(output_path=out, csv_path=csv)
        assert "RELIANCE.NS" in tickers
        assert "TCS.NS"      in tickers
        assert out.exists()

    def test_build_from_local_sp500_csv(self, tmp_path):
        """build() works end-to-end from a local CSV (no network needed)."""
        import pandas as pd
        from scripts.build_sp500 import build
        csv = tmp_path / "constituents.csv"
        pd.DataFrame({
            "Symbol": ["AAPL", "MSFT", "GOOGL", "BRK.B"],
            "Name": ["Apple", "Microsoft", "Alphabet", "Berkshire"],
        }).to_csv(csv, index=False)
        out     = tmp_path / "sp500.txt"
        tickers = build(output_path=out, csv_path=csv)
        assert "AAPL"  in tickers
        assert "BRK-B" in tickers
        assert out.exists()


# ─────────────────────────────────────────────────────────────────────────────
#  Direct engine integration (no pipeline config required)
# ─────────────────────────────────────────────────────────────────────────────

class TestDirectIntegration:

    def test_market_to_stock_bridge(self):
        """RegimeResult.to_dict() → MarketRegimeInput.from_dict() must be lossless."""
        from market_regime.src import MarketRegimeEngine
        from stock_regime.src.models import MarketRegimeInput

        result = MarketRegimeEngine().analyze(BENCHMARK)
        d      = result.to_dict()
        ctx    = MarketRegimeInput.from_dict(d)
        assert ctx.regime     == d["regime"]
        assert ctx.confidence == pytest.approx(d["confidence"])

    def test_full_three_engine_pipeline(self):
        """trading_data schema → market regime → stock regime, no errors."""
        from market_regime.src import MarketRegimeEngine
        from stock_regime.src import StockRegimeEngine
        from stock_regime.src.models import MarketRegimeInput

        ctx     = MarketRegimeInput.from_dict(
            MarketRegimeEngine().analyze(BENCHMARK).to_dict()
        )
        results = StockRegimeEngine(output_dir="/tmp/test_out").analyze_universe(
            stock_data=STOCKS, market_regime=ctx,
            benchmark_data=BENCHMARK, market_label="TEST", persist=False,
        )
        assert len(results) == len(STOCKS)
        for r in results:
            assert 0.0 <= r.confidence <= 1.0

    def test_all_results_serialise_to_valid_json(self):
        import json
        from stock_regime.src import StockRegimeEngine
        from stock_regime.src.models import MarketRegimeInput

        ctx     = MarketRegimeInput(regime="BULLISH_TREND", confidence=0.75)
        results = StockRegimeEngine(output_dir="/tmp/test_out").analyze_universe(
            STOCKS, ctx, BENCHMARK, "TEST", persist=False
        )
        for r in results:
            parsed = json.loads(json.dumps(r.to_dict()))
            assert parsed["symbol"] == r.symbol


# ─────────────────────────────────────────────────────────────────────────────
#  Pipeline orchestrator tests (mocked DataManager)
# ─────────────────────────────────────────────────────────────────────────────

class TestPipelineOrchestrator:

    @pytest.fixture
    def symbol_files(self, tmp_path):
        """Write minimal symbol files for testing."""
        uni = tmp_path / "data" / "universes"
        uni.mkdir(parents=True)
        (uni / "nifty500.txt").write_text(
            "INFY.NS\n# comment\nRELIANCE.NS\nHDFCBANK.NS\n"
        )
        (uni / "sp500.txt").write_text("AAPL\nMSFT\n")
        return tmp_path

    @pytest.fixture
    def pipeline(self, tmp_path, symbol_files):
        """Pipeline backed by tmp symbol files and a minimal config."""
        cfg_text = f"""
data:
  start_date: "2021-01-01"
  end_date:   "today"
  cache_dir:  "{tmp_path}/cache"
  cache_max_age_days: 1
  retry_attempts: 1
universes:
  NIFTY500:
    benchmark:     "^NSEI"
    symbol_source: "data/universes/nifty500.txt"
    exchange:      "NSE"
symbol_loading:
  max_symbols: null
  skip_on_fetch_error: true
  allow_missing_file: false
output:
  root_dir:  "{tmp_path}/output"
  persist:   true
  log_dir:   "{tmp_path}/logs"
  log_level: "WARNING"
market_regime_config: null
stock_regime_config:  null
"""
        cfg_path = tmp_path / "pipeline.yaml"
        cfg_path.write_text(cfg_text)
        from runner.pipeline import AlgoTradingPipeline
        return AlgoTradingPipeline(
            config_path=cfg_path,
            project_root=symbol_files,
        )

    def _mock(self, pipeline):
        from trading_data.models import FetchResult
        pipeline._data_manager.get_daily_data        = lambda *a, **k: BENCHMARK.copy()
        pipeline._data_manager.fetch_multiple_symbols = lambda syms, **k: {
            s: FetchResult(symbol=s, provider="yahoo", data=_make_df(seed=i), success=True)
            for i, s in enumerate(syms)
        }

    def test_run_returns_correct_universes(self, pipeline):
        self._mock(pipeline)
        out = pipeline.run(universes=["NIFTY500"], persist=False)
        assert "NIFTY500" in out.market_results
        assert "NIFTY500" in out.stock_results

    def test_all_file_symbols_are_classified(self, pipeline):
        """3 symbols in nifty500.txt → 3 results."""
        self._mock(pipeline)
        out = pipeline.run(universes=["NIFTY500"], persist=False)
        assert len(out.stock_results["NIFTY500"]) == 3

    def test_max_symbols_override(self, pipeline):
        self._mock(pipeline)
        out = pipeline.run(universes=["NIFTY500"], persist=False, max_symbols=2)
        assert len(out.stock_results["NIFTY500"]) == 2

    def test_symbols_loaded_count_in_detail(self, pipeline):
        self._mock(pipeline)
        out    = pipeline.run(universes=["NIFTY500"], persist=False)
        detail = out.universe_details["NIFTY500"]
        assert detail.symbols_loaded == 3    # 3 non-comment lines in nifty500.txt

    def test_market_regime_keys_present(self, pipeline):
        self._mock(pipeline)
        out = pipeline.run(universes=["NIFTY500"], persist=False)
        mr  = out.market_results["NIFTY500"]
        assert "regime"     in mr
        assert "confidence" in mr

    def test_persist_writes_parquet_files(self, pipeline, tmp_path):
        self._mock(pipeline)
        pipeline.run(universes=["NIFTY500"], persist=True)
        parquets = list(Path(tmp_path / "output").rglob("*.parquet"))
        assert len(parquets) > 0

    def test_missing_symbol_file_raises(self, tmp_path):
        cfg_text = f"""
data:
  start_date: "2021-01-01"
  end_date:   "today"
  cache_dir:  "{tmp_path}/cache"
  cache_max_age_days: 1
  retry_attempts: 1
universes:
  NIFTY500:
    benchmark:     "^NSEI"
    symbol_source: "data/universes/DOES_NOT_EXIST.txt"
    exchange:      "NSE"
symbol_loading:
  allow_missing_file: false
output:
  root_dir:  "{tmp_path}/output"
  persist:   false
  log_dir:   "{tmp_path}/logs"
  log_level: "WARNING"
market_regime_config: null
stock_regime_config:  null
"""
        cfg_path = tmp_path / "pipeline.yaml"
        cfg_path.write_text(cfg_text)
        from runner.pipeline import AlgoTradingPipeline
        p = AlgoTradingPipeline(config_path=cfg_path, project_root=tmp_path)
        p._data_manager.get_daily_data = lambda *a, **k: BENCHMARK.copy()
        with pytest.raises(FileNotFoundError, match="scripts/build"):
            p.run(universes=["NIFTY500"], persist=False)

    def test_missing_symbol_file_allowed(self, tmp_path):
        cfg_text = f"""
data:
  start_date: "2021-01-01"
  end_date:   "today"
  cache_dir:  "{tmp_path}/cache"
  cache_max_age_days: 1
  retry_attempts: 1
universes:
  NIFTY500:
    benchmark:     "^NSEI"
    symbol_source: "data/universes/DOES_NOT_EXIST.txt"
    exchange:      "NSE"
symbol_loading:
  allow_missing_file: true
output:
  root_dir:  "{tmp_path}/output"
  persist:   false
  log_dir:   "{tmp_path}/logs"
  log_level: "WARNING"
market_regime_config: null
stock_regime_config:  null
"""
        cfg_path = tmp_path / "pipeline.yaml"
        cfg_path.write_text(cfg_text)
        from runner.pipeline import AlgoTradingPipeline
        p = AlgoTradingPipeline(config_path=cfg_path, project_root=tmp_path)
        p._data_manager.get_daily_data = lambda *a, **k: BENCHMARK.copy()
        out = p.run(universes=["NIFTY500"], persist=False)
        assert out.stock_results["NIFTY500"] == []

    def test_unknown_universe_raises(self, pipeline):
        with pytest.raises(ValueError, match="Unknown universe"):
            pipeline.run(universes=["DOES_NOT_EXIST"])

    def test_elapsed_time_populated(self, pipeline):
        self._mock(pipeline)
        out = pipeline.run(universes=["NIFTY500"], persist=False)
        assert out.elapsed_seconds > 0