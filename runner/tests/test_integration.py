"""
runner/tests/test_integration.py
==================================
Full integration tests — all phases, no network required.
Covers: SymbolFileLoader, build scripts, filters, quality,
        stability, analytics, and pipeline orchestration.

Run:  cd algo_platform && pytest runner/tests/test_integration.py -v
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))


# ─────────────────────────────────────────────────────────────────────────────
#  Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _df(n=500, drift=0.001, vol=0.01, seed=0, start=18_000.0) -> pd.DataFrame:
    rng   = np.random.default_rng(seed)
    # Always end at today so staleness filter never rejects synthetic data
    dates = pd.bdate_range(end=pd.Timestamp.today().normalize(), periods=n)
    close = np.empty(n)
    price = start
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


BENCH  = _df(seed=99)
STOCKS = {
    "INFY.NS":     _df(drift=0.0018, seed=1),
    "RELIANCE.NS": _df(drift=0.0012, seed=2),
    "HDFCBANK.NS": _df(drift=-0.001, seed=3),
}


# ─────────────────────────────────────────────────────────────────────────────
#  Phase 1 — Universe Quality & Liquidity Filtering
# ─────────────────────────────────────────────────────────────────────────────

class TestHistoryFilter:
    def _filter(self, **kw):
        from stock_regime.filters.history_filter import HistoryFilter
        return HistoryFilter(**kw)

    def test_passes_sufficient_history(self):
        f = self._filter(min_bars=220, max_gap_days=5, max_stale_days=10)
        assert f.check("SYM", _df(n=250)) == []

    def test_rejects_insufficient_bars(self):
        f = self._filter(min_bars=220)
        reasons = f.check("SYM", _df(n=100))
        assert any(r.check == "min_bars" for r in reasons)

    def test_rejects_stale_data(self):
        # Build a df whose last bar is 30 days ago — well beyond max_stale_days=3
        df = _df(n=250)
        old_dates = pd.bdate_range(
            end=pd.Timestamp.today() - pd.Timedelta(days=30), periods=250
        )
        df.index  = old_dates
        f = self._filter(min_bars=220, max_stale_days=3)
        reasons = f.check("SYM", df)
        assert any(r.check == "stale_data" for r in reasons)

    def test_rejects_large_gap(self):
        df  = _df(n=250)
        idx = list(df.index)
        # Insert a 10-day gap 30 bars from the end
        gap_start = len(idx) - 30
        shifted   = [
            idx[i] + pd.Timedelta(days=10) if i >= gap_start else idx[i]
            for i in range(len(idx))
        ]
        df.index  = pd.DatetimeIndex(shifted)
        f = self._filter(min_bars=220, max_gap_days=5, gap_lookback_days=90)
        reasons = f.check("SYM", df)
        assert any(r.check == "data_gap" for r in reasons)


class TestPriceFilter:
    def _filter(self, **kw):
        from stock_regime.filters.price_filter import PriceFilter
        return PriceFilter(**kw)

    def test_passes_normal_price(self):
        f = self._filter(min_price=50.0, exchange="NSE")
        assert f.check("SYM", _df(start=1000.0)) == []

    def test_rejects_penny_stock(self):
        f = self._filter(min_price=50.0, exchange="NSE")
        df = _df(n=250)
        df["close"] = 10.0
        df["high"]  = 10.5
        df["low"]   = 9.5
        df["open"]  = 10.0
        reasons = f.check("SYM", df)
        assert any(r.check == "min_price" for r in reasons)

    def test_circuit_breaker_warning_not_fatal_by_default(self):
        f  = self._filter(min_price=10.0, exchange="NSE",
                          circuit_is_fatal=False, max_circuit_days=2)
        df = _df(n=250, start=500.0)
        # Inject circuit-like spikes
        df.iloc[-5, df.columns.get_loc("close")] = df["close"].iloc[-5] * 1.22
        df.iloc[-3, df.columns.get_loc("close")] = df["close"].iloc[-3] * 1.22
        df.iloc[-1, df.columns.get_loc("close")] = df["close"].iloc[-1] * 1.22
        reasons = f.check("SYM", df)
        assert len(reasons) == 0   # warning only, not fatal


class TestLiquidityFilter:
    def _filter(self, **kw):
        from stock_regime.filters.liquidity_filter import LiquidityFilter
        return LiquidityFilter(**kw)

    def test_passes_liquid_stock(self):
        # close≈1000, volume≈10M → ADV ≈ ₹1000Cr >> ₹50Cr
        f = self._filter(min_adv=50.0, adv_in_crore=True, adv_period=20)
        assert f.check("SYM", _df(n=250, start=1000.0)) == []

    def test_rejects_illiquid_stock(self):
        f  = self._filter(min_adv=500.0, adv_in_crore=True, adv_period=20)
        df = _df(n=250, start=10.0)
        df["volume"] = 100.0   # ₹1,000 daily value — far below ₹500Cr
        reasons = f.check("SYM", df)
        assert any(r.check == "min_adv" for r in reasons)

    def test_rejects_high_zero_volume_ratio(self):
        f  = self._filter(min_adv=1.0, adv_in_crore=True, adv_period=20,
                          max_zero_volume_ratio=0.05)
        df = _df(n=250, start=1000.0)
        df.iloc[-20:, df.columns.get_loc("volume")] = 0   # 100% zero last 20 bars
        reasons = f.check("SYM", df)
        assert any(r.check == "zero_volume_ratio" for r in reasons)


class TestUniverseFilter:
    def test_accepted_count(self):
        from stock_regime.filters.history_filter   import HistoryFilter
        from stock_regime.filters.price_filter     import PriceFilter
        from stock_regime.filters.liquidity_filter import LiquidityFilter
        from stock_regime.filters.universe_filter  import UniverseFilter

        f = UniverseFilter(
            history_filter   = HistoryFilter(min_bars=100, max_gap_days=10, max_stale_days=10),
            price_filter     = PriceFilter(min_price=5.0, exchange="NSE"),
            liquidity_filter = LiquidityFilter(min_adv=1.0, adv_in_crore=True, adv_period=10),
            universe         = "TEST",
        )
        result = f.apply({"A": _df(n=200, start=500.0), "B": _df(n=200, start=500.0)})
        assert len(result.accepted) == 2
        assert result.summary.accepted_count == 2

    def test_rejected_symbol_has_reasons(self):
        from stock_regime.filters.history_filter   import HistoryFilter
        from stock_regime.filters.price_filter     import PriceFilter
        from stock_regime.filters.liquidity_filter import LiquidityFilter
        from stock_regime.filters.universe_filter  import UniverseFilter

        f = UniverseFilter(
            history_filter   = HistoryFilter(min_bars=1000),  # impossible threshold
            price_filter     = PriceFilter(min_price=5.0, exchange="NSE"),
            liquidity_filter = LiquidityFilter(min_adv=1.0, adv_in_crore=True),
            universe="TEST",
        )
        result = f.apply({"A": _df(n=200)})
        assert "A" in result.rejected
        assert len(result.rejected["A"]) > 0

    def test_filter_result_records_for_parquet(self):
        from stock_regime.filters.history_filter   import HistoryFilter
        from stock_regime.filters.price_filter     import PriceFilter
        from stock_regime.filters.liquidity_filter import LiquidityFilter
        from stock_regime.filters.universe_filter  import UniverseFilter

        f = UniverseFilter(
            history_filter   = HistoryFilter(min_bars=1000),
            price_filter     = PriceFilter(min_price=5.0, exchange="NSE"),
            liquidity_filter = LiquidityFilter(min_adv=1.0, adv_in_crore=True),
            universe="TEST",
        )
        result  = f.apply({"BAD": _df(n=50)})
        records = result.rejected_symbols_as_records()
        assert len(records) > 0
        assert "symbol" in records[0]
        assert "filter_name" in records[0]


# ─────────────────────────────────────────────────────────────────────────────
#  Phase 2 — Data Quality Validation
# ─────────────────────────────────────────────────────────────────────────────

class TestDataQualityValidator:
    def _v(self, **kw):
        from stock_regime.quality import DataQualityValidator
        return DataQualityValidator(**kw)

    def test_clean_df_passes(self):
        report = self._v().validate({"SYM": _df(n=250)})
        assert "SYM" in report.clean
        assert "SYM" not in report.excluded

    def test_zero_close_is_fatal(self):
        df = _df(n=250)
        df.iloc[100, df.columns.get_loc("close")] = 0.0
        df.iloc[100, df.columns.get_loc("low")]   = 0.0
        report = self._v().validate({"BAD": df})
        assert "BAD" in report.excluded

    def test_inverted_ohlc_is_fatal(self):
        df = _df(n=250)
        df.iloc[50, df.columns.get_loc("high")] = 100.0
        df.iloc[50, df.columns.get_loc("low")]  = 200.0  # high < low
        report = self._v().validate({"BAD": df})
        assert "BAD" in report.excluded

    def test_large_return_spike_is_fatal(self):
        df = _df(n=250, start=1000.0)
        df.iloc[-1, df.columns.get_loc("close")] = 5000.0  # +400%
        report = self._v(fatal_spike_pct=0.60).validate({"BAD": df})
        assert "BAD" in report.excluded

    def test_zero_volume_is_corrected(self):
        df = _df(n=250)
        df.iloc[-5, df.columns.get_loc("volume")] = 0
        report = self._v(zero_volume_fill=True).validate({"SYM": df})
        assert "SYM" in report.clean
        # Volume should be filled (non-zero)
        clean_df = report.clean["SYM"]
        assert clean_df.iloc[-5]["volume"] > 0

    def test_close_above_high_is_corrected_not_fatal(self):
        df = _df(n=250)
        df.iloc[-1, df.columns.get_loc("close")] = df.iloc[-1]["high"] * 1.01
        report = self._v().validate({"SYM": df})
        assert "SYM" in report.clean
        clean_df = report.clean["SYM"]
        assert clean_df.iloc[-1]["close"] <= clean_df.iloc[-1]["high"] * 1.001

    def test_all_anomalies_serialisable(self):
        df = _df(n=250)
        df.iloc[-5, df.columns.get_loc("volume")] = 0
        report = self._v().validate({"SYM": df})
        records = report.all_anomalies_as_records()
        for r in records:
            assert "symbol" in r and "check" in r and "severity" in r

    def test_multiple_stocks_isolated(self):
        good = _df(n=250)
        bad  = _df(n=250)
        bad.iloc[0, bad.columns.get_loc("close")] = 0.0
        bad.iloc[0, bad.columns.get_loc("low")]   = 0.0
        report = self._v().validate({"GOOD": good, "BAD": bad})
        assert "GOOD" in report.clean
        assert "BAD"  in report.excluded


# ─────────────────────────────────────────────────────────────────────────────
#  Phase 3 — Regime Stability
# ─────────────────────────────────────────────────────────────────────────────

class TestRegimeStabiliser:
    def _make_result(self, regime_str: str, symbol="SYM"):
        from stock_regime.src.models import (
            StockRegime, StockRegimeResult, DimensionalScores,
            StockSignals, StockIndicatorSnapshot,
        )
        return StockRegimeResult(
            symbol             = symbol,
            market             = "TEST",
            stock_regime       = StockRegime(regime_str),
            confidence         = 0.75,
            dimensional_scores = DimensionalScores(),
            regime_scores      = {},
            signals            = StockSignals(),
            indicators         = StockIndicatorSnapshot(),
        )

    def test_regime_not_confirmed_below_threshold(self):
        from stock_regime.stability import RegimeStabiliser, SymbolHistory
        stab    = RegimeStabiliser(confirmation_bars=3)
        history = {"SYM": SymbolHistory(
            symbol="SYM",
            raw_regimes=["RANGE", "TREND_UP"],   # only 2 bars of TREND_UP
            stable_regime="RANGE",
            stable_age=5,
        )}
        results = stab.apply([self._make_result("TREND_UP")], history)
        assert results[0].stable_regime.value == "RANGE"
        assert results[0].regime_changed_today is False

    def test_regime_confirmed_at_threshold(self):
        from stock_regime.stability import RegimeStabiliser, SymbolHistory
        stab    = RegimeStabiliser(confirmation_bars=3)
        history = {"SYM": SymbolHistory(
            symbol="SYM",
            raw_regimes=["TREND_UP", "TREND_UP"],  # 2 prior + today = 3
            stable_regime="RANGE",
            stable_age=5,
        )}
        results = stab.apply([self._make_result("TREND_UP")], history)
        assert results[0].stable_regime.value == "TREND_UP"
        assert results[0].regime_changed_today is True

    def test_uncertain_does_not_overwrite_stable(self):
        from stock_regime.stability import RegimeStabiliser, SymbolHistory
        stab    = RegimeStabiliser(confirmation_bars=3, uncertain_propagates=False)
        history = {"SYM": SymbolHistory(
            symbol="SYM",
            raw_regimes=["TREND_UP", "TREND_UP"],
            stable_regime="TREND_UP",
            stable_age=8,
        )}
        results = stab.apply([self._make_result("UNCERTAIN")], history)
        assert results[0].stable_regime.value == "TREND_UP"

    def test_history_mutated_after_apply(self):
        from stock_regime.stability import RegimeStabiliser, SymbolHistory
        stab    = RegimeStabiliser(confirmation_bars=3)
        history: dict = {}
        stab.apply([self._make_result("TREND_UP")], history)
        assert "SYM" in history
        assert history["SYM"].raw_regimes[-1] == "TREND_UP"

    def test_regime_age_increments(self):
        from stock_regime.stability import RegimeStabiliser, SymbolHistory
        stab    = RegimeStabiliser(confirmation_bars=3)
        history = {"SYM": SymbolHistory(
            symbol="SYM",
            raw_regimes=["TREND_UP"] * 5,
            stable_regime="TREND_UP",
            stable_age=5,
        )}
        results = stab.apply([self._make_result("TREND_UP")], history)
        assert results[0].regime_age_bars == 6

    def test_save_and_load_history(self, tmp_path):
        from stock_regime.stability import RegimeStabiliser, SymbolHistory
        from stock_regime.src.models import (
            StockRegime, StockRegimeResult, DimensionalScores,
            StockSignals, StockIndicatorSnapshot,
        )
        stab    = RegimeStabiliser(confirmation_bars=3)
        history: dict = {}

        for _ in range(3):
            stab.apply([self._make_result("TREND_UP")], history)

        path = str(tmp_path / "history.parquet")

        # Build minimal StableRegimeResult for save
        raw = StockRegimeResult(
            symbol="SYM", market="TEST",
            stock_regime=StockRegime.TREND_UP, confidence=0.8,
            dimensional_scores=DimensionalScores(), regime_scores={},
            signals=StockSignals(), indicators=StockIndicatorSnapshot(),
        )
        stable_results = stab.apply([raw], history)
        RegimeStabiliser.save_history(stable_results, path, append=False)

        loaded = RegimeStabiliser.load_history(path)
        assert "SYM" in loaded


# ─────────────────────────────────────────────────────────────────────────────
#  Phase 8 — Regime Analytics
# ─────────────────────────────────────────────────────────────────────────────

class TestRegimeAnalytics:
    def _make_history_df(self, n_symbols=3, n_days=60) -> pd.DataFrame:
        """Synthetic regime history parquet data."""
        from stock_regime.src.models import StockRegime
        import random

        random.seed(42)
        regimes = ["TREND_UP", "TREND_DOWN", "RANGE", "MOMENTUM"]
        rows    = []
        dates   = pd.bdate_range(end="2024-12-31", periods=n_days)

        for i in range(n_symbols):
            sym     = f"SYM{i}"
            stable  = random.choice(regimes)
            age     = 0
            changed = False
            for j, d in enumerate(dates):
                if j % 10 == 0 and j > 0:   # regime changes every 10 bars
                    prev   = stable
                    stable = random.choice(regimes)
                    changed = stable != prev
                    age     = 1
                else:
                    age    += 1
                    changed = False
                rows.append({
                    "symbol":               sym,
                    "market":               "TEST",
                    "run_date":             d.date(),
                    "raw_regime":           stable,
                    "stable_regime":        stable,
                    "prior_stable_regime":  stable,
                    "confidence":           0.75,
                    "regime_age_bars":      age,
                    "stable_regime_age":    age,
                    "regime_changed_today": changed,
                })
        return pd.DataFrame(rows)

    def test_compute_returns_report(self, tmp_path):
        from stock_regime.analytics import RegimeAnalytics
        hist = self._make_history_df()
        p    = tmp_path / "history.parquet"
        hist.to_parquet(p, index=False)
        rpt  = RegimeAnalytics(p).compute("TEST")
        assert rpt.universe == "TEST"
        assert len(rpt.current_episodes) > 0

    def test_duration_stats_have_values(self, tmp_path):
        from stock_regime.analytics import RegimeAnalytics
        hist = self._make_history_df(n_symbols=10, n_days=120)
        p    = tmp_path / "history.parquet"
        hist.to_parquet(p, index=False)
        rpt  = RegimeAnalytics(p, min_episode_bars=2).compute("TEST")
        assert len(rpt.duration_stats) > 0
        for s in rpt.duration_stats:
            assert s.mean_bars > 0
            assert s.sample_count > 0

    def test_transition_matrix_rows_sum_to_one(self, tmp_path):
        from stock_regime.analytics import RegimeAnalytics
        hist = self._make_history_df(n_symbols=20, n_days=200)
        p    = tmp_path / "history.parquet"
        hist.to_parquet(p, index=False)
        rpt  = RegimeAnalytics(p).compute("TEST")
        if rpt.transition_matrix:
            row_sums = rpt.transition_matrix.matrix.sum(axis=1)
            # Rows with at least one transition should sum to ~1.0
            active = row_sums[row_sums > 0]
            assert (active - 1.0).abs().max() < 0.01

    def test_recent_changes_within_window(self, tmp_path):
        from stock_regime.analytics import RegimeAnalytics
        hist  = self._make_history_df(n_symbols=5, n_days=60)
        p     = tmp_path / "history.parquet"
        hist.to_parquet(p, index=False)
        today = pd.to_datetime("2024-12-31").date()
        rpt   = RegimeAnalytics(p, recent_change_days=10).compute("TEST", as_of_date=today)
        for ep in rpt.recent_changes:
            assert ep.last_date >= (today - __import__("datetime").timedelta(days=10))

    def test_persist_creates_parquet_files(self, tmp_path):
        from stock_regime.analytics import RegimeAnalytics
        hist = self._make_history_df(n_symbols=10, n_days=120)
        p    = tmp_path / "history.parquet"
        hist.to_parquet(p, index=False)
        rpt   = RegimeAnalytics(p, min_episode_bars=2).compute("TEST")
        saved = RegimeAnalytics(p).persist(rpt, tmp_path)
        assert len(saved) > 0
        for path in saved.values():
            assert path.exists()

    def test_empty_history_returns_empty_report(self, tmp_path):
        from stock_regime.analytics import RegimeAnalytics
        p = tmp_path / "empty.parquet"
        pd.DataFrame(columns=[
            "symbol","market","run_date","raw_regime","stable_regime",
            "prior_stable_regime","confidence","regime_age_bars",
            "stable_regime_age","regime_changed_today",
        ]).to_parquet(p, index=False)
        rpt = RegimeAnalytics(p).compute("TEST")
        assert rpt.duration_stats    == []
        assert rpt.current_episodes  == []


# ─────────────────────────────────────────────────────────────────────────────
#  SymbolFileLoader
# ─────────────────────────────────────────────────────────────────────────────

class TestSymbolFileLoader:
    def _loader(self):
        from runner.pipeline import SymbolFileLoader
        return SymbolFileLoader(project_root=ROOT)

    def test_loads_tickers(self, tmp_path):
        f = tmp_path / "t.txt"
        f.write_text("AAPL\nMSFT\nGOOGL\n")
        assert self._loader().load(str(f)) == ["AAPL", "MSFT", "GOOGL"]

    def test_skips_comments_and_blanks(self, tmp_path):
        f = tmp_path / "t.txt"
        f.write_text("AAPL\n# comment\n\nMSFT\n")
        assert self._loader().load(str(f)) == ["AAPL", "MSFT"]

    def test_max_symbols_applied(self, tmp_path):
        f = tmp_path / "t.txt"
        f.write_text("\n".join(f"T{i}" for i in range(50)))
        assert len(self._loader().load(str(f), max_symbols=10)) == 10

    def test_missing_file_raises(self):
        with pytest.raises(FileNotFoundError, match="scripts/build"):
            self._loader().load("data/universes/does_not_exist.txt", allow_missing=False)

    def test_missing_file_allowed(self):
        assert self._loader().load("data/universes/does_not_exist.txt", allow_missing=True) == []

    def test_empty_file_returns_empty(self, tmp_path):
        f = tmp_path / "e.txt"
        f.write_text("")
        assert self._loader().load(str(f)) == []


# ─────────────────────────────────────────────────────────────────────────────
#  Build-script normalisation (no network)
# ─────────────────────────────────────────────────────────────────────────────

class TestBuildScripts:
    def test_nifty_ticker_format(self):
        from scripts.build_nifty500 import _to_yahoo_ticker
        assert _to_yahoo_ticker("RELIANCE") == "RELIANCE.NS"
        assert _to_yahoo_ticker("M&M")      == "M-M.NS"

    def test_sp500_ticker_format(self):
        from scripts.build_sp500 import _to_yahoo_ticker
        assert _to_yahoo_ticker("AAPL")  == "AAPL"
        assert _to_yahoo_ticker("BRK.B") == "BRK-B"

    def test_nifty_build_from_local_csv(self, tmp_path):
        from scripts.build_nifty500 import build
        csv = tmp_path / "nifty.csv"
        pd.DataFrame({"Symbol": ["RELIANCE", "TCS", "INFY", "M&M"]}).to_csv(csv, index=False)
        tickers = build(output_path=tmp_path / "nifty500.txt", csv_path=csv)
        assert "RELIANCE.NS" in tickers
        assert "M-M.NS"      in tickers

    def test_sp500_build_from_local_csv(self, tmp_path):
        from scripts.build_sp500 import build
        csv = tmp_path / "sp500.csv"
        pd.DataFrame({"Symbol": ["AAPL", "MSFT", "BRK.B"]}).to_csv(csv, index=False)
        tickers = build(output_path=tmp_path / "sp500.txt", csv_path=csv)
        assert "AAPL"  in tickers
        assert "BRK-B" in tickers

    def test_dry_run_does_not_write(self, tmp_path):
        from scripts.build_nifty500 import build
        csv = tmp_path / "nifty.csv"
        pd.DataFrame({"Symbol": ["RELIANCE", "TCS"]}).to_csv(csv, index=False)
        out = tmp_path / "out.txt"
        build(output_path=out, csv_path=csv, dry_run=True)
        assert not out.exists()


# ─────────────────────────────────────────────────────────────────────────────
#  Direct engine integration
# ─────────────────────────────────────────────────────────────────────────────

class TestDirectIntegration:
    def test_market_to_stock_bridge(self):
        from market_regime.src import MarketRegimeEngine
        from stock_regime.src.models import MarketRegimeInput
        result = MarketRegimeEngine().analyze(BENCH)
        ctx    = MarketRegimeInput.from_dict(result.to_dict())
        assert ctx.regime     == result.to_dict()["regime"]
        assert ctx.confidence == pytest.approx(result.to_dict()["confidence"])

    def test_full_three_engine_pipeline(self):
        from market_regime.src import MarketRegimeEngine
        from stock_regime.src import StockRegimeEngine
        from stock_regime.src.models import MarketRegimeInput
        ctx     = MarketRegimeInput.from_dict(
            MarketRegimeEngine().analyze(BENCH).to_dict()
        )
        results = StockRegimeEngine(output_dir="/tmp/test_out").analyze_universe(
            STOCKS, ctx, BENCH, "TEST", persist=False,
        )
        assert len(results) == len(STOCKS)
        for r in results:
            assert 0.0 <= r.confidence <= 1.0

    def test_stable_result_serialises_to_json(self):
        import json
        from market_regime.src import MarketRegimeEngine
        from stock_regime.src import StockRegimeEngine
        from stock_regime.src.models import MarketRegimeInput
        from stock_regime.stability import RegimeStabiliser

        ctx     = MarketRegimeInput(regime="BULLISH_TREND", confidence=0.75)
        raw     = StockRegimeEngine(output_dir="/tmp/test_out").analyze_universe(
            STOCKS, ctx, BENCH, "TEST", persist=False,
        )
        history: dict = {}
        stable  = RegimeStabiliser(confirmation_bars=2).apply(raw, history)
        for r in stable:
            parsed = json.loads(json.dumps(r.to_dict()))
            assert parsed["symbol"] == r.symbol
            assert "stable_regime"  in parsed
            assert "regime_changed_today" in parsed


# ─────────────────────────────────────────────────────────────────────────────
#  Pipeline orchestrator (mocked DataManager)
# ─────────────────────────────────────────────────────────────────────────────

class TestPipelineOrchestrator:
    @pytest.fixture
    def symbol_files(self, tmp_path):
        uni = tmp_path / "data" / "universes"
        uni.mkdir(parents=True)
        (uni / "nifty500.txt").write_text("INFY.NS\n# comment\nRELIANCE.NS\nHDFCBANK.NS\n")
        return tmp_path

    @pytest.fixture
    def pipeline(self, tmp_path, symbol_files):
        cfg = f"""
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
        p = tmp_path / "pipeline.yaml"
        p.write_text(cfg)
        from runner.pipeline import AlgoTradingPipeline
        return AlgoTradingPipeline(config_path=p, project_root=symbol_files)

    def _mock(self, pipeline):
        from trading_data.models import FetchResult
        pipeline._data_manager.get_daily_data = lambda *a, **k: BENCH.copy()
        pipeline._data_manager.fetch_multiple_symbols = lambda syms, **k: {
            s: FetchResult(symbol=s, provider="yahoo",
                           data=_df(seed=i, n=300), success=True)
            for i, s in enumerate(syms)
        }

    def test_run_returns_correct_universe(self, pipeline):
        self._mock(pipeline)
        out = pipeline.run(universes=["NIFTY500"], persist=False)
        assert "NIFTY500" in out.market_results
        assert "NIFTY500" in out.stock_results

    def test_all_file_symbols_classified(self, pipeline):
        self._mock(pipeline)
        out = pipeline.run(universes=["NIFTY500"], persist=False)
        # 3 non-comment lines in nifty500.txt
        assert len(out.stock_results["NIFTY500"]) == 3

    def test_results_are_stable_regime_results(self, pipeline):
        self._mock(pipeline)
        out = pipeline.run(universes=["NIFTY500"], persist=False)
        from stock_regime.stability import StableRegimeResult
        for r in out.stock_results["NIFTY500"]:
            assert isinstance(r, StableRegimeResult)
            assert hasattr(r, "stable_regime")
            assert hasattr(r, "regime_changed_today")

    def test_max_symbols_override(self, pipeline):
        self._mock(pipeline)
        out = pipeline.run(universes=["NIFTY500"], persist=False, max_symbols=2)
        assert len(out.stock_results["NIFTY500"]) == 2

    def test_universe_detail_counts_populated(self, pipeline):
        self._mock(pipeline)
        out    = pipeline.run(universes=["NIFTY500"], persist=False)
        detail = out.universe_details["NIFTY500"]
        assert detail.symbols_loaded == 3
        assert detail.accepted_count >= 0
        assert detail.excluded_count >= 0
        assert detail.rejected_count >= 0

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

    def test_stable_classifications_parquet_written(self, pipeline, tmp_path):
        self._mock(pipeline)
        pipeline.run(universes=["NIFTY500"], persist=True)
        stables = list(Path(tmp_path / "output" / "stable_classifications").rglob("*.parquet"))
        assert len(stables) > 0
        df = pd.read_parquet(stables[0])
        assert "stable_regime" in df.columns
        assert "regime_changed_today" in df.columns

    def test_regime_history_parquet_written(self, pipeline, tmp_path):
        self._mock(pipeline)
        pipeline.run(universes=["NIFTY500"], persist=True)
        history_path = tmp_path / "output" / "regime_history" / "regime_history.parquet"
        assert history_path.exists()
        df = pd.read_parquet(history_path)
        assert "stable_regime" in df.columns

    def test_missing_symbol_file_raises(self, tmp_path):
        cfg = f"""
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
  root_dir: "{tmp_path}/output"
  persist: false
  log_dir: "{tmp_path}/logs"
  log_level: "WARNING"
market_regime_config: null
stock_regime_config:  null
"""
        p = tmp_path / "p.yaml"
        p.write_text(cfg)
        from runner.pipeline import AlgoTradingPipeline
        pipeline = AlgoTradingPipeline(config_path=p, project_root=tmp_path)
        pipeline._data_manager.get_daily_data = lambda *a, **k: BENCH.copy()
        with pytest.raises(FileNotFoundError):
            pipeline.run(universes=["NIFTY500"], persist=False)

    def test_unknown_universe_raises(self, pipeline):
        with pytest.raises(ValueError, match="Unknown universe"):
            pipeline.run(universes=["FAKE"], persist=False)

    def test_elapsed_time_populated(self, pipeline):
        self._mock(pipeline)
        out = pipeline.run(universes=["NIFTY500"], persist=False)
        assert out.elapsed_seconds > 0