"""
runner/tests/test_integration.py
==================================
Full integration tests covering all roadmap phases.
No network required — synthetic data throughout.

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
#  Shared helpers
# ─────────────────────────────────────────────────────────────────────────────

def _df(n=500, drift=0.001, vol=0.01, seed=0, start=18_000.0) -> pd.DataFrame:
    rng   = np.random.default_rng(seed)
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
#  Continuous Scoring (Improvement 1)
# ─────────────────────────────────────────────────────────────────────────────

class TestContinuousScoring:
    def _scorer(self):
        from stock_regime.src.config_loader import StockEngineConfig
        from stock_regime.src.scorer import StockRegimeScorer
        return StockRegimeScorer(StockEngineConfig())

    def test_continuous_scores_in_unit_interval(self):
        from stock_regime.src.models import StockIndicatorSnapshot
        snap = StockIndicatorSnapshot(
            close=19000, ema20=18800, ema50=18500, ema200=17000,
            ema20_slope=0.005, ema50_slope=0.003,
            adx=30, atr=200, atr_ma=150, volume=1.6e7, volume_ma=1e7,
            roc_10=2.5, roc_21=1.8, acceleration=0.7,
            rs_3m=1.06, rs_trend=0.002, ema_distance_pct=0.12,
        )
        scorer = self._scorer()
        cs = scorer._build_continuous_scores(snap)
        for field_name, val in cs.to_dict().items():
            assert 0.0 <= val <= 1.0, f"{field_name}={val} out of [0,1]"

    def test_trend_momentum_diverge(self):
        """Trend and momentum scores should differ meaningfully."""
        from stock_regime.src.models import StockIndicatorSnapshot, StockSignals
        scorer = self._scorer()
        # High RS trend + high ROC (momentum) but flat EMA + low ADX (weak trend)
        snap = StockIndicatorSnapshot(
            close=1000, ema20=1001, ema50=1002, ema200=1003,
            ema20_slope=0.00002, ema50_slope=0.00001,
            adx=12, atr=10, atr_ma=9, volume=2e7, volume_ma=1e7,
            roc_10=4.5, roc_21=1.0, acceleration=3.5,
            rs_3m=1.08, rs_trend=0.004,
        )
        sig = StockSignals()
        ds  = scorer.score_dimensions(sig, snap)
        # Momentum should be higher than trend for this setup
        assert ds.momentum > ds.trend, (
            f"Expected momentum ({ds.momentum}) > trend ({ds.trend}) "
            f"for accelerating RS stock with flat EMAs"
        )

    def test_no_binary_saturation(self):
        """With realistic inputs, trend score should not be exactly 1.0."""
        from stock_regime.src.models import StockIndicatorSnapshot, StockSignals
        scorer = self._scorer()
        snap = StockIndicatorSnapshot(
            close=19000, ema20=18900, ema50=18700, ema200=17000,
            adx=28, atr=180, atr_ma=150, volume=1.5e7, volume_ma=1e7,
            roc_10=1.5, rs_3m=1.04, ema_distance_pct=0.12,
        )
        sig = StockSignals()
        ds  = scorer.score_dimensions(sig, snap)
        assert ds.trend < 1.0, f"trend score {ds.trend} should be < 1.0 (no saturation)"
        assert ds.trend > 0.0, f"trend score {ds.trend} should be > 0.0"

    def test_continuous_scores_attached_to_dimensional(self):
        from stock_regime.src.models import StockIndicatorSnapshot, StockSignals
        scorer = self._scorer()
        snap   = StockIndicatorSnapshot(close=1000, adx=25, atr=10, atr_ma=9)
        sig    = StockSignals()
        ds     = scorer.score_dimensions(sig, snap)
        assert ds.continuous is not None
        assert hasattr(ds.continuous, "adx_score")


# ─────────────────────────────────────────────────────────────────────────────
#  New Indicators (ROC, RS multi-period, higher highs)
# ─────────────────────────────────────────────────────────────────────────────

class TestNewIndicators:
    def _calc(self):
        from stock_regime.src.config_loader import StockEngineConfig
        from stock_regime.src.indicators import StockIndicatorCalculator
        return StockIndicatorCalculator(StockEngineConfig())

    def test_roc_computed(self):
        snap = self._calc().compute(_df(n=500, seed=1))
        assert snap.roc_10  is not None, "roc_10 should be computed"
        assert snap.roc_21  is not None, "roc_21 should be computed"

    def test_acceleration_computed(self):
        snap = self._calc().compute(_df(n=500, seed=1))
        assert snap.acceleration is not None
        # acceleration = roc_10 - roc_21
        assert abs(snap.acceleration - (snap.roc_10 - snap.roc_21)) < 1e-6

    def test_ema_distance_pct_computed(self):
        snap = self._calc().compute(_df(n=500, seed=1))
        if snap.ema200 is not None and snap.ema200 > 0:
            expected = (snap.close - snap.ema200) / snap.ema200
            assert abs(snap.ema_distance_pct - expected) < 1e-4

    def test_higher_highs_count_computed(self):
        snap = self._calc().compute(_df(n=500, seed=1))
        assert snap.higher_highs_count is not None
        assert 0 <= snap.higher_highs_count <= 20  # within window

    def test_rs_profile_with_benchmark(self):
        snap = self._calc().compute(_df(n=500, seed=1), benchmark_df=BENCH)
        # rs_3m should be populated with a benchmark
        assert snap.rs_3m is not None
        assert 0.5 < snap.rs_3m < 2.0, f"rs_3m={snap.rs_3m} out of plausible range"

    def test_legacy_relative_strength_alias(self):
        """relative_strength must equal rs_3m for backward compatibility."""
        snap = self._calc().compute(_df(n=500, seed=1), benchmark_df=BENCH)
        assert snap.relative_strength == snap.rs_3m


# ─────────────────────────────────────────────────────────────────────────────
#  New Signals (roc, rs_improving, higher_highs)
# ─────────────────────────────────────────────────────────────────────────────

class TestNewSignals:
    def _extractor(self):
        from stock_regime.src.config_loader import StockEngineConfig
        from stock_regime.src.signals import StockSignalExtractor
        return StockSignalExtractor(StockEngineConfig())

    def _snap(self, **kw):
        from stock_regime.src.models import StockIndicatorSnapshot
        defaults = dict(
            close=19000, ema20=18800, ema50=18500, ema200=17000,
            ema20_slope=0.005, ema50_slope=0.003,
            adx=30, atr=200, atr_ma=150, volume=1.6e7, volume_ma=1e7,
        )
        defaults.update(kw)
        return StockIndicatorSnapshot(**defaults)

    def test_roc_positive_signal(self):
        sig = self._extractor().extract(self._snap(roc_10=2.5))
        assert sig.roc_positive is True

    def test_roc_negative_signal(self):
        sig = self._extractor().extract(self._snap(roc_10=-1.5))
        assert sig.roc_positive is False

    def test_roc_accelerating_signal(self):
        sig = self._extractor().extract(self._snap(roc_10=3.0, roc_21=1.0, acceleration=2.0))
        assert sig.roc_accelerating is True

    def test_rs_improving_signal(self):
        sig = self._extractor().extract(self._snap(rs_trend=0.002))
        assert sig.rs_improving is True
        assert sig.rs_weakening is False

    def test_rs_weakening_signal(self):
        sig = self._extractor().extract(self._snap(rs_trend=-0.003))
        assert sig.rs_weakening is True
        assert sig.rs_improving is False

    def test_higher_highs_signal(self):
        sig = self._extractor().extract(self._snap(higher_highs_count=8))
        assert sig.higher_highs is True

    def test_higher_highs_below_threshold(self):
        sig = self._extractor().extract(self._snap(higher_highs_count=2))
        assert sig.higher_highs is False

    def test_ema_extended_signal(self):
        sig = self._extractor().extract(self._snap(ema_distance_pct=0.15))
        assert sig.ema_extended is True

    def test_rs_uses_rs_3m_preferentially(self):
        """rs_3m should drive rs_positive, not legacy relative_strength."""
        snap = self._snap(rs_3m=1.08, relative_strength=0.90)
        sig  = self._extractor().extract(snap)
        assert sig.rs_positive is True   # driven by rs_3m=1.08, not 0.90


# ─────────────────────────────────────────────────────────────────────────────
#  Regime Stability — smoothing + hysteresis
# ─────────────────────────────────────────────────────────────────────────────

class TestRegimeStabilityEnhanced:
    def _make_result(self, regime_str, conf=0.75, symbol="SYM"):
        from stock_regime.src.models import (
            StockRegime, StockRegimeResult, DimensionalScores,
            StockSignals, StockIndicatorSnapshot,
        )
        return StockRegimeResult(
            symbol=symbol, market="TEST",
            stock_regime=StockRegime(regime_str), confidence=conf,
            dimensional_scores=DimensionalScores(), regime_scores={},
            signals=StockSignals(), indicators=StockIndicatorSnapshot(),
        )

    def test_smoothed_confidence_populated(self):
        from stock_regime.stability import RegimeStabiliser, SymbolHistory
        stab = RegimeStabiliser(confirmation_bars=1, smoothing_enabled=True,
                                smoothing_alpha=0.5)
        hist = {"SYM": SymbolHistory(symbol="SYM", stable_regime="TREND_UP",
                                     stable_age=3, smoothed_confidence=0.80)}
        results = stab.apply([self._make_result("TREND_UP", conf=0.70)], hist)
        # EWM: 0.5 * 0.70 + 0.5 * 0.80 = 0.75
        assert abs(results[0].smoothed_confidence - 0.75) < 0.01

    def test_hysteresis_prevents_weak_switch(self):
        """A weak new-regime signal should not overcome hysteresis."""
        from stock_regime.stability import RegimeStabiliser, SymbolHistory
        stab = RegimeStabiliser(
            confirmation_bars=1,
            hysteresis_threshold=0.10,
            regime_switch_threshold=0.40,
            smoothing_enabled=True, smoothing_alpha=0.5,
        )
        # Current stable = TREND_UP with high smoothed confidence
        hist = {"SYM": SymbolHistory(
            symbol="SYM",
            raw_regimes=["RANGE"],
            stable_regime="TREND_UP",
            stable_age=5,
            smoothed_confidence=0.80,
        )}
        # New raw regime = RANGE with low confidence — shouldn't switch
        results = stab.apply([self._make_result("RANGE", conf=0.55)], hist)
        assert results[0].stable_regime.value == "TREND_UP"
        assert results[0].regime_changed_today is False

    def test_oscillation_detected(self):
        """Stocks that flip regime every bar should trigger oscillation flag."""
        from stock_regime.stability import RegimeStabiliser, SymbolHistory
        stab = RegimeStabiliser(confirmation_bars=2, persistence_window=4)
        hist = {"SYM": SymbolHistory(
            symbol="SYM",
            raw_regimes=["TREND_UP", "RANGE", "TREND_UP", "RANGE"],
            stable_regime="TREND_UP",
            stable_age=1,
        )}
        results = stab.apply([self._make_result("RANGE", conf=0.70)], hist)
        assert results[0].oscillation_detected is True

    def test_smoothing_disabled_uses_raw_confidence(self):
        from stock_regime.stability import RegimeStabiliser, SymbolHistory
        stab = RegimeStabiliser(
            confirmation_bars=1, smoothing_enabled=False,
            regime_switch_threshold=0.40,
        )
        hist = {"SYM": SymbolHistory(symbol="SYM", stable_regime="RANGE",
                                     stable_age=2, smoothed_confidence=0.80)}
        results = stab.apply([self._make_result("TREND_UP", conf=0.70)], hist)
        assert abs(results[0].smoothed_confidence - 0.70) < 0.01


# ─────────────────────────────────────────────────────────────────────────────
#  Opportunity Quality Engine (Improvement 7)
# ─────────────────────────────────────────────────────────────────────────────

class TestOpportunityQualityEngine:
    def _make_stable_result(self, symbol="SYM", regime="TREND_UP",
                            conf=0.78, smth=0.76, age=8, oscillating=False):
        from stock_regime.stability.stabiliser import StableRegimeResult
        from stock_regime.src.models import (
            StockRegime, StockRegimeResult, DimensionalScores,
            StockSignals, StockIndicatorSnapshot, ContinuousScores,
        )
        raw = StockRegimeResult(
            symbol=symbol, market="TEST",
            stock_regime=StockRegime(regime), confidence=conf,
            dimensional_scores=DimensionalScores(
                trend=0.72, momentum=0.65, volatility=0.40,
                continuous=ContinuousScores(
                    adx_score=0.60, ema_alignment_score=0.80,
                    ema_distance_score=0.60, atr_expansion_score=0.50,
                    rs_score=0.65, rs_trend_score=0.70,
                    roc_score=0.65, volume_score=0.55,
                ),
            ),
            regime_scores={}, signals=StockSignals(),
            indicators=StockIndicatorSnapshot(
                close=1000, ema200=900, atr=15, atr_ma=14,
                volume=1.5e7, volume_ma=1e7,
                ema_distance_pct=0.11, higher_highs_count=7, rs_3m=1.05,
            ),
        )
        sr = StableRegimeResult.from_result(
            raw,
            stable_regime=StockRegime(regime),
            prior_stable_regime=StockRegime.RANGE,
            regime_age_bars=age,
            stable_regime_age=age,
            regime_changed_today=False,
            smoothed_confidence=smth,
            oscillation_detected=oscillating,
        )
        return sr

    def test_quality_score_in_unit_interval(self):
        from stock_regime.quality_engine import OpportunityQualityEngine
        engine = OpportunityQualityEngine()
        r      = self._make_stable_result()
        scores = engine.evaluate_batch([r], {r.symbol: _df(n=300, seed=1)})
        assert len(scores) == 1
        assert 0.0 <= scores[0].quality_score <= 1.0

    def test_oscillating_stock_penalised(self):
        from stock_regime.quality_engine import OpportunityQualityEngine
        engine = OpportunityQualityEngine()
        stable = self._make_stable_result(oscillating=False)
        osc    = self._make_stable_result(symbol="OSC", oscillating=True)
        scores = engine.evaluate_batch([stable, osc], {
            stable.symbol: _df(n=300, seed=1),
            osc.symbol:    _df(n=300, seed=2),
        })
        s_score = next(s for s in scores if s.symbol == stable.symbol)
        o_score = next(s for s in scores if s.symbol == "OSC")
        assert s_score.stability_quality > o_score.stability_quality

    def test_uncertain_regime_zero_stability(self):
        from stock_regime.quality_engine import OpportunityQualityEngine
        engine = OpportunityQualityEngine()
        r      = self._make_stable_result(regime="UNCERTAIN", conf=0.0, smth=0.0)
        scores = engine.evaluate_batch([r], {r.symbol: _df(n=300, seed=1)})
        assert scores[0].stability_quality == 0.0

    def test_quality_output_serialises_to_dict(self):
        from stock_regime.quality_engine import OpportunityQualityEngine
        engine = OpportunityQualityEngine()
        r      = self._make_stable_result()
        scores = engine.evaluate_batch([r], {r.symbol: _df(n=300, seed=1)})
        d = scores[0].to_dict()
        for key in ["symbol", "quality_score", "trend_quality",
                    "liquidity_quality", "stability_quality"]:
            assert key in d

    def test_persist_creates_parquet(self, tmp_path):
        from stock_regime.quality_engine import OpportunityQualityEngine
        engine = OpportunityQualityEngine()
        r      = self._make_stable_result()
        scores = engine.evaluate_batch([r], {r.symbol: _df(n=300, seed=1)})
        path   = engine.persist(scores, tmp_path, universe="TEST")
        assert path is not None and path.exists()
        df = pd.read_parquet(path)
        assert "quality_score" in df.columns


# ─────────────────────────────────────────────────────────────────────────────
#  Strict History Validation (Improvement 5)
# ─────────────────────────────────────────────────────────────────────────────

class TestStrictHistoryValidation:
    def test_short_history_rejected(self):
        from stock_regime.filters.history_filter import HistoryFilter
        f = HistoryFilter(min_bars=300)
        # Only 250 bars — should be rejected
        reasons = f.check("SYM", _df(n=250))
        assert any(r.check == "min_bars" for r in reasons)

    def test_300_bars_accepted(self):
        from stock_regime.filters.history_filter import HistoryFilter
        f = HistoryFilter(min_bars=300)
        reasons = f.check("SYM", _df(n=320))
        assert reasons == []

    def test_engine_warns_on_insufficient_bars(self):
        """Engine should warn when fewer than recommended bars are supplied."""
        import warnings
        from stock_regime.src.config_loader import StockEngineConfig
        from stock_regime.src.indicators import StockIndicatorCalculator
        calc = StockIndicatorCalculator(StockEngineConfig())
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            calc.compute(_df(n=50))  # far too few
            assert len(w) > 0
            assert "rows" in str(w[0].message).lower()


# ─────────────────────────────────────────────────────────────────────────────
#  Regime Analytics (Improvement 8)
# ─────────────────────────────────────────────────────────────────────────────

class TestRegimeAnalytics:
    def _history_df(self, n_symbols=5, n_days=100) -> pd.DataFrame:
        import random
        random.seed(42)
        regimes = ["TREND_UP", "TREND_DOWN", "RANGE", "MOMENTUM"]
        rows    = []
        dates   = pd.bdate_range(end=pd.Timestamp.today().normalize(), periods=n_days)
        for i in range(n_symbols):
            sym    = f"SYM{i}"
            stable = random.choice(regimes)
            age    = 0
            for j, d in enumerate(dates):
                changed = (j % 12 == 0 and j > 0)
                if changed:
                    stable = random.choice(regimes)
                    age    = 1
                else:
                    age   += 1
                rows.append({
                    "symbol":               sym,
                    "market":               "TEST",
                    "run_date":             d.date(),
                    "raw_regime":           stable,
                    "stable_regime":        stable,
                    "prior_stable_regime":  stable,
                    "confidence":           0.75,
                    "smoothed_confidence":  0.74,
                    "regime_age_bars":      age,
                    "stable_regime_age":    age,
                    "regime_changed_today": changed,
                    "oscillation_detected": False,
                })
        return pd.DataFrame(rows)

    def test_analytics_produces_report(self, tmp_path):
        from stock_regime.analytics import RegimeAnalytics
        p = tmp_path / "history.parquet"
        self._history_df().to_parquet(p, index=False)
        rpt = RegimeAnalytics(p).compute("TEST")
        assert rpt.universe == "TEST"
        assert len(rpt.current_episodes) > 0

    def test_transition_matrix_rows_sum_to_one(self, tmp_path):
        from stock_regime.analytics import RegimeAnalytics
        p = tmp_path / "history.parquet"
        self._history_df(n_symbols=15, n_days=200).to_parquet(p, index=False)
        rpt = RegimeAnalytics(p).compute("TEST")
        if rpt.transition_matrix:
            row_sums = rpt.transition_matrix.matrix.sum(axis=1)
            active   = row_sums[row_sums > 0]
            assert (active - 1.0).abs().max() < 0.01

    def test_analytics_persist_files(self, tmp_path):
        from stock_regime.analytics import RegimeAnalytics
        p = tmp_path / "history.parquet"
        self._history_df().to_parquet(p, index=False)
        rpt   = RegimeAnalytics(p, min_episode_bars=2).compute("TEST")
        saved = RegimeAnalytics(p).persist(rpt, tmp_path)
        assert len(saved) > 0
        for fp in saved.values():
            assert fp.exists()


# ─────────────────────────────────────────────────────────────────────────────
#  Validation plotting
# ─────────────────────────────────────────────────────────────────────────────

class TestValidationPlotting:
    def _make_stable(self, symbol="INFY.NS"):
        from stock_regime.stability.stabiliser import StableRegimeResult
        from stock_regime.src.models import (
            StockRegime, StockRegimeResult, DimensionalScores,
            StockSignals, StockIndicatorSnapshot, ContinuousScores,
        )
        raw = StockRegimeResult(
            symbol=symbol, market="TEST",
            stock_regime=StockRegime.TREND_UP, confidence=0.78,
            dimensional_scores=DimensionalScores(
                trend=0.72, momentum=0.65, volatility=0.40,
                continuous=ContinuousScores(
                    adx_score=0.60, ema_alignment_score=0.80,
                    rs_score=0.65, roc_score=0.65,
                ),
            ),
            regime_scores={}, signals=StockSignals(),
            indicators=StockIndicatorSnapshot(
                close=1820, ema20=1785, ema50=1710, ema200=1602,
                adx=28, atr=42, atr_ma=45, volume=1.5e7, volume_ma=1.2e7,
            ),
        )
        return StableRegimeResult.from_result(
            raw,
            stable_regime=StockRegime.TREND_UP,
            prior_stable_regime=StockRegime.RANGE,
            regime_age_bars=12, stable_regime_age=12,
            regime_changed_today=False,
            smoothed_confidence=0.76, oscillation_detected=False,
        )

    def test_plot_regime_chart_returns_figure(self):
        pytest.importorskip("matplotlib")
        from stock_regime.validation.plotting import RegimePlotter
        plotter = RegimePlotter()
        fig     = plotter.plot_regime_chart("INFY.NS", _df(n=300, seed=1), self._make_stable())
        import matplotlib.pyplot as plt
        assert fig is not None
        plt.close(fig)

    def test_plot_score_distribution(self):
        pytest.importorskip("matplotlib")
        from stock_regime.validation.plotting import RegimePlotter
        import matplotlib.pyplot as plt
        plotter  = RegimePlotter()
        score_df = pd.DataFrame({
            "trend":    np.random.uniform(0.2, 0.9, 50),
            "momentum": np.random.uniform(0.1, 0.8, 50),
        })
        fig = plotter.plot_score_distribution(score_df, "trend", "TEST")
        assert fig is not None
        plt.close(fig)

    def test_save_all_creates_png(self, tmp_path):
        pytest.importorskip("matplotlib")
        from stock_regime.validation.plotting import RegimePlotter
        plotter = RegimePlotter()
        saved   = plotter.save_all("INFY.NS", _df(n=300, seed=1),
                                   self._make_stable(), str(tmp_path))
        assert len(saved) == 1
        assert Path(saved[0]).exists()


# ─────────────────────────────────────────────────────────────────────────────
#  Pipeline orchestration (mocked DataManager)
# ─────────────────────────────────────────────────────────────────────────────

class TestPipelineOrchestration:
    @pytest.fixture
    def symbol_files(self, tmp_path):
        uni = tmp_path / "data" / "universes"
        uni.mkdir(parents=True)
        (uni / "nifty500.txt").write_text(
            "INFY.NS\n# comment\nRELIANCE.NS\nHDFCBANK.NS\n"
        )
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
                           data=_df(seed=i, n=350), success=True)
            for i, s in enumerate(syms)
        }

    def test_results_are_stable_regime_results(self, pipeline):
        self._mock(pipeline)
        out = pipeline.run(universes=["NIFTY500"], persist=False)
        from stock_regime.stability import StableRegimeResult
        for r in out.stock_results["NIFTY500"]:
            assert isinstance(r, StableRegimeResult)
            assert hasattr(r, "smoothed_confidence")
            assert hasattr(r, "oscillation_detected")

    def test_quality_scores_produced(self, pipeline):
        self._mock(pipeline)
        out = pipeline.run(universes=["NIFTY500"], persist=False)
        qs  = out.quality_scores.get("NIFTY500", [])
        assert len(qs) > 0
        for q in qs:
            assert 0.0 <= q.quality_score <= 1.0

    def test_continuous_scores_in_results(self, pipeline):
        self._mock(pipeline)
        out = pipeline.run(universes=["NIFTY500"], persist=False)
        for r in out.stock_results["NIFTY500"]:
            ds = r.dimensional_scores
            assert 0.0 <= ds.trend      <= 1.0
            assert 0.0 <= ds.momentum   <= 1.0
            assert 0.0 <= ds.volatility <= 1.0

    def test_scoring_parquet_written(self, pipeline, tmp_path):
        self._mock(pipeline)
        pipeline.run(universes=["NIFTY500"], persist=True)
        scoring_files = list(Path(tmp_path / "output" / "scoring").rglob("*.parquet"))
        assert len(scoring_files) > 0
        df = pd.read_parquet(scoring_files[0])
        assert "trend_score" in df.columns
        assert "momentum_score" in df.columns

    def test_stable_parquet_has_continuous_scores(self, pipeline, tmp_path):
        self._mock(pipeline)
        pipeline.run(universes=["NIFTY500"], persist=True)
        stable_files = list(
            Path(tmp_path / "output" / "stable_classifications").rglob("*.parquet")
        )
        assert len(stable_files) > 0
        df = pd.read_parquet(stable_files[0])
        assert "smoothed_confidence"  in df.columns
        assert "oscillation_detected" in df.columns

    def test_max_symbols_respected(self, pipeline):
        self._mock(pipeline)
        out = pipeline.run(universes=["NIFTY500"], persist=False, max_symbols=2)
        assert len(out.stock_results["NIFTY500"]) == 2

    def test_elapsed_time_populated(self, pipeline):
        self._mock(pipeline)
        out = pipeline.run(universes=["NIFTY500"], persist=False)
        assert out.elapsed_seconds > 0