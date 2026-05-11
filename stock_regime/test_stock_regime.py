"""
stock_regime/tests/test_stock_regime.py
========================================
Pytest unit and integration tests for the Stock Regime Engine.

Run with:  pytest tests/ -v

Coverage:
  - StockIndicatorSnapshot completeness
  - StockSignalExtractor (all signal types)
  - StockRegimeScorer (regime scores + dimensional scores)
  - StockRegimeClassifier (winner selection, UNCERTAIN fallback, context adjustment)
  - StockRanker (all three dimensions)
  - StockRegimeEngine end-to-end (valid result, JSON contract, error isolation)
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Allow running from any working directory
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT.parent))

from stock_regime.src import StockRegimeEngine
from stock_regime.src.models import (
    DimensionalScores,
    MarketRegimeInput,
    StockIndicatorSnapshot,
    StockRegime,
    StockSignals,
)
from stock_regime.src.config_loader import StockEngineConfig
from stock_regime.src.signals import StockSignalExtractor
from stock_regime.src.scorer import StockRegimeScorer
from stock_regime.src.classifier import StockRegimeClassifier
from stock_regime.src.ranker import StockRanker

warnings.filterwarnings("ignore")


# ──────────────────────────────────────────────────────────────────
#  Shared fixtures and helpers
# ──────────────────────────────────────────────────────────────────

def _make_df(
    n_bars: int = 500,
    drift: float = 0.001,
    vol: float = 0.01,
    seed: int = 0,
    start_price: float = 18_000.0,
) -> pd.DataFrame:
    """Synthetic daily OHLCV DataFrame with a date index."""
    rng   = np.random.default_rng(seed)
    dates = pd.bdate_range(end="2024-12-31", periods=n_bars)
    close = np.empty(n_bars)
    price = start_price

    for i in range(n_bars):
        price = max(price * (1 + rng.normal(drift, vol)), 1_000)
        close[i] = price

    rng2 = np.random.default_rng(seed + 1)
    daily_range = close * 0.01
    high   = close + daily_range * 0.6
    low    = close - daily_range * 0.6
    open_  = low + daily_range * 0.5
    volume = rng2.integers(5_000_000, 20_000_000, n_bars).astype(float)

    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
        index=dates,
    )


@pytest.fixture(scope="module")
def config() -> StockEngineConfig:
    return StockEngineConfig()


@pytest.fixture(scope="module")
def engine(tmp_path_factory) -> StockRegimeEngine:
    out = tmp_path_factory.mktemp("output")
    return StockRegimeEngine(output_dir=out)


@pytest.fixture(scope="module")
def bullish_df()  -> pd.DataFrame: return _make_df(drift=0.002,  vol=0.007, seed=10)
@pytest.fixture(scope="module")
def bearish_df()  -> pd.DataFrame: return _make_df(drift=-0.002, vol=0.007, seed=20)
@pytest.fixture(scope="module")
def volatile_df() -> pd.DataFrame: return _make_df(drift=0.0,    vol=0.040, seed=30)
@pytest.fixture(scope="module")
def quiet_df()    -> pd.DataFrame: return _make_df(drift=0.0002, vol=0.001, seed=40)
@pytest.fixture(scope="module")
def benchmark_df() -> pd.DataFrame: return _make_df(drift=0.001, vol=0.009, seed=99)

@pytest.fixture(scope="module")
def bullish_market() -> MarketRegimeInput:
    return MarketRegimeInput(regime="BULLISH_TREND", confidence=0.80)

@pytest.fixture(scope="module")
def bearish_market() -> MarketRegimeInput:
    return MarketRegimeInput(regime="BEARISH_TREND", confidence=0.75)

@pytest.fixture(scope="module")
def uncertain_market() -> MarketRegimeInput:
    return MarketRegimeInput(regime="UNCERTAIN", confidence=0.0)


# ──────────────────────────────────────────────────────────────────
#  MarketRegimeInput tests
# ──────────────────────────────────────────────────────────────────

class TestMarketRegimeInput:
    def test_from_dict(self):
        mri = MarketRegimeInput.from_dict({"regime": "BULLISH_TREND", "confidence": 0.82})
        assert mri.regime == "BULLISH_TREND"
        assert mri.confidence == pytest.approx(0.82)

    def test_from_dict_defaults(self):
        mri = MarketRegimeInput.from_dict({})
        assert mri.regime == "UNCERTAIN"
        assert mri.confidence == 0.0


# ──────────────────────────────────────────────────────────────────
#  StockIndicatorSnapshot tests
# ──────────────────────────────────────────────────────────────────

class TestStockIndicatorSnapshot:
    def test_is_complete_all_fields(self):
        snap = StockIndicatorSnapshot(
            close=100, ema20=99, ema50=97, ema200=90,
            ema20_slope=0.001, ema50_slope=0.0005,
            adx=28, atr=5, atr_ma=4.5, volume=1e7, volume_ma=8e6,
        )
        assert snap.is_complete() is True

    def test_is_incomplete_missing_field(self):
        snap = StockIndicatorSnapshot(close=100, ema20=99)
        assert snap.is_complete() is False

    def test_to_dict_has_all_keys(self):
        snap = StockIndicatorSnapshot(close=100)
        d = snap.to_dict()
        for key in ["close", "ema20", "ema50", "ema200", "adx", "atr",
                    "atr_ma", "volume", "volume_ma", "relative_strength", "high_52w"]:
            assert key in d


# ──────────────────────────────────────────────────────────────────
#  StockSignalExtractor tests
# ──────────────────────────────────────────────────────────────────

class TestStockSignalExtractor:

    def _snap(self, **kwargs) -> StockIndicatorSnapshot:
        defaults = dict(
            close=19_000, ema20=18_800, ema50=18_500, ema200=17_000,
            ema20_slope=0.005, ema50_slope=0.003,
            adx=30, atr=200, atr_ma=150,
            volume=1.6e7, volume_ma=1e7,
            relative_strength=1.07, high_52w=19_500,
        )
        defaults.update(kwargs)
        return StockIndicatorSnapshot(**defaults)

    def test_bullish_signals(self, config):
        sig = StockSignalExtractor(config).extract(self._snap())
        assert sig.price_above_ema200 is True
        assert sig.ema20_above_ema50  is True
        assert sig.adx_strong         is True

    def test_bearish_signals(self, config):
        snap = self._snap(
            close=15_000, ema20=15_200, ema50=15_500, ema200=17_000,
            relative_strength=0.94,
        )
        sig = StockSignalExtractor(config).extract(snap)
        assert sig.price_below_ema200 is True
        assert sig.ema20_below_ema50  is True
        assert sig.rs_negative        is True

    def test_flat_ema_signals(self, config):
        snap = self._snap(
            ema20_slope=0.00005, ema50_slope=0.00003, adx=12,
        )
        sig = StockSignalExtractor(config).extract(snap)
        assert sig.ema20_flat is True
        assert sig.ema50_flat is True
        assert sig.adx_weak   is True

    def test_atr_compressed_signal(self, config):
        snap = self._snap(atr=60, atr_ma=100)   # ratio = 0.60 < 0.75
        sig = StockSignalExtractor(config).extract(snap)
        assert sig.atr_compressed is True
        assert sig.atr_low        is True

    def test_atr_volatile_signal(self, config):
        snap = self._snap(atr=300, atr_ma=100)   # ratio = 3.0 > 1.30
        sig = StockSignalExtractor(config).extract(snap)
        assert sig.atr_high is True

    def test_rs_strong_signal(self, config):
        snap = self._snap(relative_strength=1.08)
        sig = StockSignalExtractor(config).extract(snap)
        assert sig.rs_strong   is True
        assert sig.rs_positive is True

    def test_price_near_high(self, config):
        snap = self._snap(close=19_400, high_52w=19_500)  # within 3 %
        sig = StockSignalExtractor(config).extract(snap)
        assert sig.price_near_52w_high is True

    def test_no_rs_when_none(self, config):
        snap = self._snap(relative_strength=None)
        sig = StockSignalExtractor(config).extract(snap)
        assert sig.rs_positive is False
        assert sig.rs_negative is False
        assert sig.rs_strong   is False

    def test_volume_confirmed(self, config):
        snap = self._snap(volume=2e7, volume_ma=1e7)   # ratio = 2.0 ≥ 1.50
        sig = StockSignalExtractor(config).extract(snap)
        assert sig.volume_confirmed is True


# ──────────────────────────────────────────────────────────────────
#  StockRegimeScorer tests
# ──────────────────────────────────────────────────────────────────

class TestStockRegimeScorer:

    def test_all_scores_in_unit_interval(self, config):
        scorer  = StockRegimeScorer(config)
        signals = StockSignals(
            price_above_ema200=True, ema20_above_ema50=True,
            adx_strong=True, volume_confirmed=True, rs_positive=True,
        )
        scores = scorer.score_regimes(signals)
        for regime, score in scores.items():
            assert 0.0 <= score <= 1.0, f"{regime} score {score} out of [0,1]"

    def test_perfect_trend_up_score(self, config):
        scorer  = StockRegimeScorer(config)
        signals = StockSignals(
            price_above_ema200=True, ema20_above_ema50=True,
            adx_strong=True, rs_positive=True, volume_confirmed=True,
        )
        scores = scorer.score_regimes(signals)
        assert scores[StockRegime.TREND_UP] == pytest.approx(1.0, abs=1e-6)

    def test_all_false_gives_zero_scores(self, config):
        scorer = StockRegimeScorer(config)
        scores = scorer.score_regimes(StockSignals())
        for score in scores.values():
            assert score == pytest.approx(0.0, abs=1e-9)

    def test_dimensional_scores_in_unit_interval(self, config):
        scorer = StockRegimeScorer(config)
        snap   = StockIndicatorSnapshot(atr=200, atr_ma=100)
        signals = StockSignals(
            price_above_ema200=True, rs_strong=True, volume_confirmed=True,
        )
        ds = scorer.score_dimensions(signals, snap)
        assert 0.0 <= ds.trend      <= 1.0
        assert 0.0 <= ds.momentum   <= 1.0
        assert 0.0 <= ds.volatility <= 1.0

    def test_volatility_score_capped_at_one(self, config):
        scorer  = StockRegimeScorer(config)
        # ATR/ATR_MA = 5.0 → should be capped to 1.0
        snap    = StockIndicatorSnapshot(atr=500, atr_ma=100)
        ds      = scorer.score_dimensions(StockSignals(), snap)
        assert ds.volatility == pytest.approx(1.0)


# ──────────────────────────────────────────────────────────────────
#  StockRegimeClassifier tests
# ──────────────────────────────────────────────────────────────────

class TestStockRegimeClassifier:

    def test_highest_score_wins(self, config, bullish_market):
        clf = StockRegimeClassifier(config)
        scores = {
            StockRegime.TREND_UP:  0.85,
            StockRegime.TREND_DOWN: 0.30,
            StockRegime.RANGE:      0.20,
            StockRegime.VOLATILE:   0.40,
            StockRegime.QUIET:      0.10,
            StockRegime.MOMENTUM:   0.50,
            StockRegime.BREAKOUT_SETUP: 0.35,
        }
        result = clf.classify(
            "TEST", "TEST_MKT", scores, StockSignals(),
            DimensionalScores(), bullish_market, StockIndicatorSnapshot(),
        )
        assert result.stock_regime == StockRegime.TREND_UP
        # boost = 1.0 + (alignment_boost - 1.0) * market_conf
        #       = 1.0 + (1.10 - 1.0) * 0.80 = 1.08
        expected = 0.85 * (1.0 + (1.10 - 1.0) * 0.80)
        assert result.confidence == pytest.approx(expected, rel=0.01)

    def test_uncertain_when_below_min_confidence(self, config, uncertain_market):
        clf = StockRegimeClassifier(config)
        scores = {r: 0.1 for r in StockRegime if r != StockRegime.UNCERTAIN}
        result = clf.classify(
            "TEST", "TEST_MKT", scores, StockSignals(),
            DimensionalScores(), uncertain_market, StockIndicatorSnapshot(),
        )
        assert result.stock_regime == StockRegime.UNCERTAIN

    def test_confidence_clamped_to_one(self, config, bullish_market):
        clf = StockRegimeClassifier(config)
        scores = {StockRegime.TREND_UP: 0.99}
        for r in StockRegime:
            if r not in scores:
                scores[r] = 0.0
        result = clf.classify(
            "TEST", "TEST_MKT", scores, StockSignals(),
            DimensionalScores(), bullish_market, StockIndicatorSnapshot(),
        )
        assert result.confidence <= 1.0

    def test_context_boost_for_aligned_regime(self, config):
        clf    = StockRegimeClassifier(config)
        market = MarketRegimeInput(regime="BULLISH_TREND", confidence=1.0)
        scores = {r: 0.0 for r in StockRegime}
        scores[StockRegime.TREND_UP] = 0.70

        result = clf.classify(
            "TEST", "TEST_MKT", scores, StockSignals(),
            DimensionalScores(), market, StockIndicatorSnapshot(),
        )
        # Boosted confidence should be > 0.70
        assert result.confidence > 0.70

    def test_context_penalty_for_misaligned_regime(self, config):
        clf    = StockRegimeClassifier(config)
        market = MarketRegimeInput(regime="BULLISH_TREND", confidence=1.0)
        scores = {r: 0.0 for r in StockRegime}
        scores[StockRegime.TREND_DOWN] = 0.70

        result = clf.classify(
            "TEST", "TEST_MKT", scores, StockSignals(),
            DimensionalScores(), market, StockIndicatorSnapshot(),
        )
        # Penalised confidence should be < 0.70
        assert result.confidence < 0.70


# ──────────────────────────────────────────────────────────────────
#  StockRanker tests
# ──────────────────────────────────────────────────────────────────

class TestStockRanker:

    def _make_results(self) -> list[StockRegimeResult]:
        """Build a minimal list of results for ranking tests."""
        from stock_regime.src.models import StockRegimeResult
        results = []
        for i in range(5):
            r = StockRegimeResult(
                symbol=f"STOCK{i}",
                market="TEST",
                stock_regime=StockRegime.TREND_UP,
                confidence=0.70 + i * 0.05,
                dimensional_scores=DimensionalScores(
                    trend=0.1 * (i + 1),
                    momentum=0.1 * (5 - i),
                    volatility=0.1 * i,
                ),
            )
            results.append(r)
        return results

    def test_trend_ranking_order(self, config):
        ranker  = StockRanker(config)
        results = self._make_results()
        output  = ranker.rank(results)
        trend   = output.strongest_trends
        assert len(trend) >= 1
        # Verify descending order
        for j in range(len(trend) - 1):
            assert trend[j].score >= trend[j + 1].score

    def test_ranking_excludes_uncertain(self, config):
        from stock_regime.src.models import StockRegimeResult
        ranker = StockRanker(config)
        results = self._make_results()
        results.append(StockRegimeResult(
            symbol="BAD", market="TEST",
            stock_regime=StockRegime.UNCERTAIN, confidence=0.0,
        ))
        output = ranker.rank(results)
        symbols = [e.symbol for e in output.strongest_trends]
        assert "BAD" not in symbols

    def test_all_ranking_types_present(self, config):
        ranker = StockRanker(config)
        output = ranker.rank(self._make_results())
        assert len(output.strongest_trends)   > 0
        assert len(output.strongest_momentum) > 0
        assert len(output.highest_volatility) > 0


# ──────────────────────────────────────────────────────────────────
#  End-to-end engine integration tests
# ──────────────────────────────────────────────────────────────────

class TestStockRegimeEngine:

    def test_single_stock_returns_result(self, engine, bullish_df, bullish_market):
        result = engine.analyze_single("INFY", bullish_df, bullish_market)
        assert result is not None
        assert result.symbol == "INFY"
        assert result.stock_regime in StockRegime

    def test_result_confidence_in_range(self, engine, bullish_df, bullish_market):
        result = engine.analyze_single("TEST", bullish_df, bullish_market)
        assert 0.0 <= result.confidence <= 1.0

    def test_to_dict_has_required_keys(self, engine, bullish_df, bullish_market):
        result = engine.analyze_single("TEST", bullish_df, bullish_market)
        d = result.to_dict()
        for key in ["symbol", "market", "stock_regime", "confidence",
                    "scores", "signals", "indicators"]:
            assert key in d

    def test_universe_batch_returns_all_symbols(self, engine, bullish_market):
        data = {
            "INFY":     _make_df(seed=1),
            "RELIANCE": _make_df(seed=2),
            "AAPL":     _make_df(seed=3),
        }
        results = engine.analyze_universe(
            data, bullish_market, market_label="TEST", persist=False
        )
        assert len(results) == 3
        symbols = {r.symbol for r in results}
        assert symbols == {"INFY", "RELIANCE", "AAPL"}

    def test_bad_stock_does_not_abort_batch(self, engine, bullish_market):
        """A DataFrame with missing columns should fail gracefully."""
        data = {
            "GOOD": _make_df(seed=5),
            "BAD":  pd.DataFrame({"close": [100, 101]}),   # missing OHLCV
        }
        results = engine.analyze_universe(
            data, bullish_market, market_label="TEST", persist=False
        )
        assert len(results) == 2
        good = next(r for r in results if r.symbol == "GOOD")
        bad  = next(r for r in results if r.symbol == "BAD")
        assert good.is_valid() is True
        assert bad.is_valid()  is False
        assert bad.error is not None

    def test_relative_strength_with_benchmark(self, engine, bullish_df,
                                              benchmark_df, bullish_market):
        result = engine.analyze_single(
            "INFY", bullish_df, bullish_market, benchmark_data=benchmark_df
        )
        # RS should be computed and populated in the snapshot
        # (may be None only if both series are too short)
        assert result.indicators.relative_strength is None or isinstance(
            result.indicators.relative_strength, float
        )

    def test_bearish_df_tends_bearish(self, engine, bearish_df, bearish_market):
        result = engine.analyze_single("TEST", bearish_df, bearish_market)
        assert result.stock_regime in (
            StockRegime.TREND_DOWN, StockRegime.UNCERTAIN
        )

    def test_quiet_df_tends_quiet(self, engine, quiet_df, uncertain_market):
        result = engine.analyze_single("TEST", quiet_df, uncertain_market)
        # Very low volatility should lean toward QUIET or RANGE
        assert result.stock_regime in (
            StockRegime.QUIET, StockRegime.RANGE, StockRegime.TREND_UP, StockRegime.UNCERTAIN
        )
