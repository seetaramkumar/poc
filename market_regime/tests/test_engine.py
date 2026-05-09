"""
tests/test_engine.py — Unit tests for the Market Regime Engine
==============================================================
Run with:  pytest tests/ -v
"""

from __future__ import annotations

import warnings
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from src import MarketRegimeEngine, MarketRegime  # noqa: E402
from src.models import IndicatorSnapshot, RegimeSignals  # noqa: E402
from src.signals import SignalExtractor  # noqa: E402
from src.scorer import RegimeScorer  # noqa: E402
from src.classifier import RegimeClassifier  # noqa: E402
from src.config_loader import EngineConfig  # noqa: E402

warnings.filterwarnings("ignore")


# ──────────────────────────────────────────────────────────────────
#  Fixtures
# ──────────────────────────────────────────────────────────────────

def _make_df(
    n_bars: int = 500,
    drift: float = 0.001,
    vol: float = 0.01,
    seed: int = 0,
) -> pd.DataFrame:
    """Synthetic daily OHLCV DataFrame."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(end="2024-12-31", periods=n_bars)
    close = np.empty(n_bars)
    price = 18_000.0
    for i in range(n_bars):
        price = max(price * (1 + rng.normal(drift, vol)), 1_000)
        close[i] = price
    daily_range = close * 0.01
    high   = close + daily_range * 0.6
    low    = close - daily_range * 0.6
    open_  = low + daily_range * 0.5
    volume = rng.integers(10_000_000, 20_000_000, n_bars).astype(float)
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
        index=dates,
    )


@pytest.fixture(scope="module")
def config() -> EngineConfig:
    return EngineConfig()


@pytest.fixture(scope="module")
def engine() -> MarketRegimeEngine:
    return MarketRegimeEngine()


@pytest.fixture(scope="module")
def bullish_df() -> pd.DataFrame:
    """Strong uptrend data."""
    return _make_df(drift=0.002, vol=0.008, seed=1)


@pytest.fixture(scope="module")
def bearish_df() -> pd.DataFrame:
    """Strong downtrend data."""
    return _make_df(drift=-0.002, vol=0.008, seed=2)


@pytest.fixture(scope="module")
def volatile_df() -> pd.DataFrame:
    """High-volatility data."""
    return _make_df(drift=0.0, vol=0.04, seed=3)


@pytest.fixture(scope="module")
def quiet_df() -> pd.DataFrame:
    """Low-volatility data."""
    return _make_df(drift=0.0003, vol=0.001, seed=4)


# ──────────────────────────────────────────────────────────────────
#  IndicatorSnapshot tests
# ──────────────────────────────────────────────────────────────────

class TestIndicatorSnapshot:
    def test_is_complete_when_all_fields_set(self):
        snap = IndicatorSnapshot(
            close=100,
            ema20=99, ema50=98, ema200=95,
            ema20_slope=0.001, ema50_slope=0.0005,
            adx=28, atr=50, atr_ma=45,
            volume=1e7, volume_ma=8e6,
        )
        assert snap.is_complete() is True

    def test_incomplete_when_field_missing(self):
        snap = IndicatorSnapshot(close=100, ema20=99)
        assert snap.is_complete() is False


# ──────────────────────────────────────────────────────────────────
#  SignalExtractor tests
# ──────────────────────────────────────────────────────────────────

class TestSignalExtractor:
    def test_bullish_signals(self, config):
        extractor = SignalExtractor(config)
        snap = IndicatorSnapshot(
            close=19_000,
            ema20=18_800, ema50=18_500, ema200=17_000,
            ema20_slope=0.005, ema50_slope=0.003,
            adx=35, atr=100, atr_ma=90,
            volume=1.5e7, volume_ma=1e7,
        )
        sig = extractor.extract(snap)
        assert sig.price_above_ema200 is True
        assert sig.ema20_above_ema50  is True
        assert sig.adx_strong         is True

    def test_bearish_signals(self, config):
        extractor = SignalExtractor(config)
        snap = IndicatorSnapshot(
            close=15_000,
            ema20=15_200, ema50=15_500, ema200=17_000,
            ema20_slope=-0.005, ema50_slope=-0.003,
            adx=30, atr=150, atr_ma=100,
            volume=1e7, volume_ma=8e6,
        )
        sig = extractor.extract(snap)
        assert sig.price_below_ema200 is True
        assert sig.ema20_below_ema50  is True

    def test_flat_ema_signals(self, config):
        extractor = SignalExtractor(config)
        snap = IndicatorSnapshot(
            close=18_000,
            ema20=18_050, ema50=18_100, ema200=17_500,
            ema20_slope=0.00005, ema50_slope=0.00003,  # very flat
            adx=12, atr=80, atr_ma=82,
            volume=8e6, volume_ma=9e6,
        )
        sig = extractor.extract(snap)
        assert sig.ema20_flat is True
        assert sig.ema50_flat is True
        assert sig.adx_weak   is True

    def test_volatile_signal(self, config):
        extractor = SignalExtractor(config)
        snap = IndicatorSnapshot(
            close=18_000,
            ema20=17_900, ema50=17_800, ema200=17_000,
            ema20_slope=0.001, ema50_slope=0.001,
            adx=20, atr=300, atr_ma=100,   # ratio = 3.0 > volatile_ratio
            volume=9e6, volume_ma=9e6,
        )
        sig = extractor.extract(snap)
        assert sig.atr_high is True

    def test_quiet_signal(self, config):
        extractor = SignalExtractor(config)
        snap = IndicatorSnapshot(
            close=18_000,
            ema20=17_950, ema50=17_900, ema200=17_000,
            ema20_slope=0.0001, ema50_slope=0.0001,
            adx=10, atr=40, atr_ma=100,    # ratio = 0.4 < quiet_ratio
            volume=9e6, volume_ma=9e6,
        )
        sig = extractor.extract(snap)
        assert sig.atr_low is True


# ──────────────────────────────────────────────────────────────────
#  RegimeScorer tests
# ──────────────────────────────────────────────────────────────────

class TestRegimeScorer:
    def test_scores_sum_lte_one_per_regime(self, config):
        scorer  = RegimeScorer(config)
        signals = RegimeSignals(
            price_above_ema200=True, ema20_above_ema50=True,
            adx_strong=True, volume_confirms=True,
        )
        scores = scorer.score_all(signals)
        for regime, score in scores.items():
            assert 0.0 <= score <= 1.0, f"{regime} score {score} out of [0,1]"

    def test_perfect_bullish_signals_score_1(self, config):
        scorer  = RegimeScorer(config)
        signals = RegimeSignals(
            price_above_ema200=True, ema20_above_ema50=True,
            adx_strong=True, volume_confirms=True,
        )
        scores = scorer.score_all(signals)
        assert scores[MarketRegime.BULLISH_TREND] == pytest.approx(1.0, abs=1e-6)

    def test_all_false_signals_score_zero(self, config):
        scorer  = RegimeScorer(config)
        signals = RegimeSignals()
        scores  = scorer.score_all(signals)
        for score in scores.values():
            assert score == pytest.approx(0.0, abs=1e-9)


# ──────────────────────────────────────────────────────────────────
#  RegimeClassifier tests
# ──────────────────────────────────────────────────────────────────

class TestRegimeClassifier:
    def test_wins_highest_score(self, config):
        clf = RegimeClassifier(config)
        scores = {
            MarketRegime.BULLISH_TREND: 0.85,
            MarketRegime.BEARISH_TREND: 0.30,
            MarketRegime.SIDEWAYS:      0.20,
            MarketRegime.VOLATILE:      0.40,
            MarketRegime.QUIET:         0.10,
        }
        result = clf.classify(scores, RegimeSignals())
        assert result.regime == MarketRegime.BULLISH_TREND
        assert result.confidence == pytest.approx(0.85)

    def test_uncertain_when_below_min_confidence(self, config):
        clf = RegimeClassifier(config)
        scores = {r: 0.1 for r in MarketRegime if r != MarketRegime.UNCERTAIN}
        result = clf.classify(scores, RegimeSignals())
        assert result.regime == MarketRegime.UNCERTAIN


# ──────────────────────────────────────────────────────────────────
#  End-to-end engine tests
# ──────────────────────────────────────────────────────────────────

class TestMarketRegimeEngine:
    def test_analyze_returns_result(self, engine, bullish_df):
        result = engine.analyze(bullish_df)
        assert result is not None
        assert result.regime in MarketRegime

    def test_analyze_to_json_is_valid(self, engine, bullish_df):
        import json
        raw = engine.analyze_to_json(bullish_df)
        data = json.loads(raw)
        assert "regime"     in data
        assert "confidence" in data
        assert "signals"    in data

    def test_confidence_in_range(self, engine, bullish_df):
        result = engine.analyze(bullish_df)
        assert 0.0 <= result.confidence <= 1.0

    def test_bullish_df_tends_bullish(self, engine, bullish_df):
        """A persistent uptrend should lean bullish on the last bar."""
        result = engine.analyze(bullish_df)
        assert result.regime in (MarketRegime.BULLISH_TREND, MarketRegime.UNCERTAIN)

    def test_bearish_df_tends_bearish(self, engine, bearish_df):
        result = engine.analyze(bearish_df)
        assert result.regime in (MarketRegime.BEARISH_TREND, MarketRegime.UNCERTAIN)

    def test_rolling_length_matches_input(self, engine, bullish_df):
        results = engine.analyze_rolling(bullish_df)
        assert len(results) == len(bullish_df)

    def test_missing_columns_raises(self, engine):
        bad_df = pd.DataFrame({"close": [100, 101, 102]})
        with pytest.raises(ValueError, match="missing columns"):
            engine.analyze(bad_df)

    def test_to_dict_has_required_keys(self, engine, bullish_df):
        result = engine.analyze(bullish_df)
        d = result.to_dict()
        assert set(d.keys()) >= {"regime", "confidence", "signals", "scores"}
