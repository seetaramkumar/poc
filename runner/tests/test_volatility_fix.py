"""
runner/tests/test_volatility_fix.py
=====================================
Phase 3 — Volatility classification fix.

All generators are self-contained (no shared state / fixtures that depend on
file system paths). Tests use the same path insertions as the rest of the suite.

Run:
    cd <project_root>
    pytest runner/tests/test_volatility_fix.py -v
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Adjust this to your project root
ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))
warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────────────────────────────────────
#  Data generators
# ─────────────────────────────────────────────────────────────────────────────

def _make_trend(n: int = 450, drift: float = 0.0015,
                vol: float = 0.012, seed: int = 1) -> pd.DataFrame:
    """
    Clean smooth uptrend with realistic OHLC variation.
    Consistent direction → low dir_cv → composite ≈ 0.10–0.25.
    """
    rng   = np.random.default_rng(seed)
    dates = pd.bdate_range(
        end=pd.Timestamp.today() + pd.Timedelta(days=2), periods=n + 2
    )[-n:]
    close = np.empty(n)
    p     = 1000.0
    for i in range(n):
        p = max(p * (1 + rng.normal(drift, vol)), 10.0)
        close[i] = p

    rng2  = np.random.default_rng(seed + 100)
    dv    = close * vol * 1.5
    high  = close + dv * rng2.uniform(0.2, 1.0, n)
    low   = close - dv * rng2.uniform(0.2, 1.0, n)
    open_ = np.empty(n)
    open_[0] = close[0]
    for i in range(1, n):
        open_[i] = close[i - 1] * (1 + rng2.normal(0, 0.004))
    open_ = np.clip(open_, low, high)
    vol_  = rng2.integers(3_000_000, 15_000_000, n).astype(float)
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": vol_},
        index=dates,
    )


def _make_erratic(n: int = 450, seed: int = 9) -> pd.DataFrame:
    """
    Genuinely erratic stock: direction alternates every 1–3 bars,
    large opening gaps, high per-bar vol.
    Random direction → high dir_cv → composite ≈ 0.45–0.80.
    """
    rng   = np.random.default_rng(seed)
    dates = pd.bdate_range(
        end=pd.Timestamp.today() + pd.Timedelta(days=2), periods=n + 2
    )[-n:]
    close = np.empty(n)
    p     = 1000.0
    direction = 1
    hold      = int(rng.integers(1, 4))
    for i in range(n):
        if hold <= 0:
            direction = -direction
            hold = int(rng.integers(1, 4))
        p = max(p * (1 + direction * abs(rng.normal(0.015, 0.020))), 10.0)
        close[i] = p
        hold -= 1

    rng2  = np.random.default_rng(seed + 100)
    dv    = close * 0.030
    high  = close + dv * rng2.uniform(0.3, 1.5, n)
    low   = close - dv * rng2.uniform(0.3, 1.5, n)
    open_ = np.empty(n)
    open_[0] = close[0]
    for i in range(1, n):
        open_[i] = close[i - 1] * (1 + rng2.normal(0, 0.018))  # large gaps
    open_ = np.clip(open_, low, high)
    vol_  = rng2.integers(3_000_000, 15_000_000, n).astype(float)
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": vol_},
        index=dates,
    )


def _get_calc():
    from stock_regime.src.config_loader import StockEngineConfig
    from stock_regime.src.indicators import StockIndicatorCalculator
    return StockIndicatorCalculator(StockEngineConfig())


def _get_extractor():
    from stock_regime.src.config_loader import StockEngineConfig
    from stock_regime.src.signals import StockSignalExtractor
    return StockSignalExtractor(StockEngineConfig())


def _get_scorer():
    from stock_regime.src.config_loader import StockEngineConfig
    from stock_regime.src.scorer import StockRegimeScorer
    return StockRegimeScorer(StockEngineConfig())


# ─────────────────────────────────────────────────────────────────────────────
#  1. Indicator computation
# ─────────────────────────────────────────────────────────────────────────────

class TestVolatilityIndicators:

    def test_composite_score_computed_on_trend(self):
        snap = _get_calc().compute(_make_trend())
        assert snap.volatility_instability_score is not None

    def test_composite_score_in_unit_interval(self):
        for label, df in [("trend", _make_trend()), ("erratic", _make_erratic())]:
            snap  = _get_calc().compute(df)
            score = snap.volatility_instability_score
            assert score is not None, f"{label}: composite is None"
            assert 0.0 <= score <= 1.0, f"{label}: {score} out of [0,1]"

    def test_erratic_composite_higher_than_trend(self):
        """Erratic composite MUST be higher than trend composite."""
        snap_t = _get_calc().compute(_make_trend())
        snap_e = _get_calc().compute(_make_erratic())
        assert snap_e.volatility_instability_score > snap_t.volatility_instability_score, (
            f"erratic={snap_e.volatility_instability_score:.4f} "
            f"should > trend={snap_t.volatility_instability_score:.4f}"
        )

    def test_trend_composite_below_threshold(self):
        snap  = _get_calc().compute(_make_trend())
        score = snap.volatility_instability_score
        assert score < 0.40, (
            f"Clean trend score {score:.4f} should be < 0.40 (threshold)"
        )

    def test_erratic_composite_above_threshold(self):
        snap  = _get_calc().compute(_make_erratic())
        score = snap.volatility_instability_score
        assert score > 0.40, (
            f"Erratic score {score:.4f} should be > 0.40 (threshold)"
        )

    def test_dir_cv_discriminates(self):
        """dir_cv (candle_instability) must be higher on erratic than trend."""
        snap_t = _get_calc().compute(_make_trend())
        snap_e = _get_calc().compute(_make_erratic())
        assert snap_e.candle_instability > snap_t.candle_instability, (
            f"erratic dir_cv={snap_e.candle_instability:.4f} "
            f"should > trend={snap_t.candle_instability:.4f}"
        )

    def test_backward_compat_fields_populated(self):
        snap = _get_calc().compute(_make_trend())
        assert snap.candle_instability   is not None
        assert snap.reversal_frequency   is not None
        assert snap.wickiness_score      is not None
        assert snap.gap_frequency        is not None


# ─────────────────────────────────────────────────────────────────────────────
#  2. Signal extraction
# ─────────────────────────────────────────────────────────────────────────────

class TestVolatilitySignals:

    def test_trend_does_not_trigger_volatile_instability(self):
        snap = _get_calc().compute(_make_trend())
        sig  = _get_extractor().extract(snap)
        assert sig.volatile_instability is False, (
            f"Strong trend should not be volatile "
            f"(score={snap.volatility_instability_score:.4f})"
        )

    def test_erratic_triggers_volatile_instability(self):
        snap = _get_calc().compute(_make_erratic())
        sig  = _get_extractor().extract(snap)
        assert sig.volatile_instability is True, (
            f"Erratic stock should be volatile "
            f"(score={snap.volatility_instability_score:.4f})"
        )

    def test_anti_volatile_guard_suppresses_borderline(self):
        """
        Borderline composite (0.42) on a stock with strong structural signals
        must be suppressed by the anti-volatile guard.
        """
        from stock_regime.src.models import StockIndicatorSnapshot
        ext  = _get_extractor()
        snap = StockIndicatorSnapshot(
            close=1500.0, ema20=1480.0, ema50=1420.0, ema200=1300.0,
            adx=28.0, atr=15.0, atr_ma=13.0,
            volume=8_000_000.0, volume_ma=7_000_000.0,
            volatility_instability_score=0.42,   # just above 0.40 threshold
            candle_instability=0.55,
        )
        sig = ext.extract(snap)
        assert sig.price_above_ema200 is True
        assert sig.ema20_above_ema50  is True
        assert sig.adx_strong         is True
        assert sig.volatile_instability is False, (
            "Anti-volatile guard should suppress borderline (0.42) on strong trend"
        )

    def test_anti_volatile_guard_does_not_suppress_severe(self):
        """
        Score ≥ 0.55 must NOT be suppressed even on structurally trending stocks.
        """
        from stock_regime.src.models import StockIndicatorSnapshot
        ext  = _get_extractor()
        snap = StockIndicatorSnapshot(
            close=1500.0, ema20=1480.0, ema50=1420.0, ema200=1300.0,
            adx=28.0, atr=15.0, atr_ma=13.0,
            volume=8_000_000.0, volume_ma=7_000_000.0,
            volatility_instability_score=0.62,   # well above 0.55 ceiling
            candle_instability=0.70,
        )
        sig = ext.extract(snap)
        assert sig.volatile_instability is True, (
            "Score 0.62 should not be suppressed (above severe_threshold=0.55)"
        )

    def test_fallback_path_two_of_four(self):
        """When composite is None, ≥2 of 4 individual signals trigger instability."""
        from stock_regime.src.models import StockIndicatorSnapshot
        ext  = _get_extractor()
        snap = StockIndicatorSnapshot(
            close=500.0, adx=20.0, atr=10.0, atr_ma=9.0,
            volume=5_000_000.0, volume_ma=4_000_000.0,
            volatility_instability_score=None,   # no composite
            candle_instability=0.60,             # > 0.50 ✓
            reversal_frequency=0.60,             # > 0.55 ✓
            gap_frequency=0.10,                  # < 0.15 ✗
            wickiness_score=0.30,                # < 0.45 ✗
        )
        sig = ext.extract(snap)
        assert sig.volatile_instability is True, \
            "Fallback: 2 of 4 signals should trigger volatile_instability"

    def test_fallback_one_signal_not_volatile(self):
        """Only 1 individual signal → not volatile via fallback."""
        from stock_regime.src.models import StockIndicatorSnapshot
        ext  = _get_extractor()
        snap = StockIndicatorSnapshot(
            close=500.0, adx=20.0, atr=10.0, atr_ma=9.0,
            volatility_instability_score=None,
            candle_instability=0.60,   # > threshold ✓
            reversal_frequency=0.40,   # < threshold ✗
            gap_frequency=0.05,        # < threshold ✗
            wickiness_score=0.30,      # < threshold ✗
        )
        sig = ext.extract(snap)
        assert sig.volatile_instability is False

    def test_primary_path_overrides_fallback(self):
        """When composite is set, individual signal count does NOT matter."""
        from stock_regime.src.models import StockIndicatorSnapshot
        ext  = _get_extractor()
        # All 4 individual signals fire, but composite is below threshold
        snap = StockIndicatorSnapshot(
            close=500.0, adx=15.0,
            volatility_instability_score=0.25,   # below 0.40 — NOT volatile
            candle_instability=0.80,             # would trigger fallback
            reversal_frequency=0.70,
            gap_frequency=0.30,
            wickiness_score=0.60,
        )
        sig = ext.extract(snap)
        assert sig.volatile_instability is False, (
            "Composite=0.25 should win over individual signals via primary path"
        )


# ─────────────────────────────────────────────────────────────────────────────
#  3. Regime scoring
# ─────────────────────────────────────────────────────────────────────────────

class TestVolatileRegimeScoring:

    def test_trend_up_beats_volatile_on_clean_trend(self):
        from stock_regime.src.models import StockRegime
        calc   = _get_calc()
        ext    = _get_extractor()
        scorer = _get_scorer()
        snap   = calc.compute(_make_trend())
        sig    = ext.extract(snap)
        scores = scorer.score_regimes(sig)
        assert scores[StockRegime.TREND_UP] > scores[StockRegime.VOLATILE], (
            f"TREND_UP={scores[StockRegime.TREND_UP]:.3f} should beat "
            f"VOLATILE={scores[StockRegime.VOLATILE]:.3f} on a clean trend"
        )

    def test_volatile_score_zero_with_no_instability_signals(self):
        from stock_regime.src.models import StockSignals, StockRegime
        signals = StockSignals(
            volatile_instability=False, atr_high=False, high_reversal_freq=False,
            price_above_ema200=True, ema20_above_ema50=True, adx_strong=True,
        )
        scores = _get_scorer().score_regimes(signals)
        assert scores[StockRegime.VOLATILE] == 0.0, (
            f"No instability → VOLATILE must be 0, got {scores[StockRegime.VOLATILE]}"
        )

    def test_volatile_score_above_half_with_all_instability(self):
        from stock_regime.src.models import StockSignals, StockRegime
        signals = StockSignals(
            volatile_instability=True, atr_high=True, high_reversal_freq=True,
        )
        scores = _get_scorer().score_regimes(signals)
        assert scores[StockRegime.VOLATILE] >= 0.50, (
            f"All instability signals → VOLATILE should be >= 0.50, "
            f"got {scores[StockRegime.VOLATILE]:.3f}"
        )

    def test_instability_score_read_from_composite(self):
        """
        ContinuousScores.instability_score must equal the snapshot's
        volatility_instability_score (no re-computation in scorer).
        """
        from stock_regime.src.models import StockIndicatorSnapshot, StockSignals
        scorer = _get_scorer()
        snap   = StockIndicatorSnapshot(
            volatility_instability_score=0.63,
            close=1000.0, atr=10.0, atr_ma=9.0,
        )
        ds = scorer.score_dimensions(StockSignals(), snap)
        assert ds.continuous is not None
        assert abs(ds.continuous.instability_score - 0.63) < 0.001, (
            f"instability_score should be 0.63, got {ds.continuous.instability_score}"
        )

    def test_volatile_regime_does_not_win_for_trend(self):
        """
        End-to-end: the winning regime for a clean uptrend must not be VOLATILE.
        Uses the classifier + full signal pipeline.
        """
        from stock_regime.src.models import StockRegime, MarketRegimeInput
        from stock_regime.src.config_loader import StockEngineConfig
        from stock_regime.src.indicators import StockIndicatorCalculator
        from stock_regime.src.signals import StockSignalExtractor
        from stock_regime.src.scorer import StockRegimeScorer
        from stock_regime.src.classifier import StockRegimeClassifier

        cfg     = StockEngineConfig()
        calc    = StockIndicatorCalculator(cfg)
        ext     = StockSignalExtractor(cfg)
        scorer  = StockRegimeScorer(cfg)
        clf     = StockRegimeClassifier(cfg)
        market  = MarketRegimeInput(regime="BULLISH_TREND", confidence=0.80)

        snap    = calc.compute(_make_trend())
        sig     = ext.extract(snap)
        r_scores= scorer.score_regimes(sig)
        dim     = scorer.score_dimensions(sig, snap)
        result  = clf.classify(
            symbol="TREND_STOCK", market="TEST",
            regime_scores=r_scores, signals=sig,
            dimensional_scores=dim, market_regime=market, snap=snap,
            scorer=scorer,
        )
        assert result.stock_regime != StockRegime.VOLATILE, (
            f"Clean trend classified as VOLATILE — "
            f"vi_score={snap.volatility_instability_score:.4f} "
            f"vi_signal={sig.volatile_instability}"
        )

    def test_erratic_does_not_win_as_trend_up(self):
        """
        End-to-end: a genuinely erratic stock must NOT be classified TREND_UP or MOMENTUM.
        """
        from stock_regime.src.models import StockRegime, MarketRegimeInput
        from stock_regime.src.config_loader import StockEngineConfig
        from stock_regime.src.indicators import StockIndicatorCalculator
        from stock_regime.src.signals import StockSignalExtractor
        from stock_regime.src.scorer import StockRegimeScorer
        from stock_regime.src.classifier import StockRegimeClassifier

        cfg     = StockEngineConfig()
        calc    = StockIndicatorCalculator(cfg)
        ext     = StockSignalExtractor(cfg)
        scorer  = StockRegimeScorer(cfg)
        clf     = StockRegimeClassifier(cfg)
        market  = MarketRegimeInput(regime="BULLISH_TREND", confidence=0.80)

        snap    = calc.compute(_make_erratic())
        sig     = ext.extract(snap)
        r_scores= scorer.score_regimes(sig)
        dim     = scorer.score_dimensions(sig, snap)
        result  = clf.classify(
            symbol="ERRATIC_STOCK", market="TEST",
            regime_scores=r_scores, signals=sig,
            dimensional_scores=dim, market_regime=market, snap=snap,
            scorer=scorer,
        )
        assert result.stock_regime not in (StockRegime.TREND_UP, StockRegime.MOMENTUM), (
            f"Erratic stock classified as {result.stock_regime.value} — "
            f"vi_score={snap.volatility_instability_score:.4f}"
        )


# ─────────────────────────────────────────────────────────────────────────────
#  4. Regression — Phase 2 signals must still work
# ─────────────────────────────────────────────────────────────────────────────

class TestRegressionPhase2:

    def test_range_signals_unaffected(self):
        from stock_regime.src.models import StockIndicatorSnapshot
        ext  = _get_extractor()
        snap = StockIndicatorSnapshot(
            close=1000.0, ema20=1001.0, ema50=1002.0,
            adx=12.0, atr=10.0, atr_ma=11.0,
            bb_width=0.02, directional_efficiency=0.20, ema_spread=0.001,
        )
        sig = ext.extract(snap)
        assert sig.range_bound    is True
        assert sig.bb_compressed  is True
        assert sig.ema_compressed is True

    def test_rs_signals_use_rs3m(self):
        from stock_regime.src.models import StockIndicatorSnapshot
        snap = StockIndicatorSnapshot(rs_3m=1.08, relative_strength=0.90)
        sig  = _get_extractor().extract(snap)
        assert sig.rs_positive is True   # driven by rs_3m=1.08

    def test_roc_signals_work(self):
        from stock_regime.src.models import StockIndicatorSnapshot
        snap = StockIndicatorSnapshot(roc_10=3.5, acceleration=1.2)
        sig  = _get_extractor().extract(snap)
        assert sig.roc_positive     is True
        assert sig.roc_accelerating is True