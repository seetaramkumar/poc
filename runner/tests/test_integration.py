"""
runner/tests/test_integration.py
==================================
Full integration tests — all phases including breadth, sector, router.
No network required.
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
def _df(n=500, drift=0.001, vol=0.01, seed=0, start=18_000.0):
    rng   = np.random.default_rng(seed)
    dates = pd.bdate_range(end=pd.Timestamp.today().normalize() + pd.Timedelta(days=2), periods=n + 2)[-n:]
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
# Phase 1 — Volatility Fix
# ─────────────────────────────────────────────────────────────────────────────
class TestVolatilityFix:
    def _calc(self):
        from stock_regime.src.config_loader import StockEngineConfig
        from stock_regime.src.indicators import StockIndicatorCalculator
        return StockIndicatorCalculator(StockEngineConfig())

    def _scorer(self):
        from stock_regime.src.config_loader import StockEngineConfig
        from stock_regime.src.scorer import StockRegimeScorer
        return StockRegimeScorer(StockEngineConfig())

    def test_instability_indicators_computed(self):
        snap = self._calc().compute(_df(n=500, seed=1))
        assert snap.candle_instability is not None
        assert snap.reversal_frequency is not None
        assert snap.gap_frequency      is not None
        assert snap.wickiness_score    is not None

    def test_strong_trend_not_volatile(self):
        """A clean strong uptrend should NOT get VOLATILE classification."""
        from stock_regime.src.models import StockSignals, StockIndicatorSnapshot
        from stock_regime.src.models import StockRegime
        scorer = self._scorer()
        # Clean uptrend: high ADX, expanding ATR, but LOW instability signals
        signals = StockSignals(
            price_above_ema200=True, ema20_above_ema50=True,
            adx_strong=True, atr_high=True, atr_expanding=True,
            rs_positive=True, volume_confirmed=True,
            # Instability signals all False — clean trend
            volatile_instability=False, candle_erratic=False, high_reversal_freq=False,
        )
        scores = scorer.score_regimes(signals)
        assert scores[StockRegime.VOLATILE] < scores[StockRegime.TREND_UP], (
            f"Strong trend should NOT score higher as VOLATILE: "
            f"VOLATILE={scores[StockRegime.VOLATILE]:.2f} > "
            f"TREND_UP={scores[StockRegime.TREND_UP]:.2f}"
        )

    def test_erratic_stock_is_volatile(self):
        """A stock with erratic behavior should score highest as VOLATILE."""
        from stock_regime.src.models import StockSignals, StockRegime
        scorer = self._scorer()
        signals = StockSignals(
            atr_high=True,
            volatile_instability=True, candle_erratic=True, high_reversal_freq=True,
            adx_weak=True,
        )
        scores = scorer.score_regimes(signals)
        assert scores[StockRegime.VOLATILE] > 0.50, (
            f"Erratic stock should score VOLATILE > 0.50: got {scores[StockRegime.VOLATILE]:.2f}"
        )

    def test_instability_score_in_unit_interval(self):
        from stock_regime.src.models import StockIndicatorSnapshot
        scorer = self._scorer()
        snap   = StockIndicatorSnapshot(
            close=1000, atr=15, atr_ma=14,
            candle_instability=1.5, reversal_frequency=0.6,
            gap_frequency=0.2, wickiness_score=0.6,
        )
        cs = scorer._build_continuous_scores(snap)
        assert 0.0 <= cs.instability_score <= 1.0

    def test_volatile_signals_in_snapshot(self):
        from stock_regime.src.config_loader import StockEngineConfig
        from stock_regime.src.signals import StockSignalExtractor
        from stock_regime.src.models import StockIndicatorSnapshot
        ext  = StockSignalExtractor(StockEngineConfig())
        snap = StockIndicatorSnapshot(
            close=1000, atr=15, atr_ma=14,
            candle_instability=1.5, reversal_frequency=0.6,
            gap_frequency=0.2, wickiness_score=0.7,
        )
        sig = ext.extract(snap)
        assert sig.volatile_instability is True
        assert sig.candle_erratic       is True
        assert sig.high_reversal_freq   is True


# ─────────────────────────────────────────────────────────────────────────────
# Phase 2 — Range Detection
# ─────────────────────────────────────────────────────────────────────────────
class TestRangeDetection:
    def _calc(self):
        from stock_regime.src.config_loader import StockEngineConfig
        from stock_regime.src.indicators import StockIndicatorCalculator
        return StockIndicatorCalculator(StockEngineConfig())

    def test_range_indicators_computed(self):
        snap = self._calc().compute(_df(n=500, seed=1))
        assert snap.bb_width               is not None
        assert snap.directional_efficiency is not None
        assert snap.ema_spread             is not None

    def test_directional_efficiency_in_unit_interval(self):
        snap = self._calc().compute(_df(n=500, seed=1))
        if snap.directional_efficiency is not None:
            assert 0.0 <= snap.directional_efficiency <= 1.0

    def test_range_signals_fired_on_sideways_stock(self):
        from stock_regime.src.config_loader import StockEngineConfig
        from stock_regime.src.signals import StockSignalExtractor
        from stock_regime.src.models import StockIndicatorSnapshot
        ext  = StockSignalExtractor(StockEngineConfig())
        snap = StockIndicatorSnapshot(
            close=1000, ema20=1001, ema50=1002,
            adx=12, atr=10, atr_ma=11,
            bb_width=0.02,               # compressed
            directional_efficiency=0.20, # ranging
            ema_spread=0.001,            # compressed
        )
        sig = ext.extract(snap)
        assert sig.range_bound    is True
        assert sig.bb_compressed  is True
        assert sig.ema_compressed is True

    def test_range_score_higher_on_sideways(self):
        from stock_regime.src.models import StockSignals, StockRegime
        from stock_regime.src.config_loader import StockEngineConfig
        from stock_regime.src.scorer import StockRegimeScorer
        scorer = StockRegimeScorer(StockEngineConfig())
        # All range signals True
        signals = StockSignals(
            adx_weak=True, ema20_flat=True, ema50_flat=True, atr_low=True,
            range_bound=True, bb_compressed=True, ema_compressed=True,
        )
        scores = scorer.score_regimes(signals)
        assert scores[StockRegime.RANGE] > 0.60, (
            f"RANGE score should be > 0.60 for sideways stock, got {scores[StockRegime.RANGE]:.2f}"
        )

    def test_ranging_score_in_unit_interval(self):
        from stock_regime.src.models import StockIndicatorSnapshot
        from stock_regime.src.config_loader import StockEngineConfig
        from stock_regime.src.scorer import StockRegimeScorer
        scorer = StockRegimeScorer(StockEngineConfig())
        snap   = StockIndicatorSnapshot(
            close=1000, bb_width=0.025,
            directional_efficiency=0.25, ema_spread=0.002,
        )
        cs = scorer._build_continuous_scores(snap)
        assert 0.0 <= cs.ranging_score <= 1.0


# ─────────────────────────────────────────────────────────────────────────────
# Phase 3 — Market-Context-Aware Scoring
# ─────────────────────────────────────────────────────────────────────────────
class TestMarketContextScoring:
    def _scorer(self):
        from stock_regime.src.config_loader import StockEngineConfig
        from stock_regime.src.scorer import StockRegimeScorer
        return StockRegimeScorer(StockEngineConfig())

    def test_bearish_market_reduces_bullish_score(self):
        from stock_regime.src.models import StockSignals, StockRegime, MarketRegimeInput
        scorer  = self._scorer()
        signals = StockSignals(price_above_ema200=True, ema20_above_ema50=True,
                               adx_strong=True, rs_positive=True)
        raw_scores     = scorer.score_regimes(signals)
        bearish_market = MarketRegimeInput(regime="BEARISH_TREND", confidence=1.0)
        adj_scores     = scorer.apply_market_context(raw_scores, bearish_market)
        assert adj_scores[StockRegime.TREND_UP] < raw_scores[StockRegime.TREND_UP], (
            "Bearish market should reduce TREND_UP score"
        )
        assert adj_scores[StockRegime.TREND_DOWN] > raw_scores[StockRegime.TREND_DOWN], (
            "Bearish market should increase TREND_DOWN score"
        )

    def test_bullish_market_boosts_trend_up(self):
        from stock_regime.src.models import StockSignals, StockRegime, MarketRegimeInput
        scorer  = self._scorer()
        signals = StockSignals(price_above_ema200=True, ema20_above_ema50=True, adx_strong=True)
        raw    = scorer.score_regimes(signals)
        adj    = scorer.apply_market_context(raw, MarketRegimeInput("BULLISH_TREND", 1.0))
        assert adj[StockRegime.TREND_UP] >= raw[StockRegime.TREND_UP]

    def test_sideways_market_favours_range(self):
        from stock_regime.src.models import StockSignals, StockRegime, MarketRegimeInput
        scorer  = self._scorer()
        signals = StockSignals(adx_weak=True, ema20_flat=True, ema50_flat=True, range_bound=True)
        raw    = scorer.score_regimes(signals)
        adj    = scorer.apply_market_context(raw, MarketRegimeInput("SIDEWAYS", 0.80))
        assert adj[StockRegime.RANGE] >= raw[StockRegime.RANGE]

    def test_context_adjustment_bounded(self):
        from stock_regime.src.models import StockSignals, StockRegime, MarketRegimeInput
        scorer  = self._scorer()
        signals = StockSignals(price_above_ema200=True, adx_strong=True, rs_positive=True)
        raw    = scorer.score_regimes(signals)
        adj    = scorer.apply_market_context(raw, MarketRegimeInput("BEARISH_TREND", 1.0))
        for regime, score in adj.items():
            assert 0.0 <= score <= 1.0, f"{regime}: adj score {score} out of [0,1]"

    def test_uncertain_market_minimal_adjustment(self):
        from stock_regime.src.models import StockSignals, StockRegime, MarketRegimeInput
        scorer  = self._scorer()
        signals = StockSignals(price_above_ema200=True, adx_strong=True)
        raw    = scorer.score_regimes(signals)
        adj    = scorer.apply_market_context(raw, MarketRegimeInput("UNCERTAIN", 0.0))
        for r in StockRegime:
            if r in raw and r in adj:
                assert abs(adj[r] - raw[r]) < 0.01, "UNCERTAIN market should not adjust scores"


# ─────────────────────────────────────────────────────────────────────────────
# Phase 4 — Breadth Engine
# ─────────────────────────────────────────────────────────────────────────────
class TestBreadthEngine:
    def _make_stable_results(self, regimes: list[str]):
        from stock_regime.stability.stabiliser import StableRegimeResult
        from stock_regime.src.models import (
            StockRegime, StockRegimeResult, DimensionalScores,
            StockSignals, StockIndicatorSnapshot,
        )
        results = []
        for i, regime in enumerate(regimes):
            is_above = regime in ("TREND_UP", "MOMENTUM")
            signals  = StockSignals(price_above_ema200=is_above)
            raw = StockRegimeResult(
                symbol=f"S{i}", market="TEST",
                stock_regime=StockRegime(regime), confidence=0.70,
                dimensional_scores=DimensionalScores(trend=0.60, momentum=0.50),
                regime_scores={}, signals=signals,
                indicators=StockIndicatorSnapshot(),
            )
            sr = StableRegimeResult.from_result(
                raw, stable_regime=StockRegime(regime),
                prior_stable_regime=StockRegime.UNCERTAIN,
                regime_age_bars=5, stable_regime_age=5,
                regime_changed_today=False,
                smoothed_confidence=0.70, oscillation_detected=False,
            )
            results.append(sr)
        return results

    def test_breadth_snapshot_produced(self):
        from stock_regime.breadth_engine import BreadthEngine
        engine  = BreadthEngine()
        results = self._make_stable_results(
            ["TREND_UP", "TREND_UP", "TREND_DOWN", "RANGE", "MOMENTUM"]
        )
        snap = engine.compute(results, "TEST")
        assert snap.universe == "TEST"
        assert 0.0 <= snap.regime_breadth_score <= 1.0
        assert snap.breadth_state in ("EXPANDING","NEUTRAL","CONTRACTING","EXTREME_UP","EXTREME_DOWN")

    def test_bullish_universe_expands_state(self):
        from stock_regime.breadth_engine import BreadthEngine
        engine  = BreadthEngine(expanding_threshold=50.0)
        results = self._make_stable_results(["TREND_UP"] * 8 + ["TREND_DOWN"] * 2)
        snap = engine.compute(results, "TEST")
        assert snap.breadth_state in ("EXPANDING", "EXTREME_UP")
        assert snap.pct_bullish > 50.0

    def test_bearish_universe_contracts_state(self):
        from stock_regime.breadth_engine import BreadthEngine
        engine  = BreadthEngine(contracting_threshold=40.0)
        results = self._make_stable_results(["TREND_DOWN"] * 6 + ["RANGE"] * 4)
        snap = engine.compute(results, "TEST")
        assert snap.breadth_state in ("CONTRACTING", "EXTREME_DOWN")

    def test_ad_ratio_correct(self):
        from stock_regime.breadth_engine import BreadthEngine
        engine  = BreadthEngine()
        results = self._make_stable_results(
            ["TREND_UP"] * 6 + ["TREND_DOWN"] * 2 + ["RANGE"] * 2
        )
        snap = engine.compute(results, "TEST")
        assert abs(snap.advance_decline_ratio - 3.0) < 0.01

    def test_breadth_persists_to_parquet(self, tmp_path):
        from stock_regime.breadth_engine import BreadthEngine
        engine  = BreadthEngine()
        results = self._make_stable_results(["TREND_UP"] * 5 + ["RANGE"] * 5)
        snap    = engine.compute(results, "TEST")
        path    = engine.persist(snap, tmp_path, append=False)
        assert path.exists()
        df = pd.read_parquet(path)
        assert "pct_bullish" in df.columns
        assert "breadth_state" in df.columns

    def test_breadth_thrust_computed(self, tmp_path):
        from stock_regime.breadth_engine import BreadthEngine
        engine  = BreadthEngine()
        results = self._make_stable_results(["TREND_UP"] * 5 + ["RANGE"] * 5)
        snap1   = engine.compute(results, "TEST")
        engine.persist(snap1, tmp_path, append=False)
        prior   = BreadthEngine.load_prior(tmp_path, "TEST")
        results2 = self._make_stable_results(["TREND_UP"] * 8 + ["RANGE"] * 2)
        snap2    = engine.compute(results2, "TEST", prior_snapshot=prior)
        assert snap2.breadth_thrust > 0   # more bullish than before


# ─────────────────────────────────────────────────────────────────────────────
# Phase 5 — Sector Engine
# ─────────────────────────────────────────────────────────────────────────────
class TestSectorEngine:
    def _make_stable_results(self, symbol_regimes: dict[str, str]):
        from stock_regime.stability.stabiliser import StableRegimeResult
        from stock_regime.src.models import (
            StockRegime, StockRegimeResult, DimensionalScores,
            StockSignals, StockIndicatorSnapshot,
        )
        results = []
        for sym, regime in symbol_regimes.items():
            raw = StockRegimeResult(
                symbol=sym, market="TEST",
                stock_regime=StockRegime(regime), confidence=0.70,
                dimensional_scores=DimensionalScores(trend=0.65, momentum=0.55),
                regime_scores={}, signals=StockSignals(),
                indicators=StockIndicatorSnapshot(),
            )
            sr = StableRegimeResult.from_result(
                raw, stable_regime=StockRegime(regime),
                prior_stable_regime=StockRegime.UNCERTAIN,
                regime_age_bars=5, stable_regime_age=5,
                regime_changed_today=False,
                smoothed_confidence=0.70, oscillation_detected=False,
            )
            results.append(sr)
        return results

    def test_sector_engine_no_map_uses_unknown(self):
        from stock_regime.sector_engine import SectorEngine
        engine  = SectorEngine(sector_map_path=None)
        results = self._make_stable_results({"AAPL": "TREND_UP", "MSFT": "MOMENTUM"})
        snaps   = engine.compute(results, "SP500")
        assert len(snaps) == 1
        assert snaps[0].sector == "UNKNOWN"

    def test_sector_engine_with_map(self, tmp_path):
        from stock_regime.sector_engine import SectorEngine
        # Write a sector CSV
        csv = tmp_path / "sectors.csv"
        csv.write_text("symbol,sector\nAAPL,TECHNOLOGY\nMSFT,TECHNOLOGY\nJPM,BANKING\n")
        engine  = SectorEngine(sector_map_path=csv)
        results = self._make_stable_results({
            "AAPL": "TREND_UP", "MSFT": "MOMENTUM", "JPM": "TREND_DOWN"
        })
        snaps   = engine.compute(results, "SP500")
        sectors = {s.sector for s in snaps}
        assert "TECHNOLOGY" in sectors
        assert "BANKING"    in sectors

    def test_leading_sector_classification(self, tmp_path):
        from stock_regime.sector_engine import SectorEngine
        csv = tmp_path / "sectors.csv"
        csv.write_text("symbol,sector\n" +
                       "\n".join(f"S{i},IT" for i in range(10)))
        engine  = SectorEngine(sector_map_path=csv, leading_threshold=60.0)
        results = self._make_stable_results({f"S{i}": "TREND_UP" for i in range(10)})
        snaps   = engine.compute(results, "NIFTY500")
        it_snap = next(s for s in snaps if s.sector == "IT")
        assert it_snap.sector_state == "LEADING"

    def test_sector_persist_parquet(self, tmp_path):
        from stock_regime.sector_engine import SectorEngine
        engine  = SectorEngine()
        results = self._make_stable_results({"A": "TREND_UP", "B": "RANGE"})
        snaps   = engine.compute(results, "TEST")
        path    = engine.persist(snaps, tmp_path, append=False)
        assert path is not None and path.exists()
        df = pd.read_parquet(path)
        assert "sector_state"    in df.columns
        assert "trend_strength"  in df.columns


# ─────────────────────────────────────────────────────────────────────────────
# Phase 7 — Strategy Router
# ─────────────────────────────────────────────────────────────────────────────
class TestStrategyRouter:
    def _make_inputs(self, regime="TREND_UP", conf=0.75, quality=0.70):
        from stock_regime.stability.stabiliser import StableRegimeResult
        from stock_regime.src.models import (
            StockRegime, StockRegimeResult, DimensionalScores,
            StockSignals, StockIndicatorSnapshot,
        )
        from stock_regime.quality_engine.opportunity_quality import QualityScore
        from datetime import date

        raw = StockRegimeResult(
            symbol="SYM", market="TEST",
            stock_regime=StockRegime(regime), confidence=conf,
            dimensional_scores=DimensionalScores(trend=0.70, momentum=0.60),
            regime_scores={}, signals=StockSignals(rs_improving=True),
            indicators=StockIndicatorSnapshot(),
        )
        sr = StableRegimeResult.from_result(
            raw, stable_regime=StockRegime(regime),
            prior_stable_regime=StockRegime.UNCERTAIN,
            regime_age_bars=8, stable_regime_age=8,
            regime_changed_today=False,
            smoothed_confidence=conf, oscillation_detected=False,
        )
        qs = QualityScore(
            symbol="SYM", market="TEST", run_date=date.today(),
            quality_score=quality, liquidity_quality=0.75,
            trend_quality=0.80, vol_health=0.70,
            stability_quality=0.65, tradability=0.75, notes=[],
        )
        return sr, qs

    def test_trend_up_bullish_market_routes_trend_following(self):
        from stock_regime.strategy_router import StrategyRouter
        router = StrategyRouter()
        sr, qs = self._make_inputs("TREND_UP", conf=0.75, quality=0.70)
        decisions = router.route_batch([sr], [qs], market_regime="BULLISH_TREND",
                                       breadth_state="EXPANDING")
        assert len(decisions) == 1
        assert decisions[0].strategy == "TREND_FOLLOWING"
        assert decisions[0].allowed  is True

    def test_momentum_regime_routes_momentum(self):
        from stock_regime.strategy_router import StrategyRouter
        router = StrategyRouter()
        sr, qs = self._make_inputs("MOMENTUM", conf=0.75, quality=0.70)
        decisions = router.route_batch([sr], [qs], market_regime="BULLISH_TREND",
                                       breadth_state="NEUTRAL")
        assert decisions[0].strategy == "MOMENTUM"

    def test_range_routes_mean_reversion(self):
        from stock_regime.strategy_router import StrategyRouter
        router = StrategyRouter()
        sr, qs = self._make_inputs("RANGE", conf=0.72, quality=0.68)
        decisions = router.route_batch([sr], [qs], market_regime="SIDEWAYS",
                                       breadth_state="NEUTRAL")
        assert decisions[0].strategy == "MEAN_REVERSION"

    def test_breakout_setup_routes_breakout(self):
        from stock_regime.strategy_router import StrategyRouter
        router = StrategyRouter()
        sr, qs = self._make_inputs("BREAKOUT_SETUP", conf=0.72, quality=0.68)
        decisions = router.route_batch([sr], [qs], market_regime="BULLISH_TREND",
                                       breadth_state="EXPANDING",
                                       sector_states={"SYM": "NEUTRAL"})
        assert decisions[0].strategy == "BREAKOUT"

    def test_volatile_regime_no_trade(self):
        from stock_regime.strategy_router import StrategyRouter
        router = StrategyRouter()
        sr, qs = self._make_inputs("VOLATILE", conf=0.70, quality=0.68)
        decisions = router.route_batch([sr], [qs], market_regime="VOLATILE",
                                       breadth_state="CONTRACTING")
        assert decisions[0].strategy == "NO_TRADE"
        assert decisions[0].allowed  is False

    def test_low_quality_no_trade(self):
        from stock_regime.strategy_router import StrategyRouter
        router = StrategyRouter(min_quality_for_trade=0.60)
        sr, qs = self._make_inputs("TREND_UP", conf=0.75, quality=0.30)
        decisions = router.route_batch([sr], [qs], market_regime="BULLISH_TREND",
                                       breadth_state="NEUTRAL")
        assert decisions[0].strategy == "NO_TRADE"

    def test_adverse_breadth_reduces_size(self):
        from stock_regime.strategy_router import StrategyRouter
        router = StrategyRouter()
        sr, qs = self._make_inputs("TREND_UP", conf=0.75, quality=0.80)
        # Good quality but bad breadth
        d_good = router.route_batch([sr], [qs], market_regime="BULLISH_TREND",
                                     breadth_state="EXPANDING")
        d_bad  = router.route_batch([sr], [qs], market_regime="BULLISH_TREND",
                                     breadth_state="CONTRACTING")
        assert d_bad[0].position_size_multiplier < d_good[0].position_size_multiplier

    def test_leading_sector_boosts_size(self):
        from stock_regime.strategy_router import StrategyRouter
        router = StrategyRouter()
        sr, qs = self._make_inputs("TREND_UP", conf=0.75, quality=0.72)
        d_leading = router.route_batch([sr], [qs], market_regime="BULLISH_TREND",
                                        breadth_state="NEUTRAL",
                                        sector_states={"SYM": "LEADING"})
        d_lagging = router.route_batch([sr], [qs], market_regime="BULLISH_TREND",
                                        breadth_state="NEUTRAL",
                                        sector_states={"SYM": "LAGGING"})
        assert d_leading[0].position_size_multiplier > d_lagging[0].position_size_multiplier

    def test_routing_decision_serialises(self):
        import json
        from stock_regime.strategy_router import StrategyRouter
        router = StrategyRouter()
        sr, qs = self._make_inputs("TREND_UP", conf=0.75, quality=0.70)
        decisions = router.route_batch([sr], [qs], market_regime="BULLISH_TREND",
                                        breadth_state="NEUTRAL")
        d = decisions[0].to_dict()
        serialised = json.dumps(d)  # must not raise
        parsed = json.loads(serialised)
        for key in ["symbol","strategy","allowed","risk_profile","position_size_multiplier","reason"]:
            assert key in parsed

    def test_router_persist_parquet(self, tmp_path):
        from stock_regime.strategy_router import StrategyRouter
        router = StrategyRouter()
        sr, qs = self._make_inputs("TREND_UP", conf=0.75, quality=0.70)
        decisions = router.route_batch([sr], [qs], market_regime="BULLISH_TREND",
                                        breadth_state="NEUTRAL")
        path = router.persist(decisions, tmp_path, universe="TEST")
        assert path is not None and path.exists()
        df = pd.read_parquet(path)
        assert "strategy" in df.columns
        assert "allowed"  in df.columns


# ─────────────────────────────────────────────────────────────────────────────
# Pipeline Integration
# ─────────────────────────────────────────────────────────────────────────────
class TestPipelineIntegration:
    @pytest.fixture
    def symbol_files(self, tmp_path):
        uni = tmp_path / "data" / "universes"
        uni.mkdir(parents=True)
        (uni / "nifty500.txt").write_text("INFY.NS\nRELIANCE.NS\nHDFCBANK.NS\n")
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
                           data=_df(seed=i, n=400), success=True)
            for i, s in enumerate(syms)
        }

    def test_pipeline_produces_routing_decisions(self, pipeline):
        self._mock(pipeline)
        out = pipeline.run(universes=["NIFTY500"], persist=False)
        rd  = out.routing_decisions.get("NIFTY500", [])
        assert len(rd) > 0
        for d in rd:
            assert d.strategy in ("TREND_FOLLOWING","MOMENTUM","MEAN_REVERSION",
                                  "BREAKOUT","NO_TRADE")

    def test_pipeline_produces_breadth_snapshot(self, pipeline):
        self._mock(pipeline)
        out = pipeline.run(universes=["NIFTY500"], persist=False)
        br  = out.breadth.get("NIFTY500")
        assert br is not None
        assert 0.0 <= br.regime_breadth_score <= 1.0

    def test_pipeline_produces_sector_snapshots(self, pipeline):
        self._mock(pipeline)
        out   = pipeline.run(universes=["NIFTY500"], persist=False)
        sects = out.sectors.get("NIFTY500", [])
        assert isinstance(sects, list)  # may be empty (no sector CSV) or have UNKNOWN

    def test_pipeline_persist_writes_router_parquet(self, pipeline, tmp_path):
        self._mock(pipeline)
        pipeline.run(universes=["NIFTY500"], persist=True)
        router_files = list(Path(tmp_path / "output" / "router").rglob("*.parquet"))
        assert len(router_files) > 0

    def test_pipeline_persist_writes_breadth_parquet(self, pipeline, tmp_path):
        self._mock(pipeline)
        pipeline.run(universes=["NIFTY500"], persist=True)
        breadth_files = list(Path(tmp_path / "output" / "breadth").rglob("*.parquet"))
        assert len(breadth_files) > 0
        df = pd.read_parquet(breadth_files[0])
        assert "pct_bullish"    in df.columns
        assert "breadth_state"  in df.columns