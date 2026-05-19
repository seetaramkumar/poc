"""
runner/tests/test_breadth_router.py
=====================================
Tests for the calibrated breadth engine and adaptive strategy router.

Run:  cd algo_platform && pytest runner/tests/test_breadth_router.py -v
"""
from __future__ import annotations
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))


# ─────────────────────────────────────────────────────────────────────────────
#  Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _make_stable(symbol: str, raw_regime: str, stable_regime: str,
                 conf: float = 0.72, above_ema200: bool = True):
    """Build a minimal StableRegimeResult."""
    from stock_regime.stability.stabiliser import StableRegimeResult
    from stock_regime.src.models import (
        StockRegime, StockRegimeResult, DimensionalScores,
        StockSignals, StockIndicatorSnapshot,
    )
    sig = StockSignals(price_above_ema200=above_ema200)
    raw = StockRegimeResult(
        symbol=symbol, market="TEST",
        stock_regime=StockRegime(raw_regime), confidence=conf,
        dimensional_scores=DimensionalScores(trend=0.60, momentum=0.50),
        regime_scores={}, signals=sig,
        indicators=StockIndicatorSnapshot(),
    )
    return StableRegimeResult.from_result(
        raw,
        stable_regime=StockRegime(stable_regime),
        prior_stable_regime=StockRegime.UNCERTAIN,
        regime_age_bars=5, stable_regime_age=5,
        regime_changed_today=False,
        smoothed_confidence=conf, oscillation_detected=False,
    )


def _make_quality(symbol: str, score: float):
    from stock_regime.quality_engine.opportunity_quality import QualityScore
    return QualityScore(
        symbol=symbol, market="TEST", run_date=date.today(),
        quality_score=score, liquidity_quality=0.70,
        trend_quality=0.70, vol_health=0.65,
        stability_quality=0.65, tradability=0.70, notes=[],
    )


# ─────────────────────────────────────────────────────────────────────────────
#  Phase 1 — Breadth Engine: Hybrid Participation
# ─────────────────────────────────────────────────────────────────────────────

class TestBreadthHybridParticipation:
    def _engine(self, **kw):
        from stock_regime.breadth_engine import BreadthEngine
        return BreadthEngine(**kw)

    def test_stable_bullish_counted(self):
        """Stocks with stable=TREND_UP count as stable bullish."""
        engine = self._engine()
        results = [
            _make_stable("A", "TREND_UP", "TREND_UP"),
            _make_stable("B", "TREND_UP", "TREND_UP"),
            _make_stable("C", "RANGE",    "RANGE"),
        ]
        snap = engine.compute(results, "TEST")
        assert snap.pct_stable_bullish > 0
        assert snap.pct_bullish > 0

    def test_emerging_bullish_counted_when_stable_uncertain(self):
        """Stocks with raw=TREND_UP but stable=UNCERTAIN should contribute to emerging."""
        engine  = self._engine(emerging_weight=1.0)  # full weight for easy assertion
        results = [
            _make_stable("A", "TREND_UP", "UNCERTAIN"),   # emerging
            _make_stable("B", "TREND_UP", "UNCERTAIN"),   # emerging
            _make_stable("C", "RANGE",    "RANGE"),
        ]
        snap = engine.compute(results, "TEST")
        assert snap.pct_emerging_bullish > 0, \
            "Stocks with raw=TREND_UP but stable=UNCERTAIN must count as emerging"
        assert snap.pct_stable_bullish == 0, \
            "No stable bullish stocks were added"

    def test_pct_bullish_higher_with_hybrid(self):
        """
        With hybrid counting, pct_bullish should be higher than if we only
        counted stable regimes — because emerging stocks are now included.
        """
        engine = self._engine(emerging_weight=0.40)
        results = [
            _make_stable("A", "TREND_UP", "UNCERTAIN"),   # raw=bullish, stable=UNCERTAIN
            _make_stable("B", "TREND_UP", "UNCERTAIN"),
            _make_stable("C", "RANGE",    "RANGE"),
            _make_stable("D", "RANGE",    "RANGE"),
        ]
        snap = engine.compute(results, "TEST")
        # pct_stable_bullish = 0, but pct_emerging_bullish > 0
        # so pct_bullish (hybrid) > 0
        assert snap.pct_bullish > snap.pct_stable_bullish, \
            "Hybrid pct_bullish should exceed stable-only count"

    def test_emerging_weight_scales_contribution(self):
        """Higher emerging_weight → higher pct_bullish from emerging stocks."""
        results = [
            _make_stable("A", "TREND_UP", "UNCERTAIN"),
            _make_stable("B", "RANGE",    "RANGE"),
        ]
        from stock_regime.breadth_engine import BreadthEngine
        snap_low  = BreadthEngine(emerging_weight=0.10).compute(results, "TEST")
        snap_high = BreadthEngine(emerging_weight=0.80).compute(results, "TEST")
        assert snap_high.pct_bullish > snap_low.pct_bullish

    def test_stable_regime_preferred_over_raw(self):
        """When stable_regime is not UNCERTAIN it must be used, NOT raw."""
        engine = self._engine()
        # Raw = TREND_DOWN, stable = TREND_UP (stock reversed and confirmed uptrend)
        results = [_make_stable("A", "TREND_DOWN", "TREND_UP")]
        snap = engine.compute(results, "TEST")
        assert snap.pct_stable_bullish > 0, "Stable TREND_UP should count as stable bullish"
        assert snap.pct_bearish == 0.0,     "Raw TREND_DOWN must be ignored when stable exists"

    def test_breadth_state_expanding_at_lower_threshold(self):
        """With expanding_threshold=45%, universe with 50% bullish → EXPANDING."""
        engine  = self._engine(expanding_threshold=45.0, emerging_weight=0.40)
        results = [
            _make_stable("A", "TREND_UP", "TREND_UP"),
            _make_stable("B", "TREND_UP", "TREND_UP"),
            _make_stable("C", "RANGE",    "RANGE"),
            _make_stable("D", "RANGE",    "RANGE"),
        ]
        snap = engine.compute(results, "TEST")
        assert snap.breadth_state == "EXPANDING", \
            f"Expected EXPANDING, got {snap.breadth_state} (pct_bullish={snap.pct_bullish})"

    def test_breadth_state_neutral_was_old_threshold(self):
        """50% bullish would be NEUTRAL under old 55% threshold but EXPANDING under new 45%."""
        from stock_regime.breadth_engine import BreadthEngine
        results = [
            _make_stable(f"S{i}", "TREND_UP", "TREND_UP") for i in range(5)
        ] + [
            _make_stable(f"R{i}", "RANGE", "RANGE") for i in range(5)
        ]
        # Old threshold (55%) → NEUTRAL
        old_engine = BreadthEngine(expanding_threshold=55.0, emerging_weight=0.0)
        old_snap   = old_engine.compute(results, "TEST")

        # New threshold (45%) → EXPANDING
        new_engine = BreadthEngine(expanding_threshold=45.0, emerging_weight=0.0)
        new_snap   = new_engine.compute(results, "TEST")

        assert old_snap.breadth_state == "NEUTRAL"
        assert new_snap.breadth_state == "EXPANDING"

    def test_participation_mode_emerging(self):
        """All-emerging universe → mode=EMERGING."""
        engine  = self._engine(emerging_weight=0.40)
        results = [_make_stable(f"S{i}", "TREND_UP", "UNCERTAIN") for i in range(5)]
        snap    = engine.compute(results, "TEST")
        assert snap.participation_mode == "EMERGING"

    def test_participation_mode_stable(self):
        """All-stable universe → mode=STABLE."""
        engine  = self._engine(emerging_weight=0.40)
        results = [_make_stable(f"S{i}", "TREND_UP", "TREND_UP") for i in range(5)]
        snap    = engine.compute(results, "TEST")
        assert snap.participation_mode == "STABLE"

    def test_breadth_snapshot_new_fields_present(self):
        engine  = self._engine()
        results = [_make_stable("A", "TREND_UP", "TREND_UP")]
        snap    = engine.compute(results, "TEST")
        d = snap.to_dict()
        for key in ["pct_stable_bullish", "pct_stable_bearish",
                    "pct_emerging_bullish", "pct_emerging_bearish",
                    "participation_mode"]:
            assert key in d, f"Missing field: {key}"

    def test_persist_backward_compat(self, tmp_path):
        """New fields gracefully added to existing parquet without errors."""
        from stock_regime.breadth_engine import BreadthEngine
        engine  = BreadthEngine()
        results = [_make_stable("A", "TREND_UP", "TREND_UP")]
        snap    = engine.compute(results, "TEST")
        path    = engine.persist(snap, tmp_path, append=False)
        assert path.exists()
        df = pd.read_parquet(path)
        assert "pct_emerging_bullish" in df.columns
        assert "participation_mode"   in df.columns

    def test_thrust_detection_on_improving_universe(self, tmp_path):
        """breadth_thrust should be positive when pct_bullish increases."""
        from stock_regime.breadth_engine import BreadthEngine
        engine   = BreadthEngine(emerging_weight=0.40)
        # Day 1: 20% bullish
        day1 = [
            _make_stable("A", "TREND_UP", "TREND_UP"),
            _make_stable("B", "RANGE",    "RANGE"),
            _make_stable("C", "RANGE",    "RANGE"),
            _make_stable("D", "RANGE",    "RANGE"),
        ]
        snap1 = engine.compute(day1, "TEST", run_date=date(2024, 1, 1))
        engine.persist(snap1, tmp_path, append=False)
        prior = BreadthEngine.load_prior(tmp_path, "TEST")

        # Day 2: 75% bullish
        day2 = [
            _make_stable("A", "TREND_UP", "TREND_UP"),
            _make_stable("B", "TREND_UP", "TREND_UP"),
            _make_stable("C", "TREND_UP", "TREND_UP"),
            _make_stable("D", "RANGE",    "RANGE"),
        ]
        snap2 = engine.compute(day2, "TEST", run_date=date(2024, 1, 2),
                               prior_snapshot=prior)
        assert snap2.breadth_thrust > 0, \
            f"Expected positive thrust, got {snap2.breadth_thrust}"


# ─────────────────────────────────────────────────────────────────────────────
#  Phase 2 — Strategy Router: Adaptive Participation
# ─────────────────────────────────────────────────────────────────────────────

class TestAdaptiveRouter:
    def _router(self, **kw):
        from stock_regime.strategy_router import StrategyRouter
        return StrategyRouter(**kw)

    def _route_one(self, regime, market, quality=0.65, conf=0.70,
                   breadth="NEUTRAL", breadth_score=0.55, sector="NEUTRAL"):
        router  = self._router()
        sr      = _make_stable("SYM", regime, regime, conf=conf)
        qs      = _make_quality("SYM", quality)
        return router.route_batch(
            [sr], [qs], market_regime=market,
            breadth_state=breadth, breadth_score=breadth_score,
            sector_states={"SYM": sector},
        )[0]

    # ── Bearish market does NOT globally block all trades ─────────────
    def test_bearish_market_allows_high_quality_trend_up(self):
        """A high-quality TREND_UP stock must be allowed even in bearish market."""
        d = self._route_one("TREND_UP", "BEARISH_TREND", quality=0.75, conf=0.72)
        assert d.allowed  is True,  f"Expected allowed, got blocked: {d.reason}"
        assert d.strategy == "TREND_FOLLOWING"

    def test_bearish_market_blocks_low_quality_trend_up(self):
        """Low-quality stock in bearish market must be blocked (stricter gate)."""
        d = self._route_one("TREND_UP", "BEARISH_TREND", quality=0.40, conf=0.55)
        assert d.allowed is False
        assert d.strategy == "NO_TRADE"

    def test_bearish_market_reduces_psm_vs_bullish(self):
        """Bearish market should produce smaller PSM than bullish for same stock."""
        d_bull = self._route_one("TREND_UP", "BULLISH_TREND", quality=0.75, conf=0.72)
        d_bear = self._route_one("TREND_UP", "BEARISH_TREND", quality=0.75, conf=0.72)
        assert d_bear.position_size_multiplier < d_bull.position_size_multiplier, \
            f"Bearish PSM {d_bear.position_size_multiplier} should be < " \
            f"bullish PSM {d_bull.position_size_multiplier}"

    # ── Volatile market participates at reduced sizing ────────────────
    def test_volatile_market_allows_high_quality_trend_up(self):
        """High-quality TREND_UP allowed in volatile market at reduced size."""
        d = self._route_one("TREND_UP", "VOLATILE", quality=0.80, conf=0.75)
        assert d.allowed  is True,  f"Expected allowed: {d.reason}"
        assert d.strategy == "TREND_FOLLOWING"

    def test_volatile_market_blocks_mediocre_quality(self):
        """Volatile market has strictest gate — mediocre quality blocked."""
        d = self._route_one("TREND_UP", "VOLATILE", quality=0.50, conf=0.60)
        assert d.allowed is False

    def test_volatile_market_psm_is_lowest(self):
        """VOLATILE market should give smallest PSM (CAPITAL_PRESERVATION base)."""
        d_bull = self._route_one("TREND_UP", "BULLISH_TREND", quality=0.80, conf=0.75)
        d_vol  = self._route_one("TREND_UP", "VOLATILE",      quality=0.80, conf=0.75)
        assert d_vol.position_size_multiplier < d_bull.position_size_multiplier

    # ── Sideways market favours mean-reversion ────────────────────────
    def test_sideways_market_routes_range_to_mean_reversion(self):
        d = self._route_one("RANGE", "SIDEWAYS", quality=0.60, conf=0.62)
        assert d.strategy == "MEAN_REVERSION"
        assert d.allowed  is True

    def test_sideways_market_allows_trend_defensively(self):
        """TREND_UP should be allowed in sideways but at DEFENSIVE sizing."""
        d = self._route_one("TREND_UP", "SIDEWAYS", quality=0.62, conf=0.62)
        assert d.allowed  is True
        assert d.strategy == "TREND_FOLLOWING"
        assert d.risk_profile in ("DEFENSIVE", "CAPITAL_PRESERVATION")

    # ── Bullish market → normal/aggressive sizing ─────────────────────
    def test_bullish_market_high_quality_gets_normal_or_aggressive(self):
        d = self._route_one("TREND_UP", "BULLISH_TREND", quality=0.80, conf=0.78,
                            breadth="EXPANDING", breadth_score=0.70, sector="LEADING")
        assert d.allowed is True
        assert d.risk_profile in ("NORMAL", "AGGRESSIVE"), \
            f"Expected NORMAL or AGGRESSIVE, got {d.risk_profile}"

    # ── Unconditional blocks remain ────────────────────────────────────
    def test_trend_down_always_no_trade(self):
        for market in ("BULLISH_TREND", "BEARISH_TREND", "SIDEWAYS"):
            d = self._route_one("TREND_DOWN", market, quality=0.90, conf=0.90)
            assert d.strategy == "NO_TRADE", \
                f"TREND_DOWN should always be NO_TRADE, got {d.strategy} in {market}"

    def test_volatile_regime_always_no_trade(self):
        d = self._route_one("VOLATILE", "BULLISH_TREND", quality=0.90, conf=0.90)
        assert d.strategy == "NO_TRADE"

    def test_uncertain_regime_always_no_trade(self):
        d = self._route_one("UNCERTAIN", "BULLISH_TREND", quality=0.90, conf=0.90)
        assert d.strategy == "NO_TRADE"

    # ── Momentum routing ─────────────────────────────────────────────
    def test_momentum_with_contracting_breadth_allowed_defensive(self):
        """MOMENTUM + contracting breadth → allowed but defensive (not blocked)."""
        d = self._route_one("MOMENTUM", "BULLISH_TREND", quality=0.68, conf=0.68,
                            breadth="CONTRACTING", breadth_score=0.35)
        assert d.allowed  is True,  f"MOMENTUM should be allowed in contracting: {d.reason}"
        assert d.strategy == "MOMENTUM"

    def test_momentum_with_extreme_down_blocked(self):
        """EXTREME_DOWN breadth should block MOMENTUM entirely."""
        d = self._route_one("MOMENTUM", "BULLISH_TREND", quality=0.68, conf=0.68,
                            breadth="EXTREME_DOWN", breadth_score=0.15)
        assert d.strategy == "NO_TRADE"

    # ── Risk profiles ─────────────────────────────────────────────────
    def test_all_four_risk_profiles_reachable(self):
        from stock_regime.strategy_router.router import (
            RISK_AGGRESSIVE, RISK_NORMAL, RISK_DEFENSIVE, RISK_CAPITAL_PRESERVATION
        )
        router = self._router()

        configs = [
            # (regime, market, quality, conf, breadth, b_score, sector, expected_profile)
            ("TREND_UP", "BULLISH_TREND", 0.90, 0.85, "EXPANDING", 0.80, "LEADING",  RISK_AGGRESSIVE),
            ("TREND_UP", "BULLISH_TREND", 0.72, 0.70, "NEUTRAL",   0.55, "NEUTRAL",  RISK_NORMAL),
            ("TREND_UP", "BEARISH_TREND", 0.70, 0.68, "NEUTRAL",   0.40, "NEUTRAL",  RISK_DEFENSIVE),
            ("TREND_UP", "VOLATILE",      0.78, 0.72, "NEUTRAL",   0.35, "NEUTRAL",  RISK_CAPITAL_PRESERVATION),
        ]
        for regime, market, quality, conf, breadth, b_score, sector, expected in configs:
            sr = _make_stable("SYM", regime, regime, conf=conf)
            qs = _make_quality("SYM", quality)
            decisions = router.route_batch(
                [sr], [qs], market_regime=market,
                breadth_state=breadth, breadth_score=b_score,
                sector_states={"SYM": sector},
            )
            d = decisions[0]
            if d.allowed:
                assert d.risk_profile == expected, (
                    f"market={market} quality={quality} breadth={breadth} "
                    f"sector={sector}: expected {expected}, got {d.risk_profile} "
                    f"(psm={d.position_size_multiplier})"
                )

    def test_psm_increases_with_quality(self):
        """Higher quality → higher PSM, all else equal."""
        router = self._router()
        psms = []
        for q in [0.42, 0.55, 0.70, 0.85]:
            sr = _make_stable("SYM", "TREND_UP", "TREND_UP", conf=0.72)
            qs = _make_quality("SYM", q)
            d  = router.route_batch([sr], [qs], market_regime="BULLISH_TREND",
                                     breadth_state="NEUTRAL", breadth_score=0.55)[0]
            if d.allowed:
                psms.append((q, d.position_size_multiplier))
        # Verify monotonically increasing
        for i in range(1, len(psms)):
            assert psms[i][1] >= psms[i-1][1], \
                f"PSM should increase with quality: {psms}"

    def test_leading_sector_boosts_psm_vs_lagging(self):
        router = self._router()
        def _route(sector):
            sr = _make_stable("SYM", "TREND_UP", "TREND_UP", conf=0.72)
            qs = _make_quality("SYM", 0.70)
            return router.route_batch(
                [sr], [qs], market_regime="BULLISH_TREND",
                breadth_state="NEUTRAL", breadth_score=0.55,
                sector_states={"SYM": sector},
            )[0]
        d_lead = _route("LEADING")
        d_lag  = _route("LAGGING")
        assert d_lead.position_size_multiplier > d_lag.position_size_multiplier

    def test_breadth_score_continuous_effect(self):
        """Higher breadth_score → higher PSM (continuous, not binary)."""
        router = self._router()
        results = []
        for score in [0.20, 0.40, 0.60, 0.80]:
            sr = _make_stable("SYM", "TREND_UP", "TREND_UP", conf=0.72)
            qs = _make_quality("SYM", 0.70)
            d  = router.route_batch(
                [sr], [qs], market_regime="BULLISH_TREND",
                breadth_state="NEUTRAL", breadth_score=score,
            )[0]
            if d.allowed:
                results.append((score, d.position_size_multiplier))
        for i in range(1, len(results)):
            assert results[i][1] >= results[i-1][1], \
                f"PSM not monotone with breadth_score: {results}"

    # ── Serialisation ─────────────────────────────────────────────────
    def test_routing_decision_serialises_with_new_profiles(self):
        import json
        d = self._route_one("TREND_UP", "BEARISH_TREND", quality=0.75, conf=0.72)
        parsed = json.loads(json.dumps(d.to_dict()))
        assert parsed["risk_profile"] in (
            "AGGRESSIVE", "NORMAL", "DEFENSIVE",
            "CAPITAL_PRESERVATION", "OFF"
        )

    def test_router_persist_parquet(self, tmp_path):
        router = self._router()
        sr = _make_stable("SYM", "TREND_UP", "TREND_UP", conf=0.72)
        qs = _make_quality("SYM", 0.72)
        d  = router.route_batch([sr], [qs], market_regime="BULLISH_TREND",
                                 breadth_state="NEUTRAL", breadth_score=0.55)[0]
        path = router.persist([d], tmp_path, universe="TEST")
        assert path.exists()
        df = pd.read_parquet(path)
        assert "risk_profile" in df.columns
        assert "strategy"     in df.columns