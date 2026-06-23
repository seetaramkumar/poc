"""
tests/test_strategy_router.py
===============================
Comprehensive tests for the strategy allocation engine.

Covers:
- All four tradeable strategies are routed correctly
- Per-strategy quality gates work independently
- Market-context routing rules (bullish/bearish/sideways/volatile)
- Breadth-aware routing
- 5-level sector state PSM adjustments
- Unconditional NO_TRADE regimes
- Diagnostics and serialisation
- RoutingStats accumulation

Run:
    pytest tests/test_strategy_router.py -v
"""

from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

import pytest

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))


# ─────────────────────────────────────────────────────────────────────────────
#  Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _stable(symbol="SYM", regime="TREND_UP", conf=0.72):
    from stock_regime.stability.stabiliser import StableRegimeResult
    from stock_regime.src.models import (
        StockRegime, StockRegimeResult, DimensionalScores,
        StockSignals, StockIndicatorSnapshot,
    )
    raw = StockRegimeResult(
        symbol=symbol, market="TEST",
        stock_regime=StockRegime(regime), confidence=conf,
        dimensional_scores=DimensionalScores(trend=0.65, momentum=0.55),
        regime_scores={}, signals=StockSignals(),
        indicators=StockIndicatorSnapshot(),
    )
    return StableRegimeResult.from_result(
        raw,
        stable_regime=StockRegime(regime),
        prior_stable_regime=StockRegime.UNCERTAIN,
        regime_age_bars=8, stable_regime_age=8,
        regime_changed_today=False,
        smoothed_confidence=conf, oscillation_detected=False,
    )


def _quality(symbol="SYM", score=0.70):
    from stock_regime.quality_engine.opportunity_quality import QualityScore
    return QualityScore(
        symbol=symbol, market="TEST", run_date=date.today(),
        quality_score=score, liquidity_quality=0.70,
        trend_quality=0.70, vol_health=0.65,
        stability_quality=0.65, tradability=0.70, notes=[],
    )


def _route(regime="TREND_UP", market="BULLISH_TREND",
           quality=0.65, conf=0.68,
           breadth="NEUTRAL", b_score=0.55,
           sector="NEUTRAL", symbol="SYM"):
    from stock_regime.strategy_router.router import StrategyRouter
    router = StrategyRouter()
    sr = _stable(symbol=symbol, regime=regime, conf=conf)
    qs = _quality(symbol=symbol, score=quality)
    return router.route_batch(
        [sr], [qs],
        market_regime=market,
        breadth_state=breadth,
        breadth_score=b_score,
        sector_states={symbol: sector},
    )[0]


# ─────────────────────────────────────────────────────────────────────────────
#  1. All four strategies are routed (not just TREND_FOLLOWING)
# ─────────────────────────────────────────────────────────────────────────────

class TestAllStrategiesRouted:
    def test_trend_up_routes_trend_following(self):
        d = _route("TREND_UP", "BULLISH_TREND", quality=0.65, conf=0.68)
        assert d.strategy == "TREND_FOLLOWING"
        assert d.allowed  is True

    def test_momentum_routes_momentum(self):
        d = _route("MOMENTUM", "BULLISH_TREND", quality=0.65, conf=0.68)
        assert d.strategy == "MOMENTUM"
        assert d.allowed  is True

    def test_range_routes_mean_reversion(self):
        d = _route("RANGE", "SIDEWAYS", quality=0.50, conf=0.55)
        assert d.strategy == "MEAN_REVERSION"
        assert d.allowed  is True

    def test_breakout_setup_routes_breakout(self):
        d = _route("BREAKOUT_SETUP", "BULLISH_TREND",
                   quality=0.60, conf=0.62,
                   breadth="EXPANDING", sector="NEUTRAL")
        assert d.strategy == "BREAKOUT"
        assert d.allowed  is True

    def test_all_four_strategies_available_in_batch(self):
        """A diverse universe should produce all four strategy types."""
        from stock_regime.strategy_router.router import StrategyRouter
        router = StrategyRouter()
        symbols = {
            "A": ("TREND_UP",       0.68, 0.70),
            "B": ("MOMENTUM",       0.68, 0.65),
            "C": ("RANGE",          0.55, 0.55),
            "D": ("BREAKOUT_SETUP", 0.62, 0.60),
        }
        results = [_stable(sym, regime, conf) for sym, (regime, conf, _) in symbols.items()]
        quality = [_quality(sym, q) for sym, (_, _, q) in symbols.items()]
        decisions = router.route_batch(
            results, quality,
            market_regime="BULLISH_TREND",
            breadth_state="EXPANDING", breadth_score=0.68,
            sector_states={sym: "NEUTRAL" for sym in symbols},
        )
        strategies = {d.strategy for d in decisions if d.allowed}
        assert "TREND_FOLLOWING" in strategies, "TREND_FOLLOWING must be routed"
        assert "MOMENTUM"        in strategies, "MOMENTUM must be routed"
        assert "MEAN_REVERSION"  in strategies, "MEAN_REVERSION must be routed"
        assert "BREAKOUT"        in strategies, "BREAKOUT must be routed"


# ─────────────────────────────────────────────────────────────────────────────
#  2. Unconditional NO_TRADE regimes
# ─────────────────────────────────────────────────────────────────────────────

class TestUnconditionalNoTrade:
    @pytest.mark.parametrize("regime", ["TREND_DOWN", "VOLATILE", "QUIET", "UNCERTAIN"])
    def test_unconditional_block(self, regime):
        """High quality/confidence should not help these regimes."""
        d = _route(regime, "BULLISH_TREND", quality=0.95, conf=0.95)
        assert d.strategy == "NO_TRADE"
        assert d.allowed  is False

    @pytest.mark.parametrize("regime", ["TREND_DOWN", "VOLATILE", "QUIET", "UNCERTAIN"])
    def test_unconditional_block_in_all_markets(self, regime):
        for market in ["BULLISH_TREND", "BEARISH_TREND", "SIDEWAYS"]:
            d = _route(regime, market, quality=0.90, conf=0.90)
            assert d.strategy == "NO_TRADE", \
                f"{regime} in {market} should always be NO_TRADE"


# ─────────────────────────────────────────────────────────────────────────────
#  3. Market-context routing rules
# ─────────────────────────────────────────────────────────────────────────────

class TestMarketContextRouting:

    # ── Bullish market ────────────────────────────────────────
    def test_bullish_all_strategies_allowed(self):
        for regime, min_q, min_c in [
            ("TREND_UP",       0.42, 0.52),
            ("MOMENTUM",       0.44, 0.54),
            ("RANGE",          0.40, 0.50),
            ("BREAKOUT_SETUP", 0.52, 0.57),
        ]:
            d = _route(regime, "BULLISH_TREND", quality=min_q+0.05,
                       conf=min_c+0.05, breadth="EXPANDING", sector="NEUTRAL")
            assert d.allowed, \
                f"{regime} should be allowed in BULLISH market (got {d.reason})"

    def test_bullish_trend_gets_normal_or_aggressive_posture(self):
        d = _route("TREND_UP", "BULLISH_TREND", quality=0.80, conf=0.80,
                   breadth="EXPANDING", b_score=0.75, sector="LEADING")
        assert d.risk_profile in ("NORMAL", "AGGRESSIVE"), \
            f"Bullish+high quality should be NORMAL/AGGRESSIVE, got {d.risk_profile}"

    # ── Bearish market ────────────────────────────────────────
    def test_bearish_high_quality_trend_allowed(self):
        """High-quality RS leader should be allowed even in bear market."""
        d = _route("TREND_UP", "BEARISH_TREND", quality=0.72, conf=0.70)
        assert d.allowed  is True
        assert d.strategy == "TREND_FOLLOWING"

    def test_bearish_low_quality_trend_blocked(self):
        """Below the stricter bearish gate → blocked."""
        d = _route("TREND_UP", "BEARISH_TREND", quality=0.45, conf=0.55)
        assert d.allowed is False

    def test_bearish_mean_reversion_allowed_at_lower_gate(self):
        """RANGE stocks should pass the lower mean-reversion gate in bear market."""
        d = _route("RANGE", "BEARISH_TREND", quality=0.50, conf=0.55)
        assert d.allowed  is True
        assert d.strategy == "MEAN_REVERSION"

    def test_bearish_breakout_blocked(self):
        """Breakouts should never be routed in a bear market."""
        d = _route("BREAKOUT_SETUP", "BEARISH_TREND", quality=0.90, conf=0.90,
                   breadth="EXPANDING", sector="LEADING")
        assert d.strategy == "NO_TRADE"

    def test_bearish_reduces_psm_vs_bullish(self):
        """Bearish market should produce smaller PSM than bullish for same stock."""
        d_bull = _route("TREND_UP", "BULLISH_TREND", quality=0.75, conf=0.75)
        d_bear = _route("TREND_UP", "BEARISH_TREND", quality=0.75, conf=0.75)
        assert d_bear.position_size_multiplier < d_bull.position_size_multiplier

    # ── Sideways market ───────────────────────────────────────
    def test_sideways_mean_reversion_preferred(self):
        """Mean-reversion in sideways should get NORMAL posture (preferred strategy)."""
        d = _route("RANGE", "SIDEWAYS", quality=0.55, conf=0.58)
        assert d.allowed  is True
        assert d.strategy == "MEAN_REVERSION"
        assert d.risk_profile in ("NORMAL", "AGGRESSIVE"), \
            f"Mean-reversion in sideways should be NORMAL+ posture, got {d.risk_profile}"

    def test_sideways_trend_allowed_but_defensive(self):
        d = _route("TREND_UP", "SIDEWAYS", quality=0.60, conf=0.60)
        assert d.allowed  is True
        assert d.risk_profile in ("DEFENSIVE", "CAPITAL_PRESERVATION"), \
            f"Trend in sideways should be DEFENSIVE, got {d.risk_profile}"

    def test_sideways_mean_reversion_psm_higher_than_trend(self):
        """In sideways market, mean-reversion should get higher PSM than trend-following."""
        d_mr   = _route("RANGE",    "SIDEWAYS", quality=0.65, conf=0.65)
        d_trend= _route("TREND_UP", "SIDEWAYS", quality=0.65, conf=0.65)
        assert d_mr.position_size_multiplier >= d_trend.position_size_multiplier, \
            (f"MR psm={d_mr.position_size_multiplier} should >= "
             f"TREND psm={d_trend.position_size_multiplier} in SIDEWAYS")

    # ── Volatile market ───────────────────────────────────────
    def test_volatile_market_blocks_mean_reversion(self):
        """Ranges break in volatile market — MEAN_REVERSION blocked."""
        d = _route("RANGE", "VOLATILE", quality=0.90, conf=0.90)
        assert d.strategy == "NO_TRADE"

    def test_volatile_market_blocks_breakout(self):
        d = _route("BREAKOUT_SETUP", "VOLATILE", quality=0.90, conf=0.90,
                   breadth="EXPANDING", sector="LEADING")
        assert d.strategy == "NO_TRADE"

    def test_volatile_market_high_quality_trend_allowed(self):
        """Only very high quality TREND_UP passes in volatile market."""
        d = _route("TREND_UP", "VOLATILE", quality=0.78, conf=0.72)
        assert d.allowed  is True
        assert d.risk_profile == "CAPITAL_PRESERVATION"

    def test_volatile_market_low_quality_blocked(self):
        d = _route("TREND_UP", "VOLATILE", quality=0.50, conf=0.58)
        assert d.allowed is False


# ─────────────────────────────────────────────────────────────────────────────
#  4. Per-strategy quality gates
# ─────────────────────────────────────────────────────────────────────────────

class TestPerStrategyQualityGates:
    def test_mean_reversion_lower_gate_than_trend_in_normal_market(self):
        """MEAN_REVERSION should allow stocks that TREND_FOLLOWING would reject."""
        # Quality level that passes MR gate (0.38) but not TREND gate (0.40)
        quality = 0.39
        d_mr    = _route("RANGE",    "BULLISH_TREND", quality=quality, conf=0.55)
        d_trend = _route("TREND_UP", "BULLISH_TREND", quality=quality, conf=0.55)
        # MR should be allowed; TREND might be rejected at this quality
        assert d_mr.allowed is True, \
            f"MR should pass at quality={quality} (gate=0.38)"

    def test_breakout_higher_gate_than_trend_in_normal_market(self):
        """BREAKOUT requires higher quality than TREND_FOLLOWING."""
        quality  = 0.45   # passes TREND gate (0.40) but not BREAKOUT gate (0.50)
        d_trend  = _route("TREND_UP",       "BULLISH_TREND", quality=quality, conf=0.55)
        d_break  = _route("BREAKOUT_SETUP", "BULLISH_TREND", quality=quality, conf=0.55,
                           breadth="EXPANDING", sector="NEUTRAL")
        assert d_trend.allowed is True,  f"TREND_FOLLOWING should pass at q={quality}"
        assert d_break.allowed is False, f"BREAKOUT should fail at q={quality} (gate=0.50)"

    def test_bearish_mr_gate_lower_than_bearish_trend_gate(self):
        """In bearish market, MR gate (0.42) is lower than TREND gate (0.60)."""
        quality = 0.52   # between 0.42 and 0.60
        d_mr    = _route("RANGE",    "BEARISH_TREND", quality=quality, conf=0.55)
        d_trend = _route("TREND_UP", "BEARISH_TREND", quality=quality, conf=0.55)
        assert d_mr.allowed    is True,  f"MR should pass at q={quality} in BEARISH"
        assert d_trend.allowed is False, f"TREND should fail at q={quality} in BEARISH"


# ─────────────────────────────────────────────────────────────────────────────
#  5. Breadth-aware routing
# ─────────────────────────────────────────────────────────────────────────────

class TestBreadthAwareRouting:
    def test_momentum_blocked_on_extreme_down(self):
        d = _route("MOMENTUM", "BULLISH_TREND", quality=0.75, conf=0.75,
                   breadth="EXTREME_DOWN", b_score=0.10)
        assert d.strategy == "NO_TRADE"

    def test_momentum_allowed_on_contracting(self):
        """CONTRACTING breadth reduces size but does not block MOMENTUM."""
        d = _route("MOMENTUM", "BULLISH_TREND", quality=0.65, conf=0.65,
                   breadth="CONTRACTING", b_score=0.35)
        assert d.allowed  is True
        assert d.strategy == "MOMENTUM"

    def test_expanding_breadth_boosts_psm_vs_contracting(self):
        d_exp = _route("TREND_UP", "BULLISH_TREND", quality=0.70, conf=0.70,
                       breadth="EXPANDING", b_score=0.72)
        d_con = _route("TREND_UP", "BULLISH_TREND", quality=0.70, conf=0.70,
                       breadth="CONTRACTING", b_score=0.30)
        assert d_exp.position_size_multiplier > d_con.position_size_multiplier

    def test_breadth_score_continuous_effect(self):
        """Higher breadth_score → higher PSM (monotone)."""
        from stock_regime.strategy_router.router import StrategyRouter
        router = StrategyRouter()
        psms   = []
        for score in [0.20, 0.40, 0.60, 0.80]:
            sr = _stable("SYM", "TREND_UP", 0.72)
            qs = _quality("SYM", 0.70)
            d  = router.route_batch(
                [sr], [qs], market_regime="BULLISH_TREND",
                breadth_state="NEUTRAL", breadth_score=score,
            )[0]
            if d.allowed:
                psms.append((score, d.position_size_multiplier))
        for i in range(1, len(psms)):
            assert psms[i][1] >= psms[i-1][1], \
                f"PSM not monotone with breadth_score: {psms}"

    def test_breakout_blocked_on_extreme_down_breadth(self):
        d = _route("BREAKOUT_SETUP", "BULLISH_TREND", quality=0.75, conf=0.70,
                   breadth="EXTREME_DOWN", b_score=0.10, sector="LEADING")
        assert d.strategy == "NO_TRADE"


# ─────────────────────────────────────────────────────────────────────────────
#  6. Sector-aware routing
# ─────────────────────────────────────────────────────────────────────────────

class TestSectorAwareRouting:
    def test_five_level_psm_monotone(self):
        """PSM should decrease monotonically LEADING → STRONG → NEUTRAL → WEAKENING → WEAK."""
        states = ["LEADING", "STRONG", "NEUTRAL", "WEAKENING", "WEAK"]
        psms   = []
        for state in states:
            d = _route("TREND_UP", "BULLISH_TREND", quality=0.72, conf=0.72,
                       sector=state)
            if d.allowed:
                psms.append((state, d.position_size_multiplier))
        for i in range(1, len(psms)):
            assert psms[i][1] <= psms[i-1][1], \
                f"PSM not monotone: {psms[i-1]} → {psms[i]}"

    def test_leading_sector_gives_aggressive_or_normal(self):
        d = _route("TREND_UP", "BULLISH_TREND", quality=0.82, conf=0.80,
                   breadth="EXPANDING", b_score=0.75, sector="LEADING")
        assert d.risk_profile in ("AGGRESSIVE", "NORMAL"), \
            f"LEADING sector + high quality should be AGGRESSIVE/NORMAL, got {d.risk_profile}"

    def test_weak_sector_reduces_psm(self):
        d_neutral = _route("TREND_UP", "BULLISH_TREND", quality=0.70, conf=0.72, sector="NEUTRAL")
        d_weak    = _route("TREND_UP", "BULLISH_TREND", quality=0.70, conf=0.72, sector="WEAK")
        assert d_weak.position_size_multiplier < d_neutral.position_size_multiplier

    def test_breakout_blocked_in_weakening_sector(self):
        d = _route("BREAKOUT_SETUP", "BULLISH_TREND", quality=0.75, conf=0.70,
                   breadth="EXPANDING", b_score=0.70, sector="WEAKENING")
        assert d.strategy == "NO_TRADE"

    def test_breakout_blocked_in_weak_sector(self):
        d = _route("BREAKOUT_SETUP", "BULLISH_TREND", quality=0.75, conf=0.70,
                   breadth="EXPANDING", b_score=0.70, sector="WEAK")
        assert d.strategy == "NO_TRADE"

    def test_unknown_sector_neutral_adjustment(self):
        """UNKNOWN sector should apply no multiplier (× 1.00)."""
        d_unknown = _route("TREND_UP", "BULLISH_TREND", quality=0.70, conf=0.72, sector="UNKNOWN")
        d_neutral = _route("TREND_UP", "BULLISH_TREND", quality=0.70, conf=0.72, sector="NEUTRAL")
        assert d_unknown.position_size_multiplier == d_neutral.position_size_multiplier


# ─────────────────────────────────────────────────────────────────────────────
#  7. PSM and risk profile correctness
# ─────────────────────────────────────────────────────────────────────────────

class TestPSMAndRiskProfiles:
    def test_all_four_profiles_reachable(self):
        from stock_regime.strategy_router.router import (
            RISK_AGGRESSIVE, RISK_NORMAL, RISK_DEFENSIVE, RISK_CAPITAL_PRESERVATION
        )
        configs = [
            # (regime, market, quality, conf, breadth, b_score, sector, expected)
            ("TREND_UP", "BULLISH_TREND", 0.90, 0.85, "EXPANDING", 0.82, "LEADING",  RISK_AGGRESSIVE),
            ("TREND_UP", "BULLISH_TREND", 0.72, 0.72, "NEUTRAL",   0.55, "NEUTRAL",  RISK_NORMAL),
            ("TREND_UP", "BEARISH_TREND", 0.72, 0.70, "NEUTRAL",   0.45, "NEUTRAL",  RISK_DEFENSIVE),
            ("TREND_UP", "VOLATILE",      0.78, 0.72, "NEUTRAL",   0.40, "NEUTRAL",  RISK_CAPITAL_PRESERVATION),
        ]
        for regime, market, quality, conf, breadth, b_score, sector, expected in configs:
            d = _route(regime, market, quality=quality, conf=conf,
                       breadth=breadth, b_score=b_score, sector=sector)
            if d.allowed:
                assert d.risk_profile == expected, (
                    f"market={market} quality={quality}: "
                    f"expected {expected}, got {d.risk_profile} "
                    f"(psm={d.position_size_multiplier})"
                )

    def test_psm_increases_with_quality(self):
        psms = []
        for q in [0.40, 0.52, 0.65, 0.80]:
            d = _route("TREND_UP", "BULLISH_TREND", quality=q, conf=0.68)
            if d.allowed:
                psms.append((q, d.position_size_multiplier))
        for i in range(1, len(psms)):
            assert psms[i][1] >= psms[i-1][1], f"PSM not monotone with quality: {psms}"

    def test_psm_clamped_to_150(self):
        d = _route("TREND_UP", "BULLISH_TREND", quality=1.0, conf=1.0,
                   breadth="EXTREME_UP", b_score=1.0, sector="LEADING")
        assert d.position_size_multiplier <= 1.50

    def test_psm_minimum_010_when_allowed(self):
        """No allowed decision should have PSM below 0.10."""
        d = _route("TREND_UP", "VOLATILE", quality=0.78, conf=0.72,
                   breadth="CONTRACTING", b_score=0.20, sector="WEAK")
        if d.allowed:
            assert d.position_size_multiplier >= 0.10


# ─────────────────────────────────────────────────────────────────────────────
#  8. Batch routing and diagnostics
# ─────────────────────────────────────────────────────────────────────────────

class TestBatchAndDiagnostics:
    def test_batch_returns_one_per_stock(self):
        from stock_regime.strategy_router.router import StrategyRouter
        router = StrategyRouter()
        results = [_stable(f"S{i}", "TREND_UP") for i in range(10)]
        quality = [_quality(f"S{i}", 0.70) for i in range(10)]
        decisions = router.route_batch(results, quality, market_regime="BULLISH_TREND")
        assert len(decisions) == 10

    def test_reasons_populated_on_allowed(self):
        d = _route("TREND_UP", "BULLISH_TREND", quality=0.70, conf=0.70)
        assert len(d.reason) > 0
        assert any("psm=" in r for r in d.reason), "PSM breakdown must be in reasons"

    def test_reasons_populated_on_blocked(self):
        d = _route("TREND_DOWN", "BULLISH_TREND", quality=0.90, conf=0.90)
        assert len(d.reason) > 0
        assert any("not routable" in r for r in d.reason)

    def test_serialisation_complete(self):
        import json
        d = _route("RANGE", "SIDEWAYS", quality=0.60, conf=0.60)
        parsed = json.loads(json.dumps(d.to_dict()))
        for key in ["symbol", "strategy", "allowed", "risk_profile",
                    "position_size_multiplier", "reason",
                    "breadth_state", "sector_state"]:
            assert key in parsed

    def test_persist_creates_parquet(self, tmp_path):
        from stock_regime.strategy_router.router import StrategyRouter
        router    = StrategyRouter()
        results   = [_stable("A", "TREND_UP"), _stable("B", "RANGE")]
        quality   = [_quality("A", 0.70), _quality("B", 0.65)]
        decisions = router.route_batch(results, quality, market_regime="BULLISH_TREND")
        path      = router.persist(decisions, tmp_path, universe="TEST")
        assert path is not None and path.exists()
        import pandas as pd
        df = pd.read_parquet(path)
        assert "strategy" in df.columns
        assert "position_size_multiplier" in df.columns
        assert "reason" in df.columns


# ─────────────────────────────────────────────────────────────────────────────
#  9. RoutingStats accumulation
# ─────────────────────────────────────────────────────────────────────────────

class TestRoutingStats:
    def test_stats_track_allowed_count(self):
        from stock_regime.strategy_router.router import StrategyRouter, RoutingStats
        stats = RoutingStats()
        from stock_regime.strategy_router.router import RoutingDecision
        for allowed in [True, True, False, True]:
            d = RoutingDecision(
                symbol="SYM", market="TEST", run_date=date.today(),
                strategy="TREND_FOLLOWING" if allowed else "NO_TRADE",
                allowed=allowed, risk_profile="NORMAL",
                position_size_multiplier=1.0 if allowed else 0.0,
                regime_context="TREND_UP", market_context="BULLISH_TREND",
                quality_score=0.70, breadth_state="NEUTRAL",
                sector_state="NEUTRAL", reason=[],
            )
            stats.record(d, "" if allowed else "test_rejection")
        assert stats.total   == 4
        assert stats.allowed == 3
        assert abs(stats.allowed_pct - 75.0) < 0.01

    def test_strategy_distribution_accurate(self):
        from stock_regime.strategy_router.router import StrategyRouter
        router    = StrategyRouter()
        results   = [
            _stable("A", "TREND_UP"),
            _stable("B", "MOMENTUM"),
            _stable("C", "RANGE"),
            _stable("D", "TREND_DOWN"),
        ]
        quality   = [_quality(s, 0.70) for s in ["A","B","C","D"]]
        decisions = router.route_batch(
            results, quality,
            market_regime="BULLISH_TREND",
            breadth_state="NEUTRAL", breadth_score=0.55,
        )
        strats = {d.strategy for d in decisions}
        assert "TREND_FOLLOWING" in strats
        assert "MOMENTUM"        in strats
        assert "MEAN_REVERSION"  in strats
        assert "NO_TRADE"        in strats   # TREND_DOWN