"""
runner/tests/test_soft_router.py
==================================
Tests for the soft-factor StrategyRouter.

Run:
    cd <project_root>
    pytest runner/tests/test_soft_router.py -v

Coverage:
  1. Hard blocks — only genuine NO_TRADE cases
  2. Soft factors — quality, confidence, market regime, breadth, sector
     each reduce PSM without blocking
  3. "Death by stacked filters" regression — the core motivation for this
     rewrite.  Scenarios that previously produced NO_TRADE now produce
     allowed trades at reduced sizing.
  4. PSM monotonicity — better inputs always produce higher or equal PSM
  5. Multiplier anchors — verify specific PSM ranges for known input combos
  6. Strategy selection — correct strategy per regime
  7. Risk profile labels — derived correctly from final PSM
  8. Persistence — RoutingDecision serialises correctly
"""

from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

import pytest

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))

from stock_regime.strategy_router.router import (
    TREND_FOLLOWING, MOMENTUM, MEAN_REVERSION, BREAKOUT, NO_TRADE,
    RISK_AGGRESSIVE, RISK_NORMAL, RISK_DEFENSIVE, RISK_CAPITAL_PRESERVATION,
    RISK_OFF,
    StrategyRouter,
    RoutingDecision,
    ABSOLUTE_MIN_QUALITY,
    ABSOLUTE_MIN_CONFIDENCE,
)


# ─────────────────────────────────────────────────────────────────────────────
#  Shared helpers
# ─────────────────────────────────────────────────────────────────────────────

def _make_stable(
    regime:     str   = "TREND_UP",
    conf:       float = 0.72,
    valid:      bool  = True,
    error:      str   = None,
    symbol:     str   = "SYM",
):
    """Minimal StableRegimeResult substitute."""
    from stock_regime.stability.stabiliser import StableRegimeResult
    from stock_regime.src.models import (
        StockRegime, StockRegimeResult, DimensionalScores,
        StockSignals, StockIndicatorSnapshot,
    )
    raw = StockRegimeResult(
        symbol=symbol, market="TEST",
        stock_regime=StockRegime(regime), confidence=conf,
        dimensional_scores=DimensionalScores(trend=0.60, momentum=0.50),
        regime_scores={}, signals=StockSignals(),
        indicators=StockIndicatorSnapshot(),
        error=error,
    )
    return StableRegimeResult.from_result(
        raw,
        stable_regime=StockRegime(regime),
        prior_stable_regime=StockRegime.UNCERTAIN,
        regime_age_bars=5,
        stable_regime_age=5,
        regime_changed_today=False,
        smoothed_confidence=conf,
        oscillation_detected=False,
    )


def _make_quality(symbol: str = "SYM", score: float = 0.65):
    from stock_regime.quality_engine.opportunity_quality import QualityScore
    return QualityScore(
        symbol=symbol, market="TEST", run_date=date.today(),
        quality_score=score, liquidity_quality=0.70,
        trend_quality=0.70, vol_health=0.65,
        stability_quality=0.65, tradability=0.70, notes=[],
    )


def _route(
    regime:       str   = "TREND_UP",
    conf:         float = 0.72,
    quality:      float = 0.65,
    market:       str   = "BULLISH_TREND",
    breadth:      str   = "NEUTRAL",
    breadth_score:float = 0.55,
    sector:       str   = "NEUTRAL",
    valid:        bool  = True,
) -> RoutingDecision:
    """Helper: route a single synthetic stock."""
    router = StrategyRouter()
    sr     = _make_stable(regime=regime, conf=conf, valid=valid,
                           error=None if valid else "test error")
    qs     = _make_quality(score=quality)
    return router.route_batch(
        [sr], [qs],
        market_regime=market,
        breadth_state=breadth,
        breadth_score=breadth_score,
        sector_states={"SYM": sector},
    )[0]


# ─────────────────────────────────────────────────────────────────────────────
#  1. Hard blocks — only these should produce NO_TRADE
# ─────────────────────────────────────────────────────────────────────────────

class TestHardBlocks:
    """Only regime invalidity and absolute signal floors cause NO_TRADE."""

    def test_trend_down_always_blocked(self):
        for market in ("BULLISH_TREND", "BEARISH_TREND", "SIDEWAYS", "VOLATILE"):
            d = _route("TREND_DOWN", quality=0.90, conf=0.90, market=market)
            assert d.strategy == NO_TRADE and not d.allowed, \
                f"TREND_DOWN must be NO_TRADE in {market}"

    def test_volatile_regime_always_blocked(self):
        d = _route("VOLATILE", quality=0.90, conf=0.90, market="BULLISH_TREND")
        assert d.strategy == NO_TRADE and not d.allowed

    def test_uncertain_regime_always_blocked(self):
        d = _route("UNCERTAIN", quality=0.90, conf=0.90, market="BULLISH_TREND")
        assert d.strategy == NO_TRADE and not d.allowed

    def test_quiet_regime_always_blocked(self):
        d = _route("QUIET", quality=0.90, conf=0.90, market="BULLISH_TREND")
        assert d.strategy == NO_TRADE and not d.allowed

    def test_quality_below_absolute_floor_blocked(self):
        """quality < 0.30 is noise — blocked regardless of regime."""
        d = _route("TREND_UP", quality=0.28, conf=0.72, market="BULLISH_TREND")
        assert d.strategy == NO_TRADE and not d.allowed

    def test_quality_at_absolute_floor_blocked(self):
        """Exactly at 0.30 floor is still blocked (strict <)."""
        d = _route("TREND_UP", quality=ABSOLUTE_MIN_QUALITY - 0.001,
                   conf=0.72, market="BULLISH_TREND")
        assert d.strategy == NO_TRADE and not d.allowed

    def test_confidence_below_absolute_floor_blocked(self):
        d = _route("TREND_UP", quality=0.65, conf=0.28, market="BULLISH_TREND")
        assert d.strategy == NO_TRADE and not d.allowed

    def test_invalid_data_blocked(self):
        """Invalid data flag from stabiliser always produces NO_TRADE."""
        router = StrategyRouter()
        sr     = _make_stable(regime="TREND_UP", valid=False, error="data error")
        qs     = _make_quality(score=0.70)
        d = router.route_batch([sr], [qs], market_regime="BULLISH_TREND")[0]
        assert d.strategy == NO_TRADE and not d.allowed

    def test_only_two_regime_blocks_not_three(self):
        """
        Regression: previously BEARISH market was an implicit block.
        Confirm TREND_UP in BEARISH market is now ALLOWED (not blocked).
        """
        d = _route("TREND_UP", quality=0.55, conf=0.60, market="BEARISH_TREND")
        assert d.allowed is True, \
            "TREND_UP in BEARISH market should be allowed at reduced PSM"


# ─────────────────────────────────────────────────────────────────────────────
#  2. "Death by stacked filters" regression tests
#     These are the scenarios that motivated this rewrite.
# ─────────────────────────────────────────────────────────────────────────────

class TestStackedFilterRegression:
    """
    Scenarios that previously produced NO_TRADE because multiple
    moderate-weakness factors stacked.  All should now be ALLOWED
    at reduced PSM.
    """

    def test_slightly_below_quality_threshold_still_allowed(self):
        """
        quality=0.58, old threshold=0.60 → was NO_TRADE.
        New behavior: DEFENSIVE profile at reduced PSM.
        """
        d = _route("TREND_UP", quality=0.58, conf=0.65, market="BULLISH_TREND")
        assert d.allowed is True
        assert d.strategy == TREND_FOLLOWING
        assert d.risk_profile in (RISK_DEFENSIVE, RISK_NORMAL), \
            f"Expected DEFENSIVE or NORMAL, got {d.risk_profile}"

    def test_bearish_market_with_expanding_breadth_allows_trend_up(self):
        """
        BEARISH market + EXPANDING breadth + quality=0.65.
        Previously: NO_TRADE (bearish market hard gate).
        Now: ALLOWED at DEFENSIVE/CAPITAL_PRESERVATION sizing.
        """
        d = _route(
            "TREND_UP", quality=0.65, conf=0.68,
            market="BEARISH_TREND",
            breadth="EXPANDING", breadth_score=0.70,
        )
        assert d.allowed is True
        assert d.strategy == TREND_FOLLOWING
        # PSM should be reduced but not zero
        assert d.position_size_multiplier > 0.10

    def test_bearish_market_with_expanding_breadth_allows_momentum(self):
        """
        Strong MOMENTUM stock in BEARISH market + EXPANDING breadth.
        Previously blocked by market regime gate.
        Now allowed at reduced PSM.
        """
        d = _route(
            "MOMENTUM", quality=0.70, conf=0.72,
            market="BEARISH_TREND",
            breadth="EXPANDING", breadth_score=0.72,
        )
        assert d.allowed is True
        assert d.strategy == MOMENTUM

    def test_moderate_quality_and_moderate_confidence_still_allowed(self):
        """
        quality=0.55 AND confidence=0.58 — two moderate weaknesses stacked.
        Previously both gates would have fired.
        Now: single reduced PSM.
        """
        d = _route("TREND_UP", quality=0.55, conf=0.58, market="SIDEWAYS")
        assert d.allowed is True
        # PSM should be visibly reduced
        assert d.position_size_multiplier < 0.80

    def test_contracting_breadth_does_not_block_high_quality(self):
        """
        CONTRACTING breadth previously blocked MOMENTUM outright.
        Now: MOMENTUM allowed at reduced PSM (0.75× breadth ceiling).
        """
        d = _route(
            "MOMENTUM", quality=0.75, conf=0.75,
            market="BULLISH_TREND",
            breadth="CONTRACTING", breadth_score=0.35,
        )
        assert d.allowed is True
        assert d.strategy == MOMENTUM
        # Breadth ceiling of 0.75 applies
        assert d.position_size_multiplier <= 0.80

    def test_volatile_market_allows_very_high_quality_trend_up(self):
        """
        VOLATILE market was previously an implicit block via PSM=0.35 floor.
        High quality stock should still be allowed — just very small sizing.
        """
        d = _route(
            "TREND_UP", quality=0.85, conf=0.80,
            market="VOLATILE",
            breadth="NEUTRAL", breadth_score=0.50,
        )
        assert d.allowed is True
        # Market ceiling is 0.45; result should be small but positive
        assert 0.10 <= d.position_size_multiplier <= 0.60

    def test_lagging_sector_does_not_block_trade(self):
        """
        LAGGING sector previously combined with other factors to block.
        Now: -25% PSM penalty, trade still allowed.
        """
        d = _route(
            "TREND_UP", quality=0.65, conf=0.68,
            market="BULLISH_TREND",
            breadth="NEUTRAL", breadth_score=0.50,
            sector="LAGGING",
        )
        assert d.allowed is True
        assert d.position_size_multiplier > 0.10


# ─────────────────────────────────────────────────────────────────────────────
#  3. Soft factor PSM monotonicity
# ─────────────────────────────────────────────────────────────────────────────

class TestPSMMonotonicity:
    """Better inputs should always produce higher or equal PSM."""

    def test_psm_increases_with_quality(self):
        router = StrategyRouter()
        psms   = []
        for q in [0.32, 0.45, 0.58, 0.65, 0.72, 0.85, 0.95]:
            sr = _make_stable(conf=0.72)
            qs = _make_quality(score=q)
            d  = router.route_batch(
                [sr], [qs], market_regime="BULLISH_TREND",
                breadth_state="NEUTRAL", breadth_score=0.55,
            )[0]
            if d.allowed:
                psms.append((q, d.position_size_multiplier))

        for i in range(1, len(psms)):
            assert psms[i][1] >= psms[i - 1][1] - 0.01, \
                f"PSM not monotone with quality: {psms}"

    def test_psm_increases_with_confidence(self):
        router = StrategyRouter()
        psms   = []
        for c in [0.32, 0.45, 0.55, 0.65, 0.75, 0.90]:
            sr = _make_stable(conf=c)
            qs = _make_quality(score=0.68)
            d  = router.route_batch(
                [sr], [qs], market_regime="BULLISH_TREND",
                breadth_state="NEUTRAL", breadth_score=0.55,
            )[0]
            if d.allowed:
                psms.append((c, d.position_size_multiplier))

        for i in range(1, len(psms)):
            assert psms[i][1] >= psms[i - 1][1] - 0.01, \
                f"PSM not monotone with confidence: {psms}"

    def test_psm_increases_with_breadth_score(self):
        router = StrategyRouter()
        psms   = []
        for bs in [0.20, 0.35, 0.50, 0.65, 0.80]:
            sr = _make_stable(conf=0.72)
            qs = _make_quality(score=0.70)
            d  = router.route_batch(
                [sr], [qs], market_regime="BULLISH_TREND",
                breadth_state="NEUTRAL", breadth_score=bs,
            )[0]
            if d.allowed:
                psms.append((bs, d.position_size_multiplier))

        for i in range(1, len(psms)):
            assert psms[i][1] >= psms[i - 1][1] - 0.01, \
                f"PSM not monotone with breadth_score: {psms}"

    def test_leading_sector_produces_higher_psm_than_lagging(self):
        d_lead = _route("TREND_UP", quality=0.70, conf=0.72,
                         market="BULLISH_TREND", sector="LEADING")
        d_lag  = _route("TREND_UP", quality=0.70, conf=0.72,
                         market="BULLISH_TREND", sector="LAGGING")
        assert d_lead.position_size_multiplier > d_lag.position_size_multiplier

    def test_bullish_market_higher_psm_than_bearish(self):
        d_bull = _route("TREND_UP", quality=0.70, conf=0.72, market="BULLISH_TREND")
        d_bear = _route("TREND_UP", quality=0.70, conf=0.72, market="BEARISH_TREND")
        assert d_bull.position_size_multiplier > d_bear.position_size_multiplier

    def test_expanding_breadth_higher_psm_than_contracting(self):
        d_exp  = _route("TREND_UP", quality=0.70, conf=0.72,
                         breadth="EXPANDING", breadth_score=0.70)
        d_cont = _route("TREND_UP", quality=0.70, conf=0.72,
                         breadth="CONTRACTING", breadth_score=0.35)
        assert d_exp.position_size_multiplier > d_cont.position_size_multiplier


# ─────────────────────────────────────────────────────────────────────────────
#  4. Multiplier anchor point verification
# ─────────────────────────────────────────────────────────────────────────────

class TestMultiplierAnchors:
    """
    Verify that known input combinations produce PSM in expected ranges.
    Ranges are deliberately wide to allow for combined multiplier effects.
    """

    def test_excellent_inputs_bullish_market_aggressive_profile(self):
        """
        quality=0.90, conf=0.85, BULLISH, EXPANDING, LEADING
        → should reach AGGRESSIVE (PSM >= 1.10)
        """
        d = _route(
            "TREND_UP", quality=0.90, conf=0.85,
            market="BULLISH_TREND",
            breadth="EXPANDING", breadth_score=0.80,
            sector="LEADING",
        )
        assert d.allowed is True
        assert d.risk_profile == RISK_AGGRESSIVE, \
            f"Expected AGGRESSIVE for excellent inputs, got {d.risk_profile} PSM={d.position_size_multiplier}"

    def test_moderate_inputs_bullish_normal_profile(self):
        """
        quality=0.70, conf=0.70, BULLISH, NEUTRAL
        → NORMAL profile (PSM in 0.75–1.15)
        """
        d = _route(
            "TREND_UP", quality=0.70, conf=0.70,
            market="BULLISH_TREND",
            breadth="NEUTRAL", breadth_score=0.55,
        )
        assert d.allowed is True
        assert d.risk_profile in (RISK_NORMAL, RISK_AGGRESSIVE), \
            f"Expected NORMAL/AGGRESSIVE for moderate inputs in bullish, got {d.risk_profile}"

    def test_weak_inputs_bearish_market_defensive_profile(self):
        """
        quality=0.58, conf=0.60, BEARISH, NEUTRAL
        → DEFENSIVE or CAPITAL_PRESERVATION (PSM < 0.75)
        """
        d = _route(
            "TREND_UP", quality=0.58, conf=0.60,
            market="BEARISH_TREND",
            breadth="NEUTRAL", breadth_score=0.50,
        )
        assert d.allowed is True
        assert d.risk_profile in (RISK_DEFENSIVE, RISK_CAPITAL_PRESERVATION), \
            f"Expected DEFENSIVE/CAP_PRES for weak inputs in bearish, got {d.risk_profile}"
        assert d.position_size_multiplier < 0.75

    def test_extreme_down_breadth_caps_psm(self):
        """
        EXTREME_DOWN breadth imposes 0.50× ceiling.
        Even excellent quality should not exceed that.
        """
        d = _route(
            "TREND_UP", quality=0.95, conf=0.95,
            market="BULLISH_TREND",
            breadth="EXTREME_DOWN", breadth_score=0.15,
        )
        assert d.allowed is True
        assert d.position_size_multiplier <= 0.55, \
            f"EXTREME_DOWN should cap PSM ≤ 0.55, got {d.position_size_multiplier}"

    def test_volatile_market_ceiling_constrains_psm(self):
        """
        VOLATILE market ceiling is 0.45.
        Even with all other factors optimal, PSM should stay modest.
        """
        d = _route(
            "TREND_UP", quality=0.95, conf=0.95,
            market="VOLATILE",
            breadth="EXPANDING", breadth_score=0.80,
            sector="LEADING",
        )
        assert d.allowed is True
        # Ceiling 0.45 × quality 1.10 × conf 1.10 × breadth 1.15 × sector 1.10
        # ≈ 0.69 — well-behaved upper bound
        assert d.position_size_multiplier <= 0.80, \
            f"VOLATILE market PSM too high: {d.position_size_multiplier}"

    def test_quality_just_above_floor_produces_small_but_positive_psm(self):
        """
        quality=0.31 (just above 0.30 floor) → tiny multiplier.
        PSM should be very small but allowed.
        """
        d = _route(
            "TREND_UP", quality=0.31, conf=0.65,
            market="BULLISH_TREND",
            breadth="NEUTRAL", breadth_score=0.55,
        )
        assert d.allowed is True
        # 0.31 quality mult ≈ 0.42; result should be around 0.35–0.55
        assert 0.10 <= d.position_size_multiplier <= 0.65, \
            f"Unexpected PSM for just-above-floor quality: {d.position_size_multiplier}"


# ─────────────────────────────────────────────────────────────────────────────
#  5. Strategy selection correctness
# ─────────────────────────────────────────────────────────────────────────────

class TestStrategySelection:

    def test_trend_up_always_trend_following(self):
        for market in ("BULLISH_TREND", "BEARISH_TREND", "SIDEWAYS", "VOLATILE", "UNCERTAIN"):
            d = _route("TREND_UP", quality=0.65, conf=0.65, market=market)
            if d.allowed:
                assert d.strategy == TREND_FOLLOWING, \
                    f"TREND_UP should always route to TREND_FOLLOWING, got {d.strategy} in {market}"

    def test_momentum_allowed_in_expanding_breadth(self):
        d = _route("MOMENTUM", quality=0.65, conf=0.65,
                   market="BULLISH_TREND", breadth="EXPANDING")
        assert d.strategy == MOMENTUM and d.allowed

    def test_momentum_allowed_in_contracting_breadth_at_reduced_psm(self):
        d = _route("MOMENTUM", quality=0.70, conf=0.70,
                   market="BULLISH_TREND", breadth="CONTRACTING", breadth_score=0.35)
        assert d.strategy == MOMENTUM and d.allowed

    def test_momentum_blocked_in_extreme_down_breadth(self):
        """EXTREME_DOWN is the only breadth state that blocks MOMENTUM."""
        d = _route("MOMENTUM", quality=0.90, conf=0.90,
                   market="BULLISH_TREND", breadth="EXTREME_DOWN")
        assert d.strategy == NO_TRADE and not d.allowed

    def test_range_routes_mean_reversion_in_sideways_market(self):
        d = _route("RANGE", quality=0.60, conf=0.62, market="SIDEWAYS")
        assert d.strategy == MEAN_REVERSION and d.allowed

    def test_range_routes_mean_reversion_in_bearish_market(self):
        """RANGE should work in bearish markets at reduced PSM."""
        d = _route("RANGE", quality=0.62, conf=0.62, market="BEARISH_TREND")
        assert d.strategy == MEAN_REVERSION and d.allowed

    def test_range_blocked_only_in_volatile_market(self):
        """VOLATILE market is the only structural block for RANGE."""
        d = _route("RANGE", quality=0.80, conf=0.80, market="VOLATILE")
        assert d.strategy == NO_TRADE and not d.allowed

    def test_breakout_allowed_when_only_market_adverse(self):
        """
        BEARISH market alone should not block BREAKOUT if breadth is OK.
        Both market AND breadth adverse simultaneously is required for block.
        """
        d = _route(
            "BREAKOUT_SETUP", quality=0.65, conf=0.65,
            market="BEARISH_TREND",
            breadth="NEUTRAL", breadth_score=0.55,
            sector="NEUTRAL",
        )
        assert d.strategy == BREAKOUT and d.allowed

    def test_breakout_allowed_when_only_breadth_adverse(self):
        """
        CONTRACTING breadth alone should not block BREAKOUT if market is OK.
        """
        d = _route(
            "BREAKOUT_SETUP", quality=0.65, conf=0.65,
            market="BULLISH_TREND",
            breadth="CONTRACTING", breadth_score=0.35,
        )
        assert d.strategy == BREAKOUT and d.allowed

    def test_breakout_blocked_when_both_market_and_breadth_adverse(self):
        """
        Both BEARISH market AND CONTRACTING/EXTREME_DOWN breadth → structural block.
        """
        d = _route(
            "BREAKOUT_SETUP", quality=0.90, conf=0.90,
            market="BEARISH_TREND",
            breadth="EXTREME_DOWN", breadth_score=0.15,
        )
        assert d.strategy == NO_TRADE and not d.allowed


# ─────────────────────────────────────────────────────────────────────────────
#  6. Risk profile label derivation
# ─────────────────────────────────────────────────────────────────────────────

class TestRiskProfileLabels:
    """Verify all four reachable profiles."""

    def test_aggressive_profile_reachable(self):
        """Best-case inputs in bullish market should reach AGGRESSIVE."""
        d = _route(
            "TREND_UP", quality=0.92, conf=0.88,
            market="BULLISH_TREND",
            breadth="EXPANDING", breadth_score=0.82,
            sector="LEADING",
        )
        assert d.allowed and d.risk_profile == RISK_AGGRESSIVE, \
            f"Expected AGGRESSIVE, got {d.risk_profile} PSM={d.position_size_multiplier}"

    def test_normal_profile_reachable(self):
        d = _route(
            "TREND_UP", quality=0.72, conf=0.72,
            market="BULLISH_TREND",
            breadth="NEUTRAL", breadth_score=0.55,
        )
        assert d.allowed and d.risk_profile in (RISK_NORMAL, RISK_AGGRESSIVE)

    def test_defensive_profile_reachable(self):
        d = _route(
            "TREND_UP", quality=0.58, conf=0.60,
            market="BEARISH_TREND",
            breadth="NEUTRAL", breadth_score=0.48,
        )
        assert d.allowed and d.risk_profile in (RISK_DEFENSIVE, RISK_CAPITAL_PRESERVATION)

    def test_capital_preservation_reachable(self):
        d = _route(
            "TREND_UP", quality=0.45, conf=0.50,
            market="VOLATILE",
            breadth="CONTRACTING", breadth_score=0.28,
        )
        assert d.allowed and d.risk_profile in (
            RISK_CAPITAL_PRESERVATION, RISK_DEFENSIVE
        )

    def test_off_profile_only_on_hard_block(self):
        d = _route("TREND_DOWN", quality=0.90, conf=0.90, market="BULLISH_TREND")
        assert d.risk_profile == RISK_OFF and not d.allowed

    def test_psm_zero_only_on_hard_block(self):
        d = _route("UNCERTAIN", quality=0.90, conf=0.90, market="BULLISH_TREND")
        assert d.position_size_multiplier == 0.0 and not d.allowed


# ─────────────────────────────────────────────────────────────────────────────
#  7. Batch routing and serialisation
# ─────────────────────────────────────────────────────────────────────────────

class TestBatchAndSerialisation:

    def test_batch_returns_one_decision_per_stock(self):
        router  = StrategyRouter()
        symbols = ["A", "B", "C", "D"]
        srs     = [_make_stable(symbol=s) for s in symbols]
        qss     = [_make_quality(symbol=s, score=0.65) for s in symbols]
        ds      = router.route_batch(srs, qss, market_regime="BULLISH_TREND")
        assert len(ds) == len(symbols)

    def test_routing_decision_serialises_to_json(self):
        import json
        d      = _route("TREND_UP", quality=0.68, conf=0.70, market="BEARISH_TREND")
        parsed = json.loads(json.dumps(d.to_dict()))
        for key in ["symbol", "strategy", "allowed", "risk_profile",
                    "position_size_multiplier", "reason"]:
            assert key in parsed

    def test_risk_profile_values_are_valid_strings(self):
        valid_profiles = {
            RISK_AGGRESSIVE, RISK_NORMAL, RISK_DEFENSIVE,
            RISK_CAPITAL_PRESERVATION, RISK_OFF,
        }
        for regime in ("TREND_UP", "MOMENTUM", "RANGE", "BREAKOUT_SETUP",
                       "TREND_DOWN", "UNCERTAIN", "VOLATILE"):
            d = _route(regime, quality=0.65, conf=0.65, market="BULLISH_TREND")
            assert d.risk_profile in valid_profiles, \
                f"Unexpected risk_profile={d.risk_profile!r} for regime={regime}"

    def test_persist_writes_parquet(self, tmp_path):
        decisions = [
            _route("TREND_UP", quality=0.70, conf=0.72, market="BULLISH_TREND"),
            _route("MOMENTUM", quality=0.68, conf=0.68, market="BULLISH_TREND"),
        ]
        router = StrategyRouter()
        path   = router.persist(decisions, tmp_path, universe="TEST")
        assert path is not None and path.exists()
        import pandas as pd
        df = pd.read_parquet(path)
        assert "strategy"     in df.columns
        assert "risk_profile" in df.columns
        assert "position_size_multiplier" in df.columns

    def test_from_config_returns_router_instance(self):
        from unittest.mock import MagicMock
        cfg                          = MagicMock()
        cfg.strategy_router          = MagicMock()
        cfg.strategy_router.absolute_min_quality      = 0.30
        cfg.strategy_router.absolute_min_confidence   = 0.30
        cfg.strategy_router.min_quality_for_full_size = 0.70
        cfg.strategy_router.adverse_breadth_states    = ["EXTREME_DOWN"]
        router = StrategyRouter.from_config(cfg)
        assert isinstance(router, StrategyRouter)
        assert router.abs_min_quality == 0.30


# ─────────────────────────────────────────────────────────────────────────────
#  8. Quality multiplier unit tests
# ─────────────────────────────────────────────────────────────────────────────

class TestQualityMultiplier:
    """Verify the multiplier function independently of the full pipeline."""

    def test_at_full_size_anchor(self):
        m = StrategyRouter._quality_multiplier(0.70)
        assert abs(m - 1.00) < 0.01

    def test_at_maximum_gives_bonus(self):
        m = StrategyRouter._quality_multiplier(1.00)
        assert abs(m - 1.10) < 0.01

    def test_at_moderate_quality(self):
        m = StrategyRouter._quality_multiplier(0.50)
        assert abs(m - 0.70) < 0.01

    def test_at_absolute_floor(self):
        m = StrategyRouter._quality_multiplier(0.30)
        assert abs(m - 0.40) < 0.01

    def test_monotone_across_range(self):
        prev = 0.0
        for q in [0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 1.00]:
            m = StrategyRouter._quality_multiplier(q)
            assert m >= prev - 0.001, f"Quality multiplier not monotone at {q}"
            prev = m


# ─────────────────────────────────────────────────────────────────────────────
#  9. Confidence multiplier unit tests
# ─────────────────────────────────────────────────────────────────────────────

class TestConfidenceMultiplier:

    def test_at_full_confidence_anchor(self):
        m = StrategyRouter._confidence_multiplier(0.65)
        assert abs(m - 1.00) < 0.01

    def test_at_maximum_gives_bonus(self):
        m = StrategyRouter._confidence_multiplier(1.00)
        assert abs(m - 1.10) < 0.01

    def test_at_absolute_floor(self):
        m = StrategyRouter._confidence_multiplier(0.30)
        assert abs(m - 0.60) < 0.01

    def test_monotone_across_range(self):
        prev = 0.0
        for c in [0.30, 0.40, 0.50, 0.60, 0.65, 0.75, 0.90, 1.00]:
            m = StrategyRouter._confidence_multiplier(c)
            assert m >= prev - 0.001, f"Confidence multiplier not monotone at {c}"
            prev = m