"""
stock_regime/strategy_router/router.py
========================================
Adaptive strategy router with market-regime-aware risk profiles.

Changes from prior version
---------------------------
Phase 2 — Adaptive participation:
  - Removed hard Gate 3 that blocked ALL trades in adverse market regimes.
    "VOLATILE" and "UNCERTAIN" markets now use CAPITAL_PRESERVATION posture
    rather than NO_TRADE.
  - Added four named risk profiles: AGGRESSIVE | NORMAL | DEFENSIVE |
    CAPITAL_PRESERVATION. Each market regime maps to a starting posture.
  - Per-market-regime quality/confidence thresholds: bearish markets apply
    stricter thresholds for bullish strategies (only high-RS names pass),
    but do NOT globally disable participation.
  - BEARISH market: allows strongest RS/TREND_UP stocks at reduced sizing.
  - VOLATILE market: reduced participation, stricter quality gates, but not
    globally disabled.
  - SIDEWAYS market: favours RANGE/MEAN_REVERSION, penalises breakouts.

New constants:
  RISK_AGGRESSIVE, RISK_NORMAL, RISK_DEFENSIVE, RISK_CAPITAL_PRESERVATION

Routing rules summary (first-match per stock):
  TREND_UP  + BULLISH market      → TREND_FOLLOWING  NORMAL/AGGRESSIVE
  TREND_UP  + BEARISH market      → TREND_FOLLOWING  DEFENSIVE  (strict quality gate)
  TREND_UP  + VOLATILE market     → TREND_FOLLOWING  CAPITAL_PRESERVATION (very strict)
  MOMENTUM  + EXPANDING breadth   → MOMENTUM         NORMAL
  MOMENTUM  + CONTRACTING breadth → MOMENTUM         DEFENSIVE
  RANGE     + SIDEWAYS/BULLISH    → MEAN_REVERSION   NORMAL
  BREAKOUT  + EXPANDING breadth   → BREAKOUT         NORMAL
  TREND_DOWN / VOLATILE / QUIET   → NO_TRADE
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)

# ── Strategy identifiers ─────────────────────────────────────────
TREND_FOLLOWING = "TREND_FOLLOWING"
MOMENTUM        = "MOMENTUM"
MEAN_REVERSION  = "MEAN_REVERSION"
BREAKOUT        = "BREAKOUT"
NO_TRADE        = "NO_TRADE"

# ── Risk profiles (new) ──────────────────────────────────────────
RISK_AGGRESSIVE            = "AGGRESSIVE"
RISK_NORMAL                = "NORMAL"
RISK_DEFENSIVE             = "DEFENSIVE"
RISK_CAPITAL_PRESERVATION  = "CAPITAL_PRESERVATION"
RISK_OFF                   = "OFF"

# ── Market regime → base risk posture ────────────────────────────
# This is the starting point before per-stock adjustments.
_MARKET_BASE_POSTURE = {
    "BULLISH_TREND": RISK_NORMAL,
    "SIDEWAYS":      RISK_DEFENSIVE,
    "BEARISH_TREND": RISK_DEFENSIVE,
    "VOLATILE":      RISK_CAPITAL_PRESERVATION,
    "QUIET":         RISK_DEFENSIVE,
    "UNCERTAIN":     RISK_DEFENSIVE,
}

# ── Per-market quality thresholds for bullish strategies ──────────
# Bearish and volatile markets require higher quality for TREND/MOMENTUM
_MARKET_QUALITY_GATES = {
    "BULLISH_TREND": {"min_quality": 0.40, "min_confidence": 0.50},
    "SIDEWAYS":      {"min_quality": 0.45, "min_confidence": 0.52},
    "BEARISH_TREND": {"min_quality": 0.60, "min_confidence": 0.62},  # stricter
    "VOLATILE":      {"min_quality": 0.65, "min_confidence": 0.65},  # strictest
    "QUIET":         {"min_quality": 0.45, "min_confidence": 0.52},
    "UNCERTAIN":     {"min_quality": 0.50, "min_confidence": 0.55},
}

# ── Base PSM per risk profile ────────────────────────────────────
_PROFILE_BASE_PSM = {
    RISK_AGGRESSIVE:           1.25,
    RISK_NORMAL:               1.00,
    RISK_DEFENSIVE:            0.65,
    RISK_CAPITAL_PRESERVATION: 0.35,
    RISK_OFF:                  0.00,
}


@dataclass
class RoutingDecision:
    symbol:                   str
    market:                   str
    run_date:                 date
    strategy:                 str
    allowed:                  bool
    risk_profile:             str
    position_size_multiplier: float
    regime_context:           str
    market_context:           str
    quality_score:            float
    breadth_state:            str
    sector_state:             str
    reason:                   list[str]

    def to_dict(self) -> dict:
        return {
            "symbol":                   self.symbol,
            "market":                   self.market,
            "run_date":                 str(self.run_date),
            "strategy":                 self.strategy,
            "allowed":                  self.allowed,
            "risk_profile":             self.risk_profile,
            "position_size_multiplier": round(self.position_size_multiplier, 2),
            "regime_context":           self.regime_context,
            "market_context":           self.market_context,
            "quality_score":            round(self.quality_score, 4),
            "breadth_state":            self.breadth_state,
            "sector_state":             self.sector_state,
            "reason":                   self.reason,
        }


class StrategyRouter:
    """
    Adaptive deterministic strategy router.

    Key behaviour change:
    ---------------------
    Adverse market regimes (BEARISH, VOLATILE) now set a stricter quality
    gate and a reduced risk posture — they do NOT globally disable all trades.
    Only TREND_DOWN, VOLATILE (stock regime), QUIET, and UNCERTAIN stocks
    are blocked unconditionally.

    Parameters
    ----------
    min_quality_for_trade : float
        Baseline quality floor (applies in normal market conditions).
    min_confidence_for_trade : float
        Baseline confidence floor.
    min_quality_for_full_size : float
        Quality score at which PSM reaches 1.0x in NORMAL posture.
    adverse_breadth_states : list
        Breadth states that trigger a defensive shift.
    """

    def __init__(
        self,
        min_quality_for_trade:    float = 0.40,
        min_confidence_for_trade: float = 0.50,
        min_quality_for_full_size:float = 0.65,
        adverse_breadth_states:   Optional[list] = None,
    ) -> None:
        self.base_min_quality    = min_quality_for_trade
        self.base_min_confidence = min_confidence_for_trade
        self.min_quality_full    = min_quality_for_full_size
        self.adverse_breadth     = set(adverse_breadth_states or ["EXTREME_DOWN", "CONTRACTING"])

    @classmethod
    def from_config(cls, config) -> "StrategyRouter":
        rc = getattr(config, "strategy_router", None)
        if rc is None:
            return cls()
        return cls(
            min_quality_for_trade    = float(getattr(rc, "min_quality_for_trade",     0.40)),
            min_confidence_for_trade = float(getattr(rc, "min_confidence_for_trade",  0.50)),
            min_quality_for_full_size= float(getattr(rc, "min_quality_for_full_size", 0.65)),
            adverse_breadth_states   = list(getattr(rc, "adverse_breadth_states",
                                                    ["EXTREME_DOWN", "CONTRACTING"])),
        )

    def route_batch(
        self,
        stable_results:  list,
        quality_scores:  list,
        market_regime:   str         = "UNCERTAIN",
        breadth_state:   str         = "NEUTRAL",
        breadth_score:   float       = 0.50,
        sector_states:   Optional[dict[str, str]] = None,
        run_date:        Optional[date] = None,
    ) -> list[RoutingDecision]:
        """
        Route every stock to a strategy.

        Parameters
        ----------
        stable_results :
            Output of RegimeStabiliser.apply().
        quality_scores :
            Output of OpportunityQualityEngine.evaluate_batch().
        market_regime :
            Current market regime string.
        breadth_state :
            BreadthEngine state string.
        breadth_score :
            BreadthEngine composite score [0, 1] for continuous adjustments.
        sector_states :
            symbol → sector_state mapping.
        run_date :
            Date label.
        """
        run_date    = run_date or date.today()
        quality_map = {q.symbol: q.quality_score for q in quality_scores}

        # Resolve market-specific thresholds once per batch
        market_gates  = _MARKET_QUALITY_GATES.get(
            market_regime,
            _MARKET_QUALITY_GATES["UNCERTAIN"],
        )
        base_posture  = _MARKET_BASE_POSTURE.get(market_regime, RISK_DEFENSIVE)

        decisions:      list[RoutingDecision] = []
        strategy_counts: dict[str, int]       = {}

        for r in stable_results:
            if not r.is_valid():
                continue

            quality = quality_map.get(r.symbol, 0.0)
            sector  = (sector_states or {}).get(r.symbol, "NEUTRAL")
            regime  = (
                r.stable_regime.value if hasattr(r, "stable_regime")
                else r.stock_regime.value
            )
            conf = (
                r.smoothed_confidence if hasattr(r, "smoothed_confidence")
                else r.confidence
            )

            decision = self._route_one(
                symbol        = r.symbol,
                market        = r.market,
                run_date      = run_date,
                stock_regime  = regime,
                confidence    = conf,
                market_regime = market_regime,
                quality_score = quality,
                breadth_state = breadth_state,
                breadth_score = breadth_score,
                sector_state  = sector,
                market_gates  = market_gates,
                base_posture  = base_posture,
            )
            decisions.append(decision)
            strategy_counts[decision.strategy] = (
                strategy_counts.get(decision.strategy, 0) + 1
            )

        total   = len(decisions)
        allowed = sum(1 for d in decisions if d.allowed)
        logger.info(
            "StrategyRouter: %d routed | %d allowed (%.0f%%) | market=%s | breadth=%s",
            total, allowed, allowed / max(total, 1) * 100,
            market_regime, breadth_state,
        )
        for strat, count in sorted(strategy_counts.items(), key=lambda x: -x[1]):
            logger.info("  %-18s %3d  (%.0f%%)", strat, count, count / max(total, 1) * 100)

        return decisions

    def persist(
        self,
        decisions:  list[RoutingDecision],
        output_dir: str | Path,
        universe:   str = "UNKNOWN",
    ) -> Optional[Path]:
        if not decisions:
            return None
        out  = Path(output_dir) / "router" / str(decisions[0].run_date)
        out.mkdir(parents=True, exist_ok=True)
        path = out / f"{universe.lower()}_routing.parquet"
        pd.DataFrame([d.to_dict() for d in decisions]).to_parquet(
            path, engine="pyarrow", compression="snappy", index=False,
        )
        logger.info("Routing decisions [%s] → '%s'.", universe, path)
        return path

    # ──────────────────────────────────────────────────────────────
    #  Core routing
    # ──────────────────────────────────────────────────────────────

    def _route_one(
        self,
        symbol:        str,
        market:        str,
        run_date:      date,
        stock_regime:  str,
        confidence:    float,
        market_regime: str,
        quality_score: float,
        breadth_state: str,
        breadth_score: float,
        sector_state:  str,
        market_gates:  dict,
        base_posture:  str,
    ) -> RoutingDecision:

        reasons: list[str] = []

        # ── Gate 1: Regime-level blocks (unconditional) ───────────────
        # Short-side regimes and genuinely unstable stocks are never routed.
        if stock_regime in ("TREND_DOWN", "VOLATILE", "QUIET"):
            reasons.append(f"regime {stock_regime} not routable long")
            return self._no_trade(symbol, market, run_date, stock_regime,
                                  market_regime, quality_score, breadth_state,
                                  sector_state, reasons)

        if stock_regime == "UNCERTAIN":
            reasons.append("regime UNCERTAIN — insufficient signal")
            return self._no_trade(symbol, market, run_date, stock_regime,
                                  market_regime, quality_score, breadth_state,
                                  sector_state, reasons)

        # ── Gate 2: Market-context quality / confidence floor ─────────
        # Gate thresholds are STRICTER in bearish / volatile markets.
        # In a bearish market, only high-quality stocks with strong RS pass.
        min_q    = market_gates["min_quality"]
        min_conf = market_gates["min_confidence"]

        if quality_score < min_q:
            reasons.append(
                f"quality={quality_score:.2f} < {min_q:.2f} "
                f"(market={market_regime} gate)"
            )
            return self._no_trade(symbol, market, run_date, stock_regime,
                                  market_regime, quality_score, breadth_state,
                                  sector_state, reasons)

        if confidence < min_conf:
            reasons.append(
                f"confidence={confidence:.2f} < {min_conf:.2f} "
                f"(market={market_regime} gate)"
            )
            return self._no_trade(symbol, market, run_date, stock_regime,
                                  market_regime, quality_score, breadth_state,
                                  sector_state, reasons)

        # ── Strategy selection ────────────────────────────────────────
        strategy = self._select_strategy(
            stock_regime, market_regime, breadth_state, sector_state, reasons
        )

        if strategy == NO_TRADE:
            return self._no_trade(symbol, market, run_date, stock_regime,
                                  market_regime, quality_score, breadth_state,
                                  sector_state, reasons)

        # ── Risk posture + PSM ────────────────────────────────────────
        risk, psm = self._compute_psm(
            quality_score, breadth_state, breadth_score,
            sector_state, market_regime, base_posture, reasons,
        )

        return RoutingDecision(
            symbol                   = symbol,
            market                   = market,
            run_date                 = run_date,
            strategy                 = strategy,
            allowed                  = True,
            risk_profile             = risk,
            position_size_multiplier = psm,
            regime_context           = stock_regime,
            market_context           = market_regime,
            quality_score            = quality_score,
            breadth_state            = breadth_state,
            sector_state             = sector_state,
            reason                   = reasons,
        )

    # ── Strategy selection rules ─────────────────────────────────────

    def _select_strategy(
        self,
        stock_regime:  str,
        market_regime: str,
        breadth_state: str,
        sector_state:  str,
        reasons:       list[str],
    ) -> str:

        # ── TREND_FOLLOWING ──────────────────────────────────────────
        if stock_regime == "TREND_UP":
            # Bullish market — full trend following
            if market_regime in ("BULLISH_TREND", "UNCERTAIN", "QUIET"):
                reasons += [f"TREND_UP + market {market_regime}"]
                return TREND_FOLLOWING

            # Sideways market — allow trend following but risk will be DEFENSIVE
            if market_regime == "SIDEWAYS":
                reasons += ["TREND_UP in sideways — DEFENSIVE trend"]
                return TREND_FOLLOWING

            # Bearish market — only allow if stock is a genuine leader
            # (quality gate above already filtered weak stocks)
            if market_regime == "BEARISH_TREND":
                reasons += ["TREND_UP vs bearish market — DEFENSIVE sizing (RS leader)"]
                return TREND_FOLLOWING

            # Volatile market — very selective, but not blocked
            if market_regime == "VOLATILE":
                reasons += ["TREND_UP in volatile market — CAPITAL_PRESERVATION"]
                return TREND_FOLLOWING

        # ── MOMENTUM ─────────────────────────────────────────────────
        if stock_regime == "MOMENTUM":
            if breadth_state in ("EXPANDING", "EXTREME_UP", "NEUTRAL"):
                reasons += [f"MOMENTUM + breadth {breadth_state}"]
                return MOMENTUM
            # CONTRACTING breadth: allow but at reduced sizing
            if breadth_state == "CONTRACTING":
                reasons += ["MOMENTUM + contracting breadth — DEFENSIVE"]
                return MOMENTUM
            # EXTREME_DOWN: block momentum — too risky
            reasons += [f"MOMENTUM blocked: breadth {breadth_state}"]
            return NO_TRADE

        # ── MEAN_REVERSION ───────────────────────────────────────────
        if stock_regime == "RANGE":
            if market_regime in ("SIDEWAYS", "BULLISH_TREND", "QUIET", "UNCERTAIN"):
                reasons += [f"RANGE + market {market_regime} → mean reversion"]
                return MEAN_REVERSION
            # Bearish market: allow mean-reversion with reduced size
            if market_regime == "BEARISH_TREND":
                reasons += ["RANGE in bearish market — DEFENSIVE mean reversion"]
                return MEAN_REVERSION
            reasons += [f"RANGE but volatile market — too unstable"]
            return NO_TRADE

        # ── BREAKOUT ─────────────────────────────────────────────────
        if stock_regime == "BREAKOUT_SETUP":
            # Breakout requires expanding/neutral breadth AND non-bearish market
            if (breadth_state in ("EXPANDING", "NEUTRAL") and
                    sector_state in ("LEADING", "NEUTRAL") and
                    market_regime not in ("BEARISH_TREND", "VOLATILE")):
                reasons += [f"BREAKOUT_SETUP + breadth={breadth_state} + sector={sector_state}"]
                return BREAKOUT
            reasons += [f"BREAKOUT_SETUP blocked: conditions insufficient "
                       f"(breadth={breadth_state}, market={market_regime})"]
            return NO_TRADE

        reasons.append(f"no rule matched for {stock_regime}")
        return NO_TRADE

    # ── PSM computation ──────────────────────────────────────────────

    def _compute_psm(
        self,
        quality:       float,
        breadth_state: str,
        breadth_score: float,
        sector_state:  str,
        market_regime: str,
        base_posture:  str,
        reasons:       list[str],
    ) -> tuple[str, float]:
        """
        Compute position-size multiplier from base_posture and per-stock factors.

        Flow:
        1. Start from base_posture PSM (driven by market regime)
        2. Scale by quality score
        3. Apply breadth adjustment (continuous, not binary)
        4. Apply sector adjustment
        5. Clamp to [0.10, 1.50]
        6. Map to risk profile label
        """
        base_psm = _PROFILE_BASE_PSM[base_posture]
        reasons.append(f"base_posture={base_posture} (psm={base_psm:.2f})")

        # ── Quality scaling ──────────────────────────────────────────
        if quality >= self.min_quality_full:
            quality_mult = 1.0
        elif quality >= self.base_min_quality:
            quality_mult = 0.50 + 0.50 * (
                (quality - self.base_min_quality) /
                (self.min_quality_full - self.base_min_quality)
            )
        else:
            quality_mult = 0.30

        psm = base_psm * quality_mult

        # ── Breadth adjustment (continuous) ─────────────────────────
        # breadth_score [0, 1]: 0.5=neutral, >0.5=bullish, <0.5=bearish
        # Maps to multiplier: score=0.3 → 0.75x, score=0.5 → 1.0x, score=0.7 → 1.15x
        breadth_mult = 0.75 + 0.50 * breadth_score   # range [0.75, 1.25]
        if breadth_state in self.adverse_breadth:
            breadth_mult = min(breadth_mult, 0.80)
            reasons.append(f"breadth {breadth_state} → size capped at 0.80×")
        psm = psm * breadth_mult

        # ── Sector adjustment ────────────────────────────────────────
        if sector_state == "LEADING":
            psm = psm * 1.10
            reasons.append("sector LEADING → +10%")
        elif sector_state == "LAGGING":
            psm = psm * 0.75
            reasons.append("sector LAGGING → -25%")

        # ── Clamp ────────────────────────────────────────────────────
        psm = round(min(max(psm, 0.10), 1.50), 2)

        # ── Map to profile label ─────────────────────────────────────
        if psm >= 1.15:    risk = RISK_AGGRESSIVE
        elif psm >= 0.80:  risk = RISK_NORMAL
        elif psm >= 0.45:  risk = RISK_DEFENSIVE
        elif psm >= 0.15:  risk = RISK_CAPITAL_PRESERVATION
        else:              risk = RISK_OFF

        return risk, psm

    @staticmethod
    def _no_trade(
        symbol, market, run_date, stock_regime,
        market_regime, quality_score, breadth_state,
        sector_state, reasons,
    ) -> RoutingDecision:
        return RoutingDecision(
            symbol                   = symbol,
            market                   = market,
            run_date                 = run_date,
            strategy                 = NO_TRADE,
            allowed                  = False,
            risk_profile             = RISK_OFF,
            position_size_multiplier = 0.0,
            regime_context           = stock_regime,
            market_context           = market_regime,
            quality_score            = quality_score,
            breadth_state            = breadth_state,
            sector_state             = sector_state,
            reason                   = reasons,
        )