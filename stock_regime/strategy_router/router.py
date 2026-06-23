"""
stock_regime/strategy_router/router.py
========================================
True strategy allocation engine.

Key improvements over prior version
--------------------------------------
1. Per-strategy quality gates
   Each strategy has its own min_quality / min_confidence per market regime.
   Mean-reversion needs a much lower gate than trend-following in a bear market.

2. Expanded routing — all four tradeable strategies actively used:
   TREND_UP       → TREND_FOLLOWING   (all market regimes, posture adjusts)
   MOMENTUM       → MOMENTUM          (only EXTREME_DOWN breadth blocks it)
   RANGE          → MEAN_REVERSION    (works in most markets, blocked in VOLATILE)
   BREAKOUT_SETUP → BREAKOUT          (blocked in BEARISH/VOLATILE only)
   TREND_DOWN     → NO_TRADE          (unconditional)
   VOLATILE       → NO_TRADE          (unconditional)
   UNCERTAIN      → NO_TRADE          (unconditional)
   QUIET          → NO_TRADE          (no edge in quiet regime)

3. Strategy-market posture overrides
   Mean-reversion in SIDEWAYS → NORMAL posture (it is the preferred strategy)
   Trend-following in SIDEWAYS → DEFENSIVE posture
   Anything in VOLATILE → CAPITAL_PRESERVATION

4. Breadth is continuous, not binary
   breadth_score [0,1] feeds PSM multiplier directly.
   Only EXTREME_DOWN hard-blocks MOMENTUM.

5. Five-level sector state with graduated PSM multipliers.

6. Full structured diagnostics per batch:
   strategy distribution, rejection reasons, regime→strategy breakdown.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)

# ── Strategy identifiers ──────────────────────────────────────
TREND_FOLLOWING = "TREND_FOLLOWING"
MOMENTUM        = "MOMENTUM"
MEAN_REVERSION  = "MEAN_REVERSION"
BREAKOUT        = "BREAKOUT"
NO_TRADE        = "NO_TRADE"

# ── Risk profiles ─────────────────────────────────────────────
RISK_AGGRESSIVE           = "AGGRESSIVE"
RISK_NORMAL               = "NORMAL"
RISK_DEFENSIVE            = "DEFENSIVE"
RISK_CAPITAL_PRESERVATION = "CAPITAL_PRESERVATION"
RISK_OFF                  = "OFF"

# ── Market regime → default base posture ─────────────────────
_MARKET_BASE_POSTURE = {
    "BULLISH_TREND": RISK_NORMAL,
    "SIDEWAYS":      RISK_DEFENSIVE,
    "BEARISH_TREND": RISK_DEFENSIVE,
    "VOLATILE":      RISK_CAPITAL_PRESERVATION,
    "QUIET":         RISK_DEFENSIVE,
    "UNCERTAIN":     RISK_DEFENSIVE,
}

# ── Per-strategy quality / confidence gates per market regime ─
# Lower gates for mean-reversion (works in range/sideways markets)
# Higher gates for trend/momentum in adverse markets
_STRATEGY_GATES = {
    TREND_FOLLOWING: {
        "BULLISH_TREND": {"min_quality": 0.40, "min_confidence": 0.50},
        "SIDEWAYS":      {"min_quality": 0.45, "min_confidence": 0.52},
        "BEARISH_TREND": {"min_quality": 0.60, "min_confidence": 0.62},
        "VOLATILE":      {"min_quality": 0.65, "min_confidence": 0.65},
        "QUIET":         {"min_quality": 0.42, "min_confidence": 0.50},
        "UNCERTAIN":     {"min_quality": 0.50, "min_confidence": 0.55},
    },
    MOMENTUM: {
        "BULLISH_TREND": {"min_quality": 0.42, "min_confidence": 0.52},
        "SIDEWAYS":      {"min_quality": 0.48, "min_confidence": 0.54},
        "BEARISH_TREND": {"min_quality": 0.62, "min_confidence": 0.64},
        "VOLATILE":      {"min_quality": 0.65, "min_confidence": 0.65},
        "QUIET":         {"min_quality": 0.42, "min_confidence": 0.50},
        "UNCERTAIN":     {"min_quality": 0.50, "min_confidence": 0.55},
    },
    MEAN_REVERSION: {
        # Preferred in sideways → very low gate
        "BULLISH_TREND": {"min_quality": 0.38, "min_confidence": 0.48},
        "SIDEWAYS":      {"min_quality": 0.36, "min_confidence": 0.46},
        "BEARISH_TREND": {"min_quality": 0.42, "min_confidence": 0.50},
        # Blocked in VOLATILE via _select_strategy, gate here is belt+suspenders
        "VOLATILE":      {"min_quality": 0.80, "min_confidence": 0.80},
        "QUIET":         {"min_quality": 0.36, "min_confidence": 0.46},
        "UNCERTAIN":     {"min_quality": 0.42, "min_confidence": 0.50},
    },
    BREAKOUT: {
        "BULLISH_TREND": {"min_quality": 0.50, "min_confidence": 0.55},
        "SIDEWAYS":      {"min_quality": 0.55, "min_confidence": 0.58},
        # Blocked via _select_strategy; 0.99 here as safety net
        "BEARISH_TREND": {"min_quality": 0.99, "min_confidence": 0.99},
        "VOLATILE":      {"min_quality": 0.99, "min_confidence": 0.99},
        "QUIET":         {"min_quality": 0.55, "min_confidence": 0.58},
        "UNCERTAIN":     {"min_quality": 0.55, "min_confidence": 0.58},
    },
}

# ── Strategy-market posture overrides ─────────────────────────
# Overrides _MARKET_BASE_POSTURE when a specific strategy+market combo applies.
_STRATEGY_MARKET_POSTURE = {
    (MEAN_REVERSION,  "SIDEWAYS"):      RISK_NORMAL,       # preferred → full sizing
    (TREND_FOLLOWING, "SIDEWAYS"):      RISK_DEFENSIVE,    # against market character
    (TREND_FOLLOWING, "BEARISH_TREND"): RISK_DEFENSIVE,
    (MEAN_REVERSION,  "BEARISH_TREND"): RISK_DEFENSIVE,
    (TREND_FOLLOWING, "VOLATILE"):      RISK_CAPITAL_PRESERVATION,
    (MOMENTUM,        "VOLATILE"):      RISK_CAPITAL_PRESERVATION,
    (BREAKOUT,        "VOLATILE"):      RISK_CAPITAL_PRESERVATION,
    (MEAN_REVERSION,  "VOLATILE"):      RISK_CAPITAL_PRESERVATION,
}

# ── Base PSM per risk profile ─────────────────────────────────
_PROFILE_BASE_PSM = {
    RISK_AGGRESSIVE:           1.25,
    RISK_NORMAL:               1.00,
    RISK_DEFENSIVE:            0.65,
    RISK_CAPITAL_PRESERVATION: 0.35,
    RISK_OFF:                  0.00,
}

# ── 5-level sector PSM multipliers ───────────────────────────
_SECTOR_STATE_MULT = {
    "LEADING":   1.15,
    "STRONG":    1.07,
    "NEUTRAL":   1.00,
    "WEAKENING": 0.80,
    "WEAK":      0.65,
    "UNKNOWN":   1.00,
}

# Breadth states that block MOMENTUM entirely
_MOMENTUM_KILL_BREADTH = {"EXTREME_DOWN"}

# Sector states that allow breakout trades
_BREAKOUT_OK_SECTORS = {"LEADING", "STRONG", "NEUTRAL", "UNKNOWN"}


# ─────────────────────────────────────────────────────────────
#  Output dataclass
# ─────────────────────────────────────────────────────────────

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


# ─────────────────────────────────────────────────────────────
#  Routing statistics container
# ─────────────────────────────────────────────────────────────

@dataclass
class RoutingStats:
    """Collected per batch for structured diagnostic logging."""
    total:             int  = 0
    allowed:           int  = 0
    strategy_dist:     dict = field(default_factory=dict)
    rejection_reasons: dict = field(default_factory=dict)
    regime_to_strategy:dict = field(default_factory=dict)

    @property
    def allowed_pct(self) -> float:
        return self.allowed / max(self.total, 1) * 100

    def record(self, decision: RoutingDecision, rej_key: str = "") -> None:
        self.total += 1
        self.strategy_dist[decision.strategy] = (
            self.strategy_dist.get(decision.strategy, 0) + 1
        )
        if decision.allowed:
            self.allowed += 1
        elif rej_key:
            self.rejection_reasons[rej_key] = (
                self.rejection_reasons.get(rej_key, 0) + 1
            )
        regime = decision.regime_context
        if regime not in self.regime_to_strategy:
            self.regime_to_strategy[regime] = {}
        self.regime_to_strategy[regime][decision.strategy] = (
            self.regime_to_strategy[regime].get(decision.strategy, 0) + 1
        )


# ─────────────────────────────────────────────────────────────
#  Strategy Router
# ─────────────────────────────────────────────────────────────

class StrategyRouter:
    """
    True strategy allocation engine with per-strategy quality gates,
    full market/breadth/sector context awareness, and structured diagnostics.
    """

    def __init__(
        self,
        min_quality_for_trade:    float = 0.36,
        min_confidence_for_trade: float = 0.46,
        min_quality_for_full_size:float = 0.65,
        adverse_breadth_states:   Optional[list] = None,
    ) -> None:
        self.base_min_quality    = min_quality_for_trade
        self.base_min_confidence = min_confidence_for_trade
        self.min_quality_full    = min_quality_for_full_size
        self.adverse_breadth     = set(
            adverse_breadth_states or ["EXTREME_DOWN", "CONTRACTING"]
        )

    @classmethod
    def from_config(cls, config) -> "StrategyRouter":
        rc = getattr(config, "strategy_router", None)
        if rc is None:
            return cls()
        return cls(
            min_quality_for_trade    = float(getattr(rc, "min_quality_for_trade",     0.36)),
            min_confidence_for_trade = float(getattr(rc, "min_confidence_for_trade",  0.46)),
            min_quality_for_full_size= float(getattr(rc, "min_quality_for_full_size", 0.65)),
            adverse_breadth_states   = list(getattr(rc, "adverse_breadth_states",
                                                    ["EXTREME_DOWN", "CONTRACTING"])),
        )

    # ──────────────────────────────────────────────────────────
    #  Public API
    # ──────────────────────────────────────────────────────────

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
        run_date    = run_date or date.today()
        quality_map = {q.symbol: q.quality_score for q in quality_scores}
        base_posture= _MARKET_BASE_POSTURE.get(market_regime, RISK_DEFENSIVE)
        stats       = RoutingStats()
        decisions:  list[RoutingDecision] = []

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
            decision, rej_key = self._route_one(
                symbol=symbol,  market=r.market, run_date=run_date,
                stock_regime=regime, confidence=conf,
                market_regime=market_regime, quality_score=quality,
                breadth_state=breadth_state, breadth_score=breadth_score,
                sector_state=sector, base_posture=base_posture,
            ) if False else self._route_one(
                symbol=r.symbol, market=r.market, run_date=run_date,
                stock_regime=regime, confidence=conf,
                market_regime=market_regime, quality_score=quality,
                breadth_state=breadth_state, breadth_score=breadth_score,
                sector_state=sector, base_posture=base_posture,
            )
            decisions.append(decision)
            stats.record(decision, rej_key)

        self._log_routing_stats(stats, market_regime, breadth_state)
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

    # ──────────────────────────────────────────────────────────
    #  Core routing
    # ──────────────────────────────────────────────────────────

    def _route_one(
        self,
        symbol, market, run_date, stock_regime, confidence,
        market_regime, quality_score, breadth_state,
        breadth_score, sector_state, base_posture,
    ) -> tuple[RoutingDecision, str]:

        reasons: list[str] = []

        # Gate 1: unconditional regime blocks
        if stock_regime in ("TREND_DOWN", "VOLATILE", "QUIET", "UNCERTAIN"):
            reasons.append(f"regime {stock_regime} — not routable long")
            return (self._no_trade(symbol, market, run_date, stock_regime,
                                   market_regime, quality_score,
                                   breadth_state, sector_state, reasons),
                    f"regime_{stock_regime}")

        # Step 2: identify candidate strategy (no quality check yet)
        candidate = self._select_strategy(
            stock_regime, market_regime, breadth_state, sector_state, reasons
        )
        if candidate == NO_TRADE:
            return (self._no_trade(symbol, market, run_date, stock_regime,
                                   market_regime, quality_score,
                                   breadth_state, sector_state, reasons),
                    f"no_rule_{stock_regime}_{market_regime}")

        # Gate 3: per-strategy quality / confidence check
        gates    = _STRATEGY_GATES.get(candidate, {}).get(
            market_regime,
            {"min_quality": self.base_min_quality,
             "min_confidence": self.base_min_confidence},
        )
        min_q    = gates["min_quality"]
        min_conf = gates["min_confidence"]

        if quality_score < min_q:
            reasons.append(
                f"quality={quality_score:.2f} < {min_q:.2f} "
                f"({candidate}/{market_regime} gate)"
            )
            return (self._no_trade(symbol, market, run_date, stock_regime,
                                   market_regime, quality_score,
                                   breadth_state, sector_state, reasons),
                    f"quality_{candidate}")

        if confidence < min_conf:
            reasons.append(
                f"confidence={confidence:.2f} < {min_conf:.2f} "
                f"({candidate}/{market_regime} gate)"
            )
            return (self._no_trade(symbol, market, run_date, stock_regime,
                                   market_regime, quality_score,
                                   breadth_state, sector_state, reasons),
                    f"confidence_{candidate}")

        # Effective posture (strategy-market override or default)
        effective_posture = _STRATEGY_MARKET_POSTURE.get(
            (candidate, market_regime), base_posture
        )

        # Compute PSM
        risk, psm = self._compute_psm(
            quality_score, breadth_state, breadth_score,
            sector_state, effective_posture, reasons,
        )

        return (RoutingDecision(
            symbol=symbol, market=market, run_date=run_date,
            strategy=candidate, allowed=True,
            risk_profile=risk, position_size_multiplier=psm,
            regime_context=stock_regime, market_context=market_regime,
            quality_score=quality_score, breadth_state=breadth_state,
            sector_state=sector_state, reason=reasons,
        ), "")

    # ── Strategy selection rules ─────────────────────────────

    def _select_strategy(
        self,
        stock_regime, market_regime, breadth_state, sector_state, reasons,
    ) -> str:

        if stock_regime == "TREND_UP":
            notes = {
                "BULLISH_TREND": "bullish market — full trend following",
                "SIDEWAYS":      "sideways market — defensive sizing",
                "BEARISH_TREND": "bearish market — RS leaders only",
                "VOLATILE":      "volatile market — capital preservation",
                "QUIET":         "quiet market — trend intact",
                "UNCERTAIN":     "uncertain market — trend intact",
            }
            reasons.append(f"TREND_UP: {notes.get(market_regime, market_regime)}")
            return TREND_FOLLOWING

        if stock_regime == "MOMENTUM":
            if breadth_state in _MOMENTUM_KILL_BREADTH:
                reasons.append(
                    f"MOMENTUM blocked: breadth={breadth_state} — "
                    "no market support for momentum to persist"
                )
                return NO_TRADE
            note = ("strong breadth tailwind"
                    if breadth_state in ("EXPANDING", "EXTREME_UP")
                    else "contracting breadth — DEFENSIVE momentum"
                    if breadth_state == "CONTRACTING"
                    else "neutral breadth")
            reasons.append(f"MOMENTUM: {note} in {market_regime}")
            return MOMENTUM

        if stock_regime == "RANGE":
            if market_regime == "VOLATILE":
                reasons.append(
                    "MEAN_REVERSION blocked: volatile market — ranges break unpredictably"
                )
                return NO_TRADE
            preferred = market_regime == "SIDEWAYS"
            reasons.append(
                f"MEAN_REVERSION: "
                f"{'PREFERRED in sideways' if preferred else 'range stock in ' + market_regime}"
            )
            return MEAN_REVERSION

        if stock_regime == "BREAKOUT_SETUP":
            if market_regime in ("BEARISH_TREND", "VOLATILE"):
                reasons.append(
                    f"BREAKOUT blocked: {market_regime} — "
                    "breakouts have low follow-through in this market"
                )
                return NO_TRADE
            if sector_state not in _BREAKOUT_OK_SECTORS:
                reasons.append(
                    f"BREAKOUT blocked: sector={sector_state} — "
                    "need NEUTRAL or better sector"
                )
                return NO_TRADE
            if breadth_state in _MOMENTUM_KILL_BREADTH:
                reasons.append(
                    f"BREAKOUT blocked: breadth={breadth_state}"
                )
                return NO_TRADE
            reasons.append(
                f"BREAKOUT_SETUP: sector={sector_state} "
                f"breadth={breadth_state} market={market_regime}"
            )
            return BREAKOUT

        reasons.append(f"no rule matched for regime={stock_regime}")
        return NO_TRADE

    # ── PSM computation ──────────────────────────────────────

    def _compute_psm(
        self,
        quality, breadth_state, breadth_score,
        sector_state, base_posture, reasons,
    ) -> tuple[str, float]:

        base_psm = _PROFILE_BASE_PSM[base_posture]

        # Quality scaling
        if quality >= self.min_quality_full:
            q_mult = 1.0
        elif quality >= self.base_min_quality:
            q_mult = 0.30 + 0.70 * (
                (quality - self.base_min_quality) /
                max(self.min_quality_full - self.base_min_quality, 1e-9)
            )
        else:
            q_mult = 0.30
        psm = base_psm * q_mult

        # Breadth: continuous [0.75, 1.25]
        b_mult = 0.75 + 0.50 * breadth_score
        if breadth_state in self.adverse_breadth:
            b_mult = min(b_mult, 0.80)
        psm = psm * b_mult

        # Sector: 5-level graduated
        s_mult = _SECTOR_STATE_MULT.get(sector_state, 1.00)
        psm    = psm * s_mult

        psm = round(min(max(psm, 0.10), 1.50), 2)

        if psm >= 1.15:    risk = RISK_AGGRESSIVE
        elif psm >= 0.80:  risk = RISK_NORMAL
        elif psm >= 0.45:  risk = RISK_DEFENSIVE
        elif psm >= 0.15:  risk = RISK_CAPITAL_PRESERVATION
        else:              risk = RISK_OFF

        reasons.append(
            f"psm={psm:.2f} [{risk}] "
            f"(base={base_psm:.2f}×q={q_mult:.2f}×b={b_mult:.2f}×s={s_mult:.2f})"
        )
        return risk, psm

    # ── Diagnostics ──────────────────────────────────────────

    @staticmethod
    def _log_routing_stats(stats, market_regime, breadth_state):
        t = max(stats.total, 1)
        logger.info("─" * 55)
        logger.info(
            "StrategyRouter: %d routed | %d allowed (%.0f%%) | "
            "market=%s | breadth=%s",
            stats.total, stats.allowed, stats.allowed_pct,
            market_regime, breadth_state,
        )
        logger.info("Strategy distribution:")
        for strat in [TREND_FOLLOWING, MOMENTUM, MEAN_REVERSION, BREAKOUT, NO_TRADE]:
            n = stats.strategy_dist.get(strat, 0)
            if n:
                logger.info("  %-18s %3d (%.0f%%)", strat, n, n/t*100)

        if stats.rejection_reasons:
            logger.info("Top rejection reasons:")
            for reason, n in sorted(
                stats.rejection_reasons.items(), key=lambda x: -x[1]
            )[:5]:
                logger.info("  %-38s %3d (%.0f%%)", reason, n, n/t*100)

        logger.info("Regime → strategy:")
        for regime, sm in sorted(stats.regime_to_strategy.items()):
            parts = "  ".join(
                f"{s}:{n}" for s, n in sorted(sm.items(), key=lambda x: -x[1])
            )
            logger.info("  %-16s → %s", regime, parts)
        logger.info("─" * 55)

    @staticmethod
    def _no_trade(
        symbol, market, run_date, stock_regime,
        market_regime, quality_score, breadth_state,
        sector_state, reasons,
    ) -> RoutingDecision:
        return RoutingDecision(
            symbol=symbol, market=market, run_date=run_date,
            strategy=NO_TRADE, allowed=False, risk_profile=RISK_OFF,
            position_size_multiplier=0.0, regime_context=stock_regime,
            market_context=market_regime, quality_score=quality_score,
            breadth_state=breadth_state, sector_state=sector_state,
            reason=reasons,
        )