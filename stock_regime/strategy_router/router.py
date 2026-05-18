"""
stock_regime/strategy_router/router.py
========================================
Routes each classified stock to an appropriate strategy type.

IMPORTANT:
  - Does NOT generate trades
  - Does NOT compute position sizes (multiplier is a hint only)
  - Does NOT connect to any broker
  - Purely deterministic rule-based routing

Routing inputs (per stock)
--------------------------
  stable_regime    : stock's confirmed regime
  confidence       : smoothed confidence
  market_regime    : overall market regime
  quality_score    : OpportunityQualityEngine output [0,1]
  breadth_state    : BreadthEngine state string
  sector_state     : SectorEngine state string

Strategy types
--------------
  TREND_FOLLOWING  : strong uptrend, ADX confirmed
  MOMENTUM         : RS leadership + acceleration
  MEAN_REVERSION   : range-bound, ADX weak
  BREAKOUT         : setup near high with compression
  NO_TRADE         : insufficient quality or adverse conditions

Routing rules
-------------
See _ROUTING_RULES in config for all thresholds.
Rules are evaluated in priority order; first match wins.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)

# Strategy identifiers
TREND_FOLLOWING  = "TREND_FOLLOWING"
MOMENTUM         = "MOMENTUM"
MEAN_REVERSION   = "MEAN_REVERSION"
BREAKOUT         = "BREAKOUT"
NO_TRADE         = "NO_TRADE"

# Risk postures
RISK_NORMAL      = "NORMAL"
RISK_REDUCED     = "REDUCED"
RISK_MINIMAL     = "MINIMAL"
RISK_OFF         = "OFF"


@dataclass
class RoutingDecision:
    """The routing decision for one stock on one run date."""
    symbol:                   str
    market:                   str
    run_date:                 date
    strategy:                 str           # one of the strategy constants
    allowed:                  bool          # True if any trade is allowed
    risk_profile:             str           # NORMAL | REDUCED | MINIMAL | OFF
    position_size_multiplier: float         # 0.0–1.5 hint for risk engine
    regime_context:           str           # stock_regime value
    market_context:           str           # market_regime value
    quality_score:            float
    breadth_state:            str
    sector_state:             str
    reason:                   list[str]     # human-readable explanation

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
    Deterministic rule-based strategy router.

    Parameters (all overridable from config)
    -----------------------------------------
    min_quality_for_trade    : float  — quality score minimum to allow any trade
    min_confidence_for_trade : float  — smoothed confidence minimum
    min_quality_for_full_size: float  — quality score for 1.0x multiplier
    adverse_breadth_states   : list   — breadth states that reduce risk
    adverse_market_regimes   : list   — market regimes that reduce risk globally
    """

    def __init__(
        self,
        min_quality_for_trade:    float = 0.45,
        min_confidence_for_trade: float = 0.52,
        min_quality_for_full_size:float = 0.65,
        adverse_breadth_states:   Optional[list] = None,
        adverse_market_regimes:   Optional[list] = None,
    ) -> None:
        self.min_quality          = min_quality_for_trade
        self.min_confidence       = min_confidence_for_trade
        self.min_quality_full     = min_quality_for_full_size
        self.adverse_breadth      = set(adverse_breadth_states or ["EXTREME_DOWN", "CONTRACTING"])
        self.adverse_market       = set(adverse_market_regimes or ["VOLATILE", "UNCERTAIN"])

    @classmethod
    def from_config(cls, config) -> "StrategyRouter":
        rc = getattr(config, "strategy_router", None)
        if rc is None:
            return cls()
        return cls(
            min_quality_for_trade    = float(getattr(rc, "min_quality_for_trade",     0.45)),
            min_confidence_for_trade = float(getattr(rc, "min_confidence_for_trade",  0.52)),
            min_quality_for_full_size= float(getattr(rc, "min_quality_for_full_size", 0.65)),
            adverse_breadth_states   = list(getattr(rc, "adverse_breadth_states",
                                                    ["EXTREME_DOWN", "CONTRACTING"])),
            adverse_market_regimes   = list(getattr(rc, "adverse_market_regimes",
                                                    ["VOLATILE", "UNCERTAIN"])),
        )

    def route_batch(
        self,
        stable_results:  list,       # list[StableRegimeResult]
        quality_scores:  list,       # list[QualityScore]
        market_regime:   str         = "UNCERTAIN",
        breadth_state:   str         = "NEUTRAL",
        sector_states:   Optional[dict[str, str]] = None,   # symbol → sector_state
        run_date:        Optional[date] = None,
    ) -> list[RoutingDecision]:
        """
        Route every stock in stable_results to a strategy.

        Parameters
        ----------
        stable_results :
            Output of RegimeStabiliser.apply().
        quality_scores :
            Output of OpportunityQualityEngine.evaluate_batch().
            Matched to stable_results by symbol.
        market_regime :
            Current market regime string.
        breadth_state :
            Current breadth state from BreadthEngine.
        sector_states :
            Optional dict mapping symbol → sector_state.
        run_date :
            Date label.
        """
        run_date = run_date or date.today()

        # Build quality score lookup
        quality_map: dict[str, float] = {
            q.symbol: q.quality_score for q in quality_scores
        }

        decisions: list[RoutingDecision] = []
        strategy_counts: dict[str, int] = {}

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
                symbol         = r.symbol,
                market         = r.market,
                run_date       = run_date,
                stock_regime   = regime,
                confidence     = conf,
                market_regime  = market_regime,
                quality_score  = quality,
                breadth_state  = breadth_state,
                sector_state   = sector,
            )
            decisions.append(decision)
            strategy_counts[decision.strategy] = (
                strategy_counts.get(decision.strategy, 0) + 1
            )

        # Summary log
        total = len(decisions)
        allowed = sum(1 for d in decisions if d.allowed)
        logger.info(
            "StrategyRouter: %d routed | %d allowed | market=%s | breadth=%s",
            total, allowed, market_regime, breadth_state,
        )
        for strat, count in sorted(strategy_counts.items(), key=lambda x: -x[1]):
            logger.info("  %-18s %3d  (%.0f%%)", strat, count, count/max(total,1)*100)

        return decisions

    def persist(
        self,
        decisions:  list[RoutingDecision],
        output_dir: str | Path,
        universe:   str = "UNKNOWN",
    ) -> Optional[Path]:
        if not decisions:
            return None
        out = Path(output_dir) / "router" / str(decisions[0].run_date)
        out.mkdir(parents=True, exist_ok=True)
        path = out / f"{universe.lower()}_routing.parquet"
        pd.DataFrame([d.to_dict() for d in decisions]).to_parquet(
            path, engine="pyarrow", compression="snappy", index=False,
        )
        logger.info("Routing decisions [%s] → '%s'.", universe, path)
        return path

    # ──────────────────────────────────────────────────────────────
    #  Core routing logic
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
        sector_state:  str,
    ) -> RoutingDecision:

        reasons: list[str] = []

        # ── Gate 1: Quality / confidence floor ───────────────────────
        if quality_score < self.min_quality or confidence < self.min_confidence:
            reasons.append(
                f"quality={quality_score:.2f} or conf={confidence:.2f} below minimum"
            )
            return self._no_trade(symbol, market, run_date, stock_regime,
                                  market_regime, quality_score, breadth_state,
                                  sector_state, reasons)

        # ── Gate 2: Uncertain regime ──────────────────────────────────
        if stock_regime == "UNCERTAIN":
            reasons.append("regime UNCERTAIN — no clear signal")
            return self._no_trade(symbol, market, run_date, stock_regime,
                                  market_regime, quality_score, breadth_state,
                                  sector_state, reasons)

        # ── Gate 3: Adverse market regime ────────────────────────────
        if market_regime in self.adverse_market:
            reasons.append(f"adverse market regime: {market_regime}")
            return self._no_trade(symbol, market, run_date, stock_regime,
                                  market_regime, quality_score, breadth_state,
                                  sector_state, reasons)

        # ── Risk posture from breadth + sector ───────────────────────
        risk, psm = self._risk_posture(
            quality_score, breadth_state, sector_state,
            market_regime, reasons,
        )

        # ── Strategy selection ────────────────────────────────────────
        strategy = self._select_strategy(
            stock_regime, market_regime, breadth_state, sector_state, reasons
        )

        if strategy == NO_TRADE:
            return self._no_trade(symbol, market, run_date, stock_regime,
                                  market_regime, quality_score, breadth_state,
                                  sector_state, reasons)

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

    def _select_strategy(
        self,
        stock_regime:  str,
        market_regime: str,
        breadth_state: str,
        sector_state:  str,
        reasons:       list[str],
    ) -> str:
        """First-match rule table."""

        # TREND_FOLLOWING rules
        if stock_regime == "TREND_UP" and market_regime in ("BULLISH_TREND", "UNCERTAIN"):
            reasons += ["stock TREND_UP", f"market {market_regime}"]
            return TREND_FOLLOWING

        if stock_regime == "TREND_UP" and market_regime == "BEARISH_TREND":
            # Counter-trend — reduced but not blocked
            reasons += ["stock TREND_UP vs bearish market — reduced risk"]
            return TREND_FOLLOWING   # risk posture will reduce size

        # MOMENTUM rules
        if stock_regime == "MOMENTUM":
            if breadth_state in ("EXPANDING", "EXTREME_UP", "NEUTRAL"):
                reasons += ["stock MOMENTUM", f"breadth {breadth_state}"]
                return MOMENTUM
            reasons += [f"stock MOMENTUM but breadth {breadth_state} — no trade"]
            return NO_TRADE

        # MEAN_REVERSION rules
        if stock_regime == "RANGE":
            if market_regime in ("SIDEWAYS", "BULLISH_TREND", "UNCERTAIN"):
                reasons += ["stock RANGE", "mean reversion setup"]
                return MEAN_REVERSION
            reasons += [f"stock RANGE but market {market_regime} — adverse"]
            return NO_TRADE

        # BREAKOUT rules
        if stock_regime == "BREAKOUT_SETUP":
            if breadth_state in ("EXPANDING", "NEUTRAL") and sector_state in ("LEADING", "NEUTRAL"):
                reasons += ["stock BREAKOUT_SETUP", f"breadth {breadth_state}", f"sector {sector_state}"]
                return BREAKOUT
            reasons += [f"BREAKOUT_SETUP but conditions weak (breadth={breadth_state})"]
            return NO_TRADE

        # VOLATILE / TREND_DOWN / QUIET → no trade
        if stock_regime in ("VOLATILE", "TREND_DOWN", "QUIET"):
            reasons.append(f"regime {stock_regime} — not tradeable long")
            return NO_TRADE

        reasons.append(f"no rule matched for regime {stock_regime}")
        return NO_TRADE

    def _risk_posture(
        self,
        quality:       float,
        breadth_state: str,
        sector_state:  str,
        market_regime: str,
        reasons:       list[str],
    ) -> tuple[str, float]:
        """Return (risk_profile, position_size_multiplier)."""
        psm = 1.0

        # Quality-based scaling
        if quality >= self.min_quality_full:
            psm = 1.0
        elif quality >= self.min_quality:
            psm = 0.5 + 0.5 * (quality - self.min_quality) / (
                self.min_quality_full - self.min_quality
            )
        else:
            psm = 0.25

        # Breadth penalty
        if breadth_state in self.adverse_breadth:
            psm *= 0.60
            reasons.append(f"breadth {breadth_state} → size reduced")

        # Sector penalty
        if sector_state == "LAGGING":
            psm *= 0.75
            reasons.append("sector LAGGING → size reduced")
        elif sector_state == "LEADING":
            psm = min(psm * 1.10, 1.25)
            reasons.append("sector LEADING → size increased")

        # Market regime risk
        if market_regime == "BEARISH_TREND":
            psm *= 0.70
            reasons.append("bearish market → size reduced")

        psm = round(min(max(psm, 0.10), 1.50), 2)

        if psm >= 0.90:   risk = RISK_NORMAL
        elif psm >= 0.60: risk = RISK_REDUCED
        elif psm >= 0.30: risk = RISK_MINIMAL
        else:             risk = RISK_OFF

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