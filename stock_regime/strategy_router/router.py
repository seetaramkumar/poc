"""
stock_regime/strategy_router/router.py
========================================
Soft-factor PSM router — replaces stacked hard gates with continuous multipliers.

Design change from prior version
----------------------------------
The previous router stacked multiple hard gates (quality, confidence, market
regime, breadth) sequentially, causing strong stocks to be rejected before
strategy selection when any single gate failed. A quality score of 0.58 vs. a
threshold of 0.60 produced the same NO_TRADE as a score of 0.05.

New model: MINIMAL hard blocks + continuous soft-factor multipliers.

HARD BLOCKS (unconditional NO_TRADE):
  - Stock regime is TREND_DOWN, VOLATILE, or UNCERTAIN
  - quality_score < ABSOLUTE_MIN_QUALITY  (0.30)
  - confidence < ABSOLUTE_MIN_CONFIDENCE  (0.30)
  - Invalid data (error flag set by stabiliser)

Everything else is a SOFT FACTOR that adjusts PSM and risk profile:
  - quality_score     → quality multiplier (0.30–1.0 → 0.40–1.10 PSM range)
  - confidence        → confidence multiplier (0.30–1.0 → 0.60–1.10 PSM range)
  - market_regime     → base posture (drives starting PSM ceiling)
  - breadth_state     → breadth multiplier (continuous, not binary block)
  - breadth_score     → fine-grained breadth scaling
  - sector_state      → sector multiplier (±10–25%)

This means:
  quality=0.58 (just below old threshold of 0.60)
  → quality_mult ≈ 0.83
  → DEFENSIVE profile at reduced PSM
  → NOT NO_TRADE

Result: BEARISH market + EXPANDING breadth allows strongest TREND_UP and
MOMENTUM stocks at reduced sizing and DEFENSIVE/CAPITAL_PRESERVATION posture,
rather than blanket NO_TRADE for the universe.

Strategy routing rules (first-match per regime):
  TREND_UP       → TREND_FOLLOWING  (all market regimes, PSM scaled by context)
  MOMENTUM       → MOMENTUM         (blocked only on EXTREME_DOWN breadth)
  RANGE          → MEAN_REVERSION   (blocked only on VOLATILE market regime)
  BREAKOUT_SETUP → BREAKOUT         (requires non-bearish market + decent breadth)
  QUIET          → HOLD_CASH        (not tradeable long)
  TREND_DOWN     → NO_TRADE         (hard block)
  VOLATILE       → NO_TRADE         (hard block)
  UNCERTAIN      → NO_TRADE         (hard block)
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
HOLD_CASH       = "HOLD_CASH"
NO_TRADE        = "NO_TRADE"

# ── Risk profiles ────────────────────────────────────────────────
RISK_AGGRESSIVE            = "AGGRESSIVE"
RISK_NORMAL                = "NORMAL"
RISK_DEFENSIVE             = "DEFENSIVE"
RISK_CAPITAL_PRESERVATION  = "CAPITAL_PRESERVATION"
RISK_OFF                   = "OFF"

# ── Hard-block floor — below these values signal is meaningless ───
# These are the ONLY values that cause unconditional NO_TRADE.
# They are set low enough that only genuinely unreliable signals hit them.
ABSOLUTE_MIN_QUALITY    = 0.30
ABSOLUTE_MIN_CONFIDENCE = 0.30

# ── Market regime → base PSM ceiling ─────────────────────────────
# This is the MAXIMUM PSM achievable in each market regime before
# per-stock soft-factor scaling.  Bullish = up to 1.25×, volatile = 0.40×.
# Note: no market regime produces 0.0 here — that would be a hard block.
_MARKET_BASE_PSM_CEILING: dict[str, float] = {
    "BULLISH_TREND": 1.25,
    "SIDEWAYS":      0.80,
    "BEARISH_TREND": 0.65,   # was hard-blocked; now soft — reduced ceiling
    "VOLATILE":      0.45,   # was hard-blocked; now soft — strongly reduced
    "QUIET":         0.75,
    "UNCERTAIN":     0.60,
}

# ── Market regime → base risk posture label ───────────────────────
# The label is derived from the ceiling after all multipliers are applied.
# This table is used only for the "base" label before per-stock adjustment.
_MARKET_BASE_POSTURE: dict[str, str] = {
    "BULLISH_TREND": RISK_NORMAL,
    "SIDEWAYS":      RISK_DEFENSIVE,
    "BEARISH_TREND": RISK_DEFENSIVE,
    "VOLATILE":      RISK_CAPITAL_PRESERVATION,
    "QUIET":         RISK_DEFENSIVE,
    "UNCERTAIN":     RISK_DEFENSIVE,
}

# ── Base PSM per risk profile label ─────────────────────────────
_PROFILE_BASE_PSM: dict[str, float] = {
    RISK_AGGRESSIVE:           1.25,
    RISK_NORMAL:               1.00,
    RISK_DEFENSIVE:            0.65,
    RISK_CAPITAL_PRESERVATION: 0.35,
    RISK_OFF:                  0.00,
}

# ── Breadth states that impose a hard ceiling on PSM ─────────────
# These do NOT block trades — they cap PSM at a reduced level.
_BREADTH_PSM_CEILING: dict[str, float] = {
    "EXTREME_UP":   1.00,   # cap exuberance — overbought risk
    "EXPANDING":    1.00,   # no cap — full PSM available
    "NEUTRAL":      1.00,   # no cap
    "CONTRACTING":  0.75,   # soft cap — reduce but don't block
    "EXTREME_DOWN": 0.50,   # strong cap — only highest quality trades
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
    Soft-factor adaptive strategy router.

    Hard blocks are limited to genuinely unreliable signals (TREND_DOWN,
    VOLATILE, UNCERTAIN regimes, and quality/confidence below 0.30).
    Everything else adjusts PSM and risk profile continuously.

    Parameters
    ----------
    absolute_min_quality : float
        Only stocks below this are hard-blocked. Default 0.30.
        This is intentionally lower than the old soft threshold (0.40–0.65)
        so moderate-quality stocks are routed at reduced sizing.
    absolute_min_confidence : float
        Same logic for confidence. Default 0.30.
    min_quality_for_full_size : float
        Quality score at which PSM quality-multiplier reaches 1.0.
        Below this, PSM scales down continuously. Default 0.70.
    adverse_breadth_cap_states : list[str]
        Breadth states that cap PSM (but do NOT block trades).
    """

    def __init__(
        self,
        absolute_min_quality:      float        = ABSOLUTE_MIN_QUALITY,
        absolute_min_confidence:   float        = ABSOLUTE_MIN_CONFIDENCE,
        min_quality_for_full_size: float        = 0.70,
        adverse_breadth_cap_states: Optional[list[str]] = None,
    ) -> None:
        self.abs_min_quality      = absolute_min_quality
        self.abs_min_confidence   = absolute_min_confidence
        self.min_quality_full     = min_quality_for_full_size
        # States that impose a PSM ceiling (not a block)
        self.adverse_breadth_caps = set(
            adverse_breadth_cap_states or ["EXTREME_DOWN", "CONTRACTING"]
        )

    @classmethod
    def from_config(cls, config) -> "StrategyRouter":
        rc = getattr(config, "strategy_router", None)
        if rc is None:
            return cls()
        return cls(
            absolute_min_quality      = float(getattr(rc, "absolute_min_quality",
                                                       ABSOLUTE_MIN_QUALITY)),
            absolute_min_confidence   = float(getattr(rc, "absolute_min_confidence",
                                                       ABSOLUTE_MIN_CONFIDENCE)),
            min_quality_for_full_size = float(getattr(rc, "min_quality_for_full_size", 0.70)),
            adverse_breadth_cap_states= list(getattr(rc, "adverse_breadth_states",
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
        Route every stock in the batch to a strategy.

        Each stock either:
        - Receives a hard NO_TRADE (regime is untradeable or signal is noise), or
        - Receives an allowed strategy with a PSM and risk profile reflecting
          the combined effect of all soft factors.

        Parameters
        ----------
        stable_results :
            Output of RegimeStabiliser.apply().
        quality_scores :
            Output of OpportunityQualityEngine.evaluate_batch().
        market_regime :
            Current market regime string from MarketRegimeEngine.
        breadth_state :
            BreadthEngine discrete state string.
        breadth_score :
            BreadthEngine composite score [0, 1] for continuous scaling.
        sector_states :
            symbol → sector_state ("LEADING" | "NEUTRAL" | "LAGGING").
        run_date :
            Date label for persistence.
        """
        run_date    = run_date or date.today()
        quality_map = {q.symbol: q.quality_score for q in quality_scores}

        # Pre-compute market-level context once per batch (not per stock)
        market_ceiling = _MARKET_BASE_PSM_CEILING.get(market_regime, 0.60)
        market_posture = _MARKET_BASE_POSTURE.get(market_regime, RISK_DEFENSIVE)

        decisions:       list[RoutingDecision] = {}
        strategy_counts: dict[str, int]        = {}

        for r in stable_results:
            if not r.is_valid():
                # Invalid data is always a hard block — engine-level error
                d = self._hard_block(
                    r.symbol, r.market, run_date,
                    "UNKNOWN", market_regime, 0.0, breadth_state,
                    (sector_states or {}).get(r.symbol, "NEUTRAL"),
                    [f"invalid data: {r.error}"],
                )
                strategy_counts[d.strategy] = strategy_counts.get(d.strategy, 0) + 1
                decisions[r.symbol] = d
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

            d = self._route_one(
                symbol         = r.symbol,
                market         = r.market,
                run_date       = run_date,
                stock_regime   = regime,
                confidence     = conf,
                market_regime  = market_regime,
                quality_score  = quality,
                breadth_state  = breadth_state,
                breadth_score  = breadth_score,
                sector_state   = sector,
                market_ceiling = market_ceiling,
                market_posture = market_posture,
            )
            strategy_counts[d.strategy] = strategy_counts.get(d.strategy, 0) + 1
            decisions[r.symbol] = d

        result_list = list(decisions.values())
        total   = len(result_list)
        allowed = sum(1 for d in result_list if d.allowed)
        logger.info(
            "StrategyRouter: %d routed | %d allowed (%.0f%%) | market=%s | breadth=%s",
            total, allowed, allowed / max(total, 1) * 100,
            market_regime, breadth_state,
        )
        for strat, count in sorted(strategy_counts.items(), key=lambda x: -x[1]):
            logger.info(
                "  %-20s %3d  (%.0f%%)", strat, count, count / max(total, 1) * 100
            )
        return result_list

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
    #  Per-symbol routing
    # ──────────────────────────────────────────────────────────────

    def _route_one(
        self,
        symbol:         str,
        market:         str,
        run_date:       date,
        stock_regime:   str,
        confidence:     float,
        market_regime:  str,
        quality_score:  float,
        breadth_state:  str,
        breadth_score:  float,
        sector_state:   str,
        market_ceiling: float,
        market_posture: str,
    ) -> RoutingDecision:

        reasons: list[str] = []

        # ── HARD BLOCK 1: regime is fundamentally untradeable long ────
        # These represent genuine signal invalidity, not weak signals.
        if stock_regime in ("TREND_DOWN", "VOLATILE", "QUIET"):
            reasons.append(f"hard block: regime={stock_regime} is not tradeable long")
            return self._hard_block(
                symbol, market, run_date, stock_regime, market_regime,
                quality_score, breadth_state, sector_state, reasons,
            )

        if stock_regime == "UNCERTAIN":
            reasons.append("hard block: regime=UNCERTAIN — signal below confidence floor")
            return self._hard_block(
                symbol, market, run_date, stock_regime, market_regime,
                quality_score, breadth_state, sector_state, reasons,
            )

        # ── HARD BLOCK 2: signal is pure noise (floor, not threshold) ─
        # 0.30 is intentionally forgiving — it blocks only garbage, not
        # stocks that are merely average.
        if quality_score < self.abs_min_quality:
            reasons.append(
                f"hard block: quality={quality_score:.2f} < floor={self.abs_min_quality:.2f}"
                " — signal is noise"
            )
            return self._hard_block(
                symbol, market, run_date, stock_regime, market_regime,
                quality_score, breadth_state, sector_state, reasons,
            )

        if confidence < self.abs_min_confidence:
            reasons.append(
                f"hard block: confidence={confidence:.2f} < floor={self.abs_min_confidence:.2f}"
                " — signal is noise"
            )
            return self._hard_block(
                symbol, market, run_date, stock_regime, market_regime,
                quality_score, breadth_state, sector_state, reasons,
            )

        # ── Strategy selection ────────────────────────────────────────
        # Returns NO_TRADE only for structural incompatibility (e.g.
        # BREAKOUT in a bearish volatile market), not for weak signals.
        strategy = self._select_strategy(
            stock_regime, market_regime, breadth_state, sector_state, reasons
        )

        if strategy == NO_TRADE:
            # Structural incompatibility — not a signal quality issue
            return self._hard_block(
                symbol, market, run_date, stock_regime, market_regime,
                quality_score, breadth_state, sector_state, reasons,
            )

        # ── PSM computation via soft factors ─────────────────────────
        # All remaining decisions are ALLOWED — PSM reflects combined
        # signal strength across quality, confidence, market, breadth, sector.
        risk, psm = self._compute_psm(
            quality_score  = quality_score,
            confidence     = confidence,
            market_ceiling = market_ceiling,
            breadth_state  = breadth_state,
            breadth_score  = breadth_score,
            sector_state   = sector_state,
            market_regime  = market_regime,
            reasons        = reasons,
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

    # ── Strategy selection rules ──────────────────────────────────────
    # Only returns NO_TRADE for structural incompatibility.
    # Weak signals are passed through and handled by PSM scaling.

    def _select_strategy(
        self,
        stock_regime:  str,
        market_regime: str,
        breadth_state: str,
        sector_state:  str,
        reasons:       list[str],
    ) -> str:

        # ── TREND_FOLLOWING ──────────────────────────────────────────
        # Allowed in ALL market regimes — PSM will be reduced in adverse ones.
        if stock_regime == "TREND_UP":
            reasons.append(
                f"TREND_UP → TREND_FOLLOWING (market={market_regime}, "
                f"breadth={breadth_state}) — PSM scaled by context"
            )
            return TREND_FOLLOWING

        # ── MOMENTUM ────────────────────────────────────────────────
        # Blocked only when breadth is EXTREME_DOWN — everything else
        # is allowed at reduced PSM via breadth multiplier.
        if stock_regime == "MOMENTUM":
            if breadth_state == "EXTREME_DOWN":
                reasons.append(
                    "MOMENTUM hard blocked: breadth=EXTREME_DOWN — "
                    "market internals are collapsing"
                )
                return NO_TRADE
            reasons.append(
                f"MOMENTUM → MOMENTUM (breadth={breadth_state}) — "
                "PSM scaled by breadth score"
            )
            return MOMENTUM

        # ── MEAN_REVERSION ──────────────────────────────────────────
        # Blocked only when market is VOLATILE (mean-reversion is unreliable
        # when price action is structurally erratic).
        if stock_regime == "RANGE":
            if market_regime == "VOLATILE":
                reasons.append(
                    "RANGE hard blocked: market=VOLATILE — "
                    "mean-reversion unreliable in structurally erratic markets"
                )
                return NO_TRADE
            reasons.append(
                f"RANGE → MEAN_REVERSION (market={market_regime})"
            )
            return MEAN_REVERSION

        # ── BREAKOUT ────────────────────────────────────────────────
        # Breakouts require a minimum environment quality because they
        # have directional momentum expectations baked in. Blocked when
        # both market AND breadth are adverse simultaneously.
        if stock_regime == "BREAKOUT_SETUP":
            market_adverse  = market_regime in ("BEARISH_TREND", "VOLATILE")
            breadth_adverse = breadth_state in ("EXTREME_DOWN", "CONTRACTING")
            if market_adverse and breadth_adverse:
                # Both conditions adverse at once — structural block
                reasons.append(
                    f"BREAKOUT_SETUP blocked: market={market_regime} AND "
                    f"breadth={breadth_state} both adverse — breakout unlikely to follow through"
                )
                return NO_TRADE
            # Only one factor adverse → allowed at reduced PSM
            reasons.append(
                f"BREAKOUT_SETUP → BREAKOUT (market={market_regime}, "
                f"breadth={breadth_state}, sector={sector_state})"
            )
            return BREAKOUT

        reasons.append(f"no rule matched for regime={stock_regime}")
        return NO_TRADE

    # ── PSM computation via soft factors ─────────────────────────────
    #
    # PSM = market_ceiling
    #       × quality_multiplier(quality_score)
    #       × confidence_multiplier(confidence)
    #       × breadth_multiplier(breadth_score, breadth_state)
    #       × sector_multiplier(sector_state)
    #
    # Each multiplier is continuous — no cliff edges.

    def _compute_psm(
        self,
        quality_score:  float,
        confidence:     float,
        market_ceiling: float,
        breadth_state:  str,
        breadth_score:  float,
        sector_state:   str,
        market_regime:  str,
        reasons:        list[str],
    ) -> tuple[str, float]:
        """
        Compute PSM as a product of soft-factor multipliers applied to the
        market-regime ceiling.

        All multipliers are clamped to [0.30, 1.10] so no single factor can
        drive PSM to zero (that's the job of hard blocks).
        """

        # ── Starting point: market regime ceiling ────────────────────
        psm = market_ceiling
        reasons.append(f"market={market_regime} ceiling={market_ceiling:.2f}")

        # ── Soft factor 1: Quality multiplier ────────────────────────
        # quality=0.30 (absolute floor) → mult=0.40
        # quality=0.70 (full size)      → mult=1.00
        # quality=1.00                  → mult=1.10 (bonus for exceptional quality)
        # Linear interpolation between anchor points.
        q_mult = self._quality_multiplier(quality_score)
        psm   *= q_mult
        reasons.append(f"quality={quality_score:.2f} → quality_mult={q_mult:.2f}")

        # ── Soft factor 2: Confidence multiplier ─────────────────────
        # confidence=0.30 → mult=0.60
        # confidence=0.65 → mult=1.00
        # confidence=1.00 → mult=1.10
        c_mult = self._confidence_multiplier(confidence)
        psm   *= c_mult
        reasons.append(f"confidence={confidence:.2f} → conf_mult={c_mult:.2f}")

        # ── Soft factor 3: Breadth multiplier (continuous) ────────────
        # breadth_score [0, 1]: 0.5=neutral → 1.0×; 0.2=weak → 0.75×; 0.8=strong → 1.15×
        # Additionally, adverse breadth STATES apply a PSM ceiling.
        b_mult    = 0.75 + 0.50 * breadth_score   # range [0.75, 1.25]
        b_ceiling = _BREADTH_PSM_CEILING.get(breadth_state, 1.00)
        psm      *= b_mult
        if psm > b_ceiling:
            reasons.append(
                f"breadth={breadth_state} (score={breadth_score:.2f}) "
                f"→ breadth_mult={b_mult:.2f}, ceiling={b_ceiling:.2f} applied"
            )
            psm = b_ceiling
        else:
            reasons.append(
                f"breadth={breadth_state} (score={breadth_score:.2f}) "
                f"→ breadth_mult={b_mult:.2f}"
            )

        # ── Soft factor 4: Sector multiplier ─────────────────────────
        # LEADING: +10% (confirmed sector rotation support)
        # NEUTRAL: no change
        # LAGGING: -25% (sector headwind — reduce but don't block)
        if sector_state == "LEADING":
            psm *= 1.10
            reasons.append("sector=LEADING → +10%")
        elif sector_state == "LAGGING":
            psm *= 0.75
            reasons.append("sector=LAGGING → -25%")

        # ── Final clamp: [0.10, 1.50] ─────────────────────────────────
        # 0.10 floor ensures we never produce a zero PSM through multipliers
        # (hard blocks handle the genuine zero case).
        psm = round(min(max(psm, 0.10), 1.50), 2)

        # ── Derive risk profile from final PSM ────────────────────────
        risk = self._psm_to_risk_profile(psm)
        reasons.append(f"final PSM={psm:.2f} → profile={risk}")

        return risk, psm

    # ── Multiplier helpers ───────────────────────────────────────────

    @staticmethod
    def _quality_multiplier(quality: float) -> float:
        """
        Continuous quality → PSM multiplier.
        Anchor points:
          0.30 → 0.40  (absolute floor — barely above noise)
          0.50 → 0.70  (moderate quality)
          0.70 → 1.00  (good quality — full size available)
          1.00 → 1.10  (exceptional quality — slight bonus)
        """
        if quality >= 0.70:
            # 0.70 → 1.00, 1.00 → 1.10
            return round(1.00 + 0.10 * (quality - 0.70) / 0.30, 4)
        elif quality >= 0.50:
            # 0.50 → 0.70, 0.70 → 1.00
            return round(0.70 + 0.30 * (quality - 0.50) / 0.20, 4)
        elif quality >= 0.30:
            # 0.30 → 0.40, 0.50 → 0.70
            return round(0.40 + 0.30 * (quality - 0.30) / 0.20, 4)
        else:
            return 0.40   # below floor — hard block should have caught this

    @staticmethod
    def _confidence_multiplier(confidence: float) -> float:
        """
        Continuous confidence → PSM multiplier.
        Anchor points:
          0.30 → 0.60  (just above noise floor)
          0.65 → 1.00  (standard confidence — full ceiling available)
          1.00 → 1.10  (very high confidence — slight bonus)
        """
        if confidence >= 0.65:
            # 0.65 → 1.00, 1.00 → 1.10
            return round(1.00 + 0.10 * (confidence - 0.65) / 0.35, 4)
        elif confidence >= 0.30:
            # 0.30 → 0.60, 0.65 → 1.00
            return round(0.60 + 0.40 * (confidence - 0.30) / 0.35, 4)
        else:
            return 0.60

    @staticmethod
    def _psm_to_risk_profile(psm: float) -> str:
        """Map final PSM to a human-readable risk profile label."""
        if psm >= 1.10:
            return RISK_AGGRESSIVE
        elif psm >= 0.75:
            return RISK_NORMAL
        elif psm >= 0.40:
            return RISK_DEFENSIVE
        elif psm >= 0.15:
            return RISK_CAPITAL_PRESERVATION
        else:
            return RISK_OFF

    # ── Hard block helper ────────────────────────────────────────────

    @staticmethod
    def _hard_block(
        symbol:        str,
        market:        str,
        run_date:      date,
        stock_regime:  str,
        market_regime: str,
        quality_score: float,
        breadth_state: str,
        sector_state:  str,
        reasons:       list[str],
    ) -> RoutingDecision:
        """Construct a NO_TRADE RoutingDecision for hard-blocked stocks."""
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