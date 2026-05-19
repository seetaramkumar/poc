"""
stock_regime/breadth_engine/breadth.py
========================================
Computes market breadth with hybrid participation logic.

Changes from prior version
---------------------------
Phase 1 — Hybrid breadth participation:
  The engine now counts regimes using a two-tier fallback:
  1. Use stable_regime when it exists AND is not UNCERTAIN.
  2. Fall back to stock_regime (raw) when stable_regime is UNCERTAIN.

  This detects EMERGING participation (stocks that are classified as
  TREND_UP in raw but haven't yet confirmed as stable) separately from
  STABLE participation (fully confirmed trends).

New BreadthSnapshot fields:
  pct_emerging_bullish  : % stocks with raw TREND_UP/MOMENTUM but UNCERTAIN stable
  pct_emerging_bearish  : % stocks with raw TREND_DOWN but UNCERTAIN stable
  pct_stable_bullish    : % with confirmed stable bullish regime
  participation_mode    : STABLE | EMERGING | MIXED

New BreadthEngine parameters:
  emerging_weight       : weight of emerging signals in breadth score (default 0.40)
  expanding_threshold   : lowered default to 45% (was 55%) for earlier detection
  contracting_threshold : lowered default to 38% (was 45%) for earlier detection
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)

_BULLISH_REGIMES = {"TREND_UP", "MOMENTUM"}
_BEARISH_REGIMES = {"TREND_DOWN"}


@dataclass
class BreadthSnapshot:
    """Point-in-time breadth state for one universe."""
    universe:              str
    run_date:              date
    total_stocks:          int

    # Hybrid participation (stable + emerging combined)
    pct_above_ema200:      float
    pct_bullish:           float   # combined stable + emerging bullish
    pct_bearish:           float   # combined stable + emerging bearish
    pct_neutral:           float

    # Stable participation (only confirmed stable regimes)
    pct_stable_bullish:    float
    pct_stable_bearish:    float

    # Emerging participation (raw regime ≠ UNCERTAIN but stable = UNCERTAIN)
    pct_emerging_bullish:  float
    pct_emerging_bearish:  float

    # Derived metrics
    advance_decline_ratio:  float
    momentum_participation: float
    breakout_participation: float
    volatile_participation: float
    breadth_thrust:         float
    regime_breadth_score:   float
    breadth_state:          str
    participation_mode:     str    # STABLE | EMERGING | MIXED

    def to_dict(self) -> dict:
        return {
            "universe":              self.universe,
            "run_date":              str(self.run_date),
            "total_stocks":          self.total_stocks,
            "pct_above_ema200":      round(self.pct_above_ema200,      2),
            "pct_bullish":           round(self.pct_bullish,           2),
            "pct_bearish":           round(self.pct_bearish,           2),
            "pct_neutral":           round(self.pct_neutral,           2),
            "pct_stable_bullish":    round(self.pct_stable_bullish,    2),
            "pct_stable_bearish":    round(self.pct_stable_bearish,    2),
            "pct_emerging_bullish":  round(self.pct_emerging_bullish,  2),
            "pct_emerging_bearish":  round(self.pct_emerging_bearish,  2),
            "advance_decline_ratio": round(self.advance_decline_ratio, 3),
            "momentum_participation":round(self.momentum_participation,2),
            "breakout_participation":round(self.breakout_participation,2),
            "volatile_participation":round(self.volatile_participation,2),
            "breadth_thrust":        round(self.breadth_thrust,        3),
            "regime_breadth_score":  round(self.regime_breadth_score,  4),
            "breadth_state":         self.breadth_state,
            "participation_mode":    self.participation_mode,
        }


class BreadthEngine:
    """
    Computes market breadth using hybrid stable + emerging participation.

    Parameters
    ----------
    extreme_threshold : float
        % bullish/bearish above which breadth is "extreme" (default 72%).
    expanding_threshold : float
        % bullish above which breadth is "expanding" (default 45%).
        Lowered from 55% to detect improving internals earlier.
    contracting_threshold : float
        % bearish above which breadth is "contracting" (default 38%).
        Lowered from 45% for earlier deterioration detection.
    emerging_weight : float
        Weight of emerging (unconfirmed) signals in the combined breadth score.
        Default 0.40 — emerging signals count 40% as much as stable signals.
    """

    def __init__(
        self,
        extreme_threshold:     float = 72.0,
        expanding_threshold:   float = 45.0,
        contracting_threshold: float = 38.0,
        emerging_weight:       float = 0.40,
    ) -> None:
        self.extreme_thr     = extreme_threshold
        self.expanding_thr   = expanding_threshold
        self.contracting_thr = contracting_threshold
        self.emerging_weight = emerging_weight

    @classmethod
    def from_config(cls, config) -> "BreadthEngine":
        bc = getattr(config, "breadth_engine", None)
        if bc is None:
            return cls()
        return cls(
            extreme_threshold     = float(getattr(bc, "extreme_threshold",     72.0)),
            expanding_threshold   = float(getattr(bc, "expanding_threshold",   45.0)),
            contracting_threshold = float(getattr(bc, "contracting_threshold", 38.0)),
            emerging_weight       = float(getattr(bc, "emerging_weight",       0.40)),
        )

    def compute(
        self,
        stable_results: list,
        universe:       str,
        run_date:       Optional[date] = None,
        prior_snapshot: Optional[BreadthSnapshot] = None,
    ) -> BreadthSnapshot:
        """
        Compute breadth using hybrid stable + emerging participation.

        Hybrid regime selection per stock:
          - If stable_regime exists AND is not UNCERTAIN → use stable_regime
          - Otherwise → fall back to stock_regime (raw classification)

        This ensures stocks with stock_regime=TREND_UP but stable_regime=UNCERTAIN
        contribute to emerging_bullish rather than being silently ignored.
        """
        run_date = run_date or date.today()
        valid    = [r for r in stable_results if r.is_valid()]
        total    = max(len(valid), 1)

        # ── Classify each stock as stable or emerging ─────────────────
        n_stable_bull   = 0
        n_stable_bear   = 0
        n_emerging_bull = 0
        n_emerging_bear = 0
        n_momentum      = 0
        n_breakout      = 0
        n_volatile      = 0
        n_above_ema200  = 0

        for r in valid:
            # EMA200 signal is from the raw signals — always available
            if r.signals.price_above_ema200:
                n_above_ema200 += 1

            # Determine which regime to use for breadth counting
            has_stable = hasattr(r, "stable_regime")
            stable_val = r.stable_regime.value if has_stable else "UNCERTAIN"
            raw_val    = r.stock_regime.value

            if has_stable and stable_val != "UNCERTAIN":
                # ── Use stable regime ─────────────────────────────────
                if stable_val in _BULLISH_REGIMES:
                    n_stable_bull += 1
                elif stable_val in _BEARISH_REGIMES:
                    n_stable_bear += 1

                if stable_val == "MOMENTUM":       n_momentum += 1
                if stable_val == "BREAKOUT_SETUP": n_breakout += 1
                if stable_val == "VOLATILE":       n_volatile += 1

            else:
                # ── Fall back to raw stock_regime ─────────────────────
                # These are emerging signals — counted with reduced weight
                if raw_val in _BULLISH_REGIMES:
                    n_emerging_bull += 1
                elif raw_val in _BEARISH_REGIMES:
                    n_emerging_bear += 1

                # Momentum/breakout/volatile from raw regime too
                if raw_val == "MOMENTUM":       n_momentum += 1
                if raw_val == "BREAKOUT_SETUP": n_breakout += 1
                if raw_val == "VOLATILE":       n_volatile += 1

        # ── Compute percentages ───────────────────────────────────────
        pct_stable_bull   = n_stable_bull   / total * 100
        pct_stable_bear   = n_stable_bear   / total * 100
        pct_emerging_bull = n_emerging_bull / total * 100
        pct_emerging_bear = n_emerging_bear / total * 100

        # Combined: stable + emerging (at reduced weight)
        ew = self.emerging_weight
        pct_bullish = pct_stable_bull + pct_emerging_bull * ew
        pct_bearish = pct_stable_bear + pct_emerging_bear * ew
        pct_neutral = max(100.0 - pct_bullish - pct_bearish, 0.0)

        pct_above_ema200 = n_above_ema200 / total * 100
        ad_ratio         = (pct_bullish / max(pct_bearish, 0.1))

        # ── Breadth thrust (change vs prior) ──────────────────────────
        prior_bullish  = prior_snapshot.pct_bullish if prior_snapshot else pct_bullish
        breadth_thrust = pct_bullish - prior_bullish

        # ── Composite breadth score [0, 1] ────────────────────────────
        # Weight stable signals more than emerging signals
        breadth_score = min(max(
            0.40 * (pct_stable_bull    / 100)       +   # stable bullish
            0.20 * (pct_emerging_bull  / 100) * ew  +   # emerging bullish (discounted)
            0.25 * (pct_above_ema200   / 100)       +   # price structure
            0.15 * (1.0 - pct_bearish  / 100),          # absence of bearish
        0.0), 1.0)

        # ── State classification ──────────────────────────────────────
        state = self._classify_state(pct_bullish, pct_bearish, breadth_thrust)

        # ── Participation mode ────────────────────────────────────────
        if pct_stable_bull > 5.0 and pct_emerging_bull < 5.0:
            mode = "STABLE"
        elif pct_emerging_bull > 5.0 and pct_stable_bull < 5.0:
            mode = "EMERGING"
        else:
            mode = "MIXED"

        snap = BreadthSnapshot(
            universe              = universe,
            run_date              = run_date,
            total_stocks          = total,
            pct_above_ema200      = round(pct_above_ema200,      2),
            pct_bullish           = round(pct_bullish,           2),
            pct_bearish           = round(pct_bearish,           2),
            pct_neutral           = round(pct_neutral,           2),
            pct_stable_bullish    = round(pct_stable_bull,       2),
            pct_stable_bearish    = round(pct_stable_bear,       2),
            pct_emerging_bullish  = round(pct_emerging_bull,     2),
            pct_emerging_bearish  = round(pct_emerging_bear,     2),
            advance_decline_ratio = round(ad_ratio,              3),
            momentum_participation= round(n_momentum / total * 100, 2),
            breakout_participation= round(n_breakout / total * 100, 2),
            volatile_participation= round(n_volatile  / total * 100, 2),
            breadth_thrust        = round(breadth_thrust,        3),
            regime_breadth_score  = round(breadth_score,         4),
            breadth_state         = state,
            participation_mode    = mode,
        )

        logger.info(
            "Breadth [%s]: %s (%s)  bull=%.1f%% (stable=%.1f%% emerging=%.1f%%)  "
            "bear=%.1f%%  AD=%.2f  above200=%.1f%%  thrust=%.2f  score=%.3f",
            universe, state, mode,
            pct_bullish, pct_stable_bull, pct_emerging_bull,
            pct_bearish, ad_ratio, pct_above_ema200,
            breadth_thrust, breadth_score,
        )

        if pct_bullish > self.extreme_thr:
            logger.warning(
                "BREADTH EXTREME_UP [%s]: %.1f%% combined bullish — potential overbought",
                universe, pct_bullish,
            )
        elif pct_bearish > self.contracting_thr:
            logger.warning(
                "BREADTH EXTREME_DOWN [%s]: %.1f%% combined bearish — potential oversold",
                universe, pct_bearish,
            )

        return snap

    def persist(
        self,
        snapshot:   BreadthSnapshot,
        output_dir: str | Path,
        append:     bool = True,
    ) -> Path:
        out = Path(output_dir) / "breadth"
        out.mkdir(parents=True, exist_ok=True)
        path = out / f"{snapshot.universe.lower()}_breadth.parquet"

        new_df = pd.DataFrame([snapshot.to_dict()])
        if append and path.exists():
            existing = pd.read_parquet(path)
            # Handle new columns not in older parquet files (backward compat)
            for col in new_df.columns:
                if col not in existing.columns:
                    existing[col] = None
            combined = pd.concat([existing, new_df], ignore_index=True)
            combined.drop_duplicates(
                subset=["universe", "run_date"], keep="last", inplace=True
            )
            combined.to_parquet(path, engine="pyarrow", compression="snappy", index=False)
        else:
            new_df.to_parquet(path, engine="pyarrow", compression="snappy", index=False)

        logger.info("Breadth persisted → '%s'.", path)
        return path

    @staticmethod
    def load_prior(output_dir: str | Path, universe: str) -> Optional[BreadthSnapshot]:
        path = Path(output_dir) / "breadth" / f"{universe.lower()}_breadth.parquet"
        if not path.exists():
            return None
        df = pd.read_parquet(path).sort_values("run_date")
        if df.empty:
            return None
        row = df.iloc[-1]
        try:
            return BreadthSnapshot(
                universe              = row["universe"],
                run_date              = row["run_date"],
                total_stocks          = int(row["total_stocks"]),
                pct_above_ema200      = float(row["pct_above_ema200"]),
                pct_bullish           = float(row["pct_bullish"]),
                pct_bearish           = float(row["pct_bearish"]),
                pct_neutral           = float(row["pct_neutral"]),
                pct_stable_bullish    = float(row.get("pct_stable_bullish",   row["pct_bullish"])),
                pct_stable_bearish    = float(row.get("pct_stable_bearish",   row["pct_bearish"])),
                pct_emerging_bullish  = float(row.get("pct_emerging_bullish", 0.0)),
                pct_emerging_bearish  = float(row.get("pct_emerging_bearish", 0.0)),
                advance_decline_ratio = float(row["advance_decline_ratio"]),
                momentum_participation= float(row["momentum_participation"]),
                breakout_participation= float(row["breakout_participation"]),
                volatile_participation= float(row["volatile_participation"]),
                breadth_thrust        = float(row["breadth_thrust"]),
                regime_breadth_score  = float(row["regime_breadth_score"]),
                breadth_state         = row["breadth_state"],
                participation_mode    = str(row.get("participation_mode", "MIXED")),
            )
        except Exception:
            return None

    def _classify_state(self, pct_bull: float, pct_bear: float, thrust: float) -> str:
        if pct_bull > self.extreme_thr:
            return "EXTREME_UP"
        if pct_bear > self.extreme_thr:
            return "EXTREME_DOWN"
        if pct_bull > self.expanding_thr:
            return "EXPANDING"
        if pct_bear > self.contracting_thr:
            return "CONTRACTING"
        # Thrust-based transition detection — detect turning points early
        if thrust > 5.0:
            return "EXPANDING"
        if thrust < -5.0:
            return "CONTRACTING"
        return "NEUTRAL"