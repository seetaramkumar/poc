"""
stock_regime/breadth_engine/breadth.py
========================================
Computes market breadth metrics from a batch of StableRegimeResult objects.

Breadth metrics produced
------------------------
pct_above_ema200         : % stocks with price above EMA-200
pct_bullish              : % stocks in TREND_UP or MOMENTUM
pct_bearish              : % stocks in TREND_DOWN
advance_decline_ratio    : bullish / bearish count ratio
momentum_participation   : % stocks in MOMENTUM regime
breakout_participation   : % stocks in BREAKOUT_SETUP
breadth_thrust           : short-term breadth acceleration (today vs prior)
regime_breadth_score     : composite breadth [0, 1]

Breadth state classification
----------------------------
EXPANDING  — majority of stocks bullish and improving
NEUTRAL    — mixed signals
CONTRACTING— majority bearish or deteriorating
EXTREME_UP — >75% bullish (potential overbought)
EXTREME_DOWN — >75% bearish (potential oversold)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)

_BULLISH_REGIMES  = {"TREND_UP", "MOMENTUM"}
_BEARISH_REGIMES  = {"TREND_DOWN"}
_NEUTRAL_REGIMES  = {"RANGE", "QUIET", "UNCERTAIN"}


@dataclass
class BreadthSnapshot:
    """Point-in-time breadth state for one universe."""
    universe:             str
    run_date:             date
    total_stocks:         int
    pct_above_ema200:     float   # % with price > EMA200
    pct_bullish:          float   # % TREND_UP + MOMENTUM
    pct_bearish:          float   # % TREND_DOWN
    pct_neutral:          float   # % RANGE + QUIET + UNCERTAIN
    advance_decline_ratio: float  # bullish_count / max(bearish_count, 1)
    momentum_participation: float # % MOMENTUM regime
    breakout_participation: float # % BREAKOUT_SETUP regime
    volatile_participation: float # % VOLATILE regime
    breadth_thrust:        float  # pct_bullish - prior_pct_bullish (daily change)
    regime_breadth_score:  float  # composite [0, 1]
    breadth_state:         str    # EXPANDING|NEUTRAL|CONTRACTING|EXTREME_UP|EXTREME_DOWN

    def to_dict(self) -> dict:
        return {
            "universe":              self.universe,
            "run_date":              str(self.run_date),
            "total_stocks":          self.total_stocks,
            "pct_above_ema200":      round(self.pct_above_ema200,      2),
            "pct_bullish":           round(self.pct_bullish,           2),
            "pct_bearish":           round(self.pct_bearish,           2),
            "pct_neutral":           round(self.pct_neutral,           2),
            "advance_decline_ratio": round(self.advance_decline_ratio, 3),
            "momentum_participation":round(self.momentum_participation,2),
            "breakout_participation":round(self.breakout_participation,2),
            "volatile_participation":round(self.volatile_participation,2),
            "breadth_thrust":        round(self.breadth_thrust,        3),
            "regime_breadth_score":  round(self.regime_breadth_score,  4),
            "breadth_state":         self.breadth_state,
        }


class BreadthEngine:
    """
    Computes market breadth from a batch of classified stocks.

    Parameters
    ----------
    extreme_threshold : float
        % bullish/bearish above which breadth is "extreme" (default 75%).
    expanding_threshold : float
        % bullish above which breadth is "expanding" (default 55%).
    contracting_threshold : float
        % bearish above which breadth is "contracting" (default 45%).
    """

    def __init__(
        self,
        extreme_threshold:     float = 75.0,
        expanding_threshold:   float = 55.0,
        contracting_threshold: float = 45.0,
    ) -> None:
        self.extreme_thr     = extreme_threshold
        self.expanding_thr   = expanding_threshold
        self.contracting_thr = contracting_threshold

    @classmethod
    def from_config(cls, config) -> "BreadthEngine":
        bc = getattr(config, "breadth_engine", None)
        if bc is None:
            return cls()
        return cls(
            extreme_threshold     = float(getattr(bc, "extreme_threshold",     75.0)),
            expanding_threshold   = float(getattr(bc, "expanding_threshold",   55.0)),
            contracting_threshold = float(getattr(bc, "contracting_threshold", 45.0)),
        )

    def compute(
        self,
        stable_results: list,       # list[StableRegimeResult]
        universe:       str,
        run_date:       Optional[date] = None,
        prior_snapshot: Optional[BreadthSnapshot] = None,
    ) -> BreadthSnapshot:
        """
        Compute breadth for one universe from today's classifications.

        Parameters
        ----------
        stable_results :
            Output of RegimeStabiliser.apply().
        universe :
            Universe label.
        run_date :
            Date of this run.
        prior_snapshot :
            Previous run's snapshot for breadth_thrust calculation.
        """
        run_date = run_date or date.today()

        valid = [r for r in stable_results if r.is_valid()]
        total = max(len(valid), 1)

        # Regime counts
        regimes = [
            r.stable_regime.value if hasattr(r, "stable_regime")
            else r.stock_regime.value
            for r in valid
        ]
        n_bullish   = sum(1 for r in regimes if r in _BULLISH_REGIMES)
        n_bearish   = sum(1 for r in regimes if r in _BEARISH_REGIMES)
        n_momentum  = sum(1 for r in regimes if r == "MOMENTUM")
        n_breakout  = sum(1 for r in regimes if r == "BREAKOUT_SETUP")
        n_volatile  = sum(1 for r in regimes if r == "VOLATILE")

        pct_bullish  = n_bullish  / total * 100
        pct_bearish  = n_bearish  / total * 100
        pct_neutral  = (total - n_bullish - n_bearish) / total * 100
        ad_ratio     = n_bullish / max(n_bearish, 1)

        # % above EMA200
        above_ema200 = sum(
            1 for r in valid
            if r.signals.price_above_ema200
        )
        pct_above_ema200 = above_ema200 / total * 100

        # Breadth thrust = change from prior
        prior_bullish   = prior_snapshot.pct_bullish if prior_snapshot else pct_bullish
        breadth_thrust  = pct_bullish - prior_bullish

        # Regime breadth score [0, 1]:
        # 0.0 = fully bearish, 0.5 = balanced, 1.0 = fully bullish
        breadth_score = min(max(
            0.5 * (pct_bullish / 100) +
            0.3 * (pct_above_ema200 / 100) +
            0.2 * (1.0 - pct_bearish / 100),
        0.0), 1.0)

        # State classification
        state = self._classify_state(pct_bullish, pct_bearish, breadth_thrust)

        snap = BreadthSnapshot(
            universe              = universe,
            run_date              = run_date,
            total_stocks          = total,
            pct_above_ema200      = round(pct_above_ema200, 2),
            pct_bullish           = round(pct_bullish,      2),
            pct_bearish           = round(pct_bearish,      2),
            pct_neutral           = round(pct_neutral,      2),
            advance_decline_ratio = round(ad_ratio,         3),
            momentum_participation= round(n_momentum / total * 100, 2),
            breakout_participation= round(n_breakout / total * 100, 2),
            volatile_participation= round(n_volatile  / total * 100, 2),
            breadth_thrust        = round(breadth_thrust,   3),
            regime_breadth_score  = round(breadth_score,    4),
            breadth_state         = state,
        )

        logger.info(
            "Breadth [%s]: %s  bullish=%.1f%%  bearish=%.1f%%  "
            "AD=%.2f  above_ema200=%.1f%%  thrust=%.2f  score=%.3f",
            universe, state, pct_bullish, pct_bearish,
            ad_ratio, pct_above_ema200, breadth_thrust, breadth_score,
        )

        if pct_bullish > self.extreme_thr:
            logger.warning(
                "BREADTH EXTREME_UP [%s]: %.1f%% bullish — potential overbought",
                universe, pct_bullish,
            )
        elif pct_bearish > self.contracting_thr:
            logger.warning(
                "BREADTH EXTREME_DOWN [%s]: %.1f%% bearish — potential oversold",
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
        """Load the most recent breadth snapshot for prior-day thrust calculation."""
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
                advance_decline_ratio = float(row["advance_decline_ratio"]),
                momentum_participation= float(row["momentum_participation"]),
                breakout_participation= float(row["breakout_participation"]),
                volatile_participation= float(row["volatile_participation"]),
                breadth_thrust        = float(row["breadth_thrust"]),
                regime_breadth_score  = float(row["regime_breadth_score"]),
                breadth_state         = row["breadth_state"],
            )
        except Exception:
            return None

    def _classify_state(self, pct_bull, pct_bear, thrust) -> str:
        if pct_bull > self.extreme_thr:
            return "EXTREME_UP"
        if pct_bear > self.contracting_thr:
            return "EXTREME_DOWN"
        if pct_bull > self.expanding_thr:
            return "EXPANDING"
        if pct_bear > 35.0:
            return "CONTRACTING"
        return "NEUTRAL"