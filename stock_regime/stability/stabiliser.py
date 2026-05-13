"""
stock_regime/stability/stabiliser.py
======================================
Prevents single-bar regime flips by requiring a regime to persist for
``confirmation_bars`` consecutive bars before it is declared stable.

How it works
------------
Every run, the stabiliser receives:
  1. Today's raw classifications (list[StockRegimeResult])
  2. The regime history for each symbol (loaded from regime_history.parquet)

It computes for each symbol:
  - How many consecutive bars the raw regime has matched
  - Whether that count meets the confirmation threshold
  - What the current stable regime is (may lag the raw regime by N bars)

Key invariants
--------------
- UNCERTAIN never replaces a confirmed stable regime.
  If today is UNCERTAIN but the last stable regime was TREND_UP, the
  stable regime stays TREND_UP until a different non-UNCERTAIN regime
  is confirmed.

- regime_changed_today is True ONLY on the first bar where stable_regime
  transitions.  It is False on all subsequent bars of the same stable regime.

- regime_age_bars counts consecutive bars of the RAW regime, not stable.
  This is intentional: it tells the consumer how long the underlying signal
  has been present, not how long the confirmed label has been assigned.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date
from typing import Optional

import pandas as pd

from ..src.models import StockRegime, StockRegimeResult

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
#  Extended result model
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class StableRegimeResult:
    """
    StockRegimeResult extended with stability metadata.

    All fields from StockRegimeResult are preserved.  The stability
    fields are added on top.
    """
    # ── Original StockRegimeResult fields ────────────────────────────────────
    symbol:             str
    market:             str
    stock_regime:       StockRegime      # raw (unconfirmed) regime from the engine
    confidence:         float
    dimensional_scores: object           # DimensionalScores
    regime_scores:      dict
    signals:            object           # StockSignals
    indicators:         object           # StockIndicatorSnapshot
    error:              Optional[str]    = None

    # ── Stability fields ─────────────────────────────────────────────────────
    stable_regime:        StockRegime    = StockRegime.UNCERTAIN
    prior_stable_regime:  StockRegime    = StockRegime.UNCERTAIN
    regime_age_bars:      int            = 0   # consecutive bars of current RAW regime
    stable_regime_age:    int            = 0   # consecutive bars of STABLE regime
    regime_changed_today: bool           = False
    run_date:             Optional[date] = None

    @classmethod
    def from_result(
        cls,
        result:               StockRegimeResult,
        stable_regime:        StockRegime,
        prior_stable_regime:  StockRegime,
        regime_age_bars:      int,
        stable_regime_age:    int,
        regime_changed_today: bool,
        run_date:             Optional[date] = None,
    ) -> "StableRegimeResult":
        return cls(
            symbol             = result.symbol,
            market             = result.market,
            stock_regime       = result.stock_regime,
            confidence         = result.confidence,
            dimensional_scores = result.dimensional_scores,
            regime_scores      = result.regime_scores,
            signals            = result.signals,
            indicators         = result.indicators,
            error              = result.error,
            stable_regime      = stable_regime,
            prior_stable_regime= prior_stable_regime,
            regime_age_bars    = regime_age_bars,
            stable_regime_age  = stable_regime_age,
            regime_changed_today = regime_changed_today,
            run_date           = run_date,
        )

    def is_valid(self) -> bool:
        return self.error is None

    def to_dict(self) -> dict:
        """Serialise to the public JSON contract (extends StockRegimeResult.to_dict)."""
        import math
        snap = self.indicators
        ds   = self.dimensional_scores

        def _clean(v):
            if v is None: return None
            try:
                return None if math.isnan(v) or math.isinf(v) else round(v, 4)
            except: return None

        return {
            "symbol":               self.symbol,
            "market":               self.market,
            "stock_regime":         self.stock_regime.value,
            "stable_regime":        self.stable_regime.value,
            "prior_stable_regime":  self.prior_stable_regime.value,
            "confidence":           round(self.confidence, 4),
            "regime_age_bars":      self.regime_age_bars,
            "stable_regime_age":    self.stable_regime_age,
            "regime_changed_today": self.regime_changed_today,
            "scores": {
                "trend":      round(ds.trend, 4),
                "momentum":   round(ds.momentum, 4),
                "volatility": round(ds.volatility, 4),
            },
            "signals":  self.signals.to_dict(),
            "indicators": {
                "close":             _clean(snap.close),
                "ema20":             _clean(snap.ema20),
                "ema50":             _clean(snap.ema50),
                "ema200":            _clean(snap.ema200),
                "adx":               _clean(snap.adx),
                "atr":               _clean(snap.atr),
                "relative_strength": _clean(snap.relative_strength),
            },
        }


# ─────────────────────────────────────────────────────────────────────────────
#  Per-symbol history record (in-memory representation)
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class SymbolHistory:
    """
    Lightweight in-memory record of a symbol's recent regime history.
    Loaded from regime_history.parquet and updated after each run.
    """
    symbol:        str
    raw_regimes:   list[str]   = field(default_factory=list)  # last N raw regime values
    stable_regime: str         = StockRegime.UNCERTAIN.value
    stable_age:    int         = 0


# ─────────────────────────────────────────────────────────────────────────────
#  Stabiliser
# ─────────────────────────────────────────────────────────────────────────────

class RegimeStabiliser:
    """
    Applies a confirmation window to raw regime classifications.

    Parameters
    ----------
    confirmation_bars :
        Number of consecutive bars a raw regime must hold before it is
        declared stable.  Default 3.
    uncertain_propagates :
        When True, UNCERTAIN overwrites any stable regime immediately
        (no confirmation required).  When False (default), UNCERTAIN is
        treated as a non-event and the last stable regime is preserved.
    """

    def __init__(
        self,
        confirmation_bars:    int  = 3,
        uncertain_propagates: bool = False,
    ) -> None:
        self.confirmation_bars    = confirmation_bars
        self.uncertain_propagates = uncertain_propagates

    # ──────────────────────────────────────────────────────────────────────────
    #  Primary API
    # ──────────────────────────────────────────────────────────────────────────

    def apply(
        self,
        today_results: list[StockRegimeResult],
        history:       dict[str, SymbolHistory],
        run_date:      Optional[date] = None,
    ) -> list[StableRegimeResult]:
        """
        Apply the confirmation window to today's raw classifications.

        Parameters
        ----------
        today_results :
            Raw StockRegimeResult objects from StockRegimeEngine.
        history :
            Per-symbol history, keyed by symbol.  Load from
            regime_history.parquet before calling.  Will be mutated
            in-place to reflect today's run (so the caller can persist it).
        run_date :
            Date of this run.  Defaults to today.

        Returns
        -------
        list[StableRegimeResult]
            One per input symbol, in the same order.
        """
        run_date = run_date or date.today()
        stable_results: list[StableRegimeResult] = []

        for result in today_results:
            sym     = result.symbol
            hist    = history.get(sym, SymbolHistory(symbol=sym))
            history[sym] = hist   # ensure it's in the dict for next time

            stable = self._stabilise(result, hist, run_date)
            stable_results.append(stable)

            # Update history in-place
            hist.raw_regimes.append(result.stock_regime.value)
            # Keep only the last confirmation_bars + 1 entries
            hist.raw_regimes = hist.raw_regimes[-(self.confirmation_bars + 1):]
            hist.stable_regime = stable.stable_regime.value
            hist.stable_age    = stable.stable_regime_age

        changed = sum(1 for r in stable_results if r.regime_changed_today)
        logger.info(
            "RegimeStabiliser: %d symbols | %d regime changes today "
            "(confirmation=%d bars)",
            len(stable_results), changed, self.confirmation_bars,
        )
        return stable_results

    # ──────────────────────────────────────────────────────────────────────────
    #  Per-symbol logic
    # ──────────────────────────────────────────────────────────────────────────

    def _stabilise(
        self,
        result:   StockRegimeResult,
        hist:     SymbolHistory,
        run_date: date,
    ) -> StableRegimeResult:
        raw_today = result.stock_regime

        # Count consecutive bars of the current raw regime (including today)
        age = 1
        for prior_raw in reversed(hist.raw_regimes):
            if prior_raw == raw_today.value:
                age += 1
            else:
                break

        # Determine the new stable regime
        prior_stable = StockRegime(hist.stable_regime) if hist.stable_regime else StockRegime.UNCERTAIN

        if raw_today == StockRegime.UNCERTAIN and not self.uncertain_propagates:
            # UNCERTAIN does not replace an established stable regime
            new_stable = prior_stable
        elif age >= self.confirmation_bars:
            # Confirmation threshold met → accept the raw regime as stable
            new_stable = raw_today
        else:
            # Not yet confirmed → hold the prior stable regime
            new_stable = prior_stable

        changed   = new_stable != prior_stable
        new_s_age = (hist.stable_age + 1) if not changed else 1

        if changed:
            logger.info(
                "%s regime_change: %s → %s (raw=%s confirmed after %d bars)",
                result.symbol,
                prior_stable.value,
                new_stable.value,
                raw_today.value,
                age,
            )

        return StableRegimeResult.from_result(
            result               = result,
            stable_regime        = new_stable,
            prior_stable_regime  = prior_stable,
            regime_age_bars      = age,
            stable_regime_age    = new_s_age,
            regime_changed_today = changed,
            run_date             = run_date,
        )

    # ──────────────────────────────────────────────────────────────────────────
    #  History persistence helpers
    # ──────────────────────────────────────────────────────────────────────────

    @staticmethod
    def load_history(path: str | None) -> dict[str, SymbolHistory]:
        """
        Load per-symbol regime history from a parquet file.

        Returns an empty dict if the path is None or the file does not exist.
        """
        from pathlib import Path
        if path is None:
            return {}
        p = Path(path)
        if not p.exists():
            logger.debug("No regime history file at '%s' — starting fresh.", p)
            return {}

        df = pd.read_parquet(p)
        history: dict[str, SymbolHistory] = {}

        # We keep the last confirmation_bars rows per symbol
        for symbol, sym_df in df.groupby("symbol"):
            sym_df = sym_df.sort_values("run_date")
            raw_regimes = sym_df["raw_regime"].tolist()
            last_stable = sym_df["stable_regime"].iloc[-1]
            stable_age  = int(sym_df["stable_regime_age"].iloc[-1])
            history[symbol] = SymbolHistory(
                symbol        = symbol,
                raw_regimes   = raw_regimes,
                stable_regime = last_stable,
                stable_age    = stable_age,
            )
        logger.debug("Loaded history for %d symbols from '%s'.", len(history), p)
        return history

    @staticmethod
    def save_history(
        stable_results: list[StableRegimeResult],
        path: str,
        append: bool = True,
    ) -> None:
        """
        Persist today's stable regime results to parquet (append mode).

        Parameters
        ----------
        stable_results :
            Output of RegimeStabiliser.apply().
        path :
            Destination parquet file path.
        append :
            When True (default), read existing file and append.
            When False, overwrite.  Use False only for testing.
        """
        from pathlib import Path

        rows = []
        for r in stable_results:
            rows.append({
                "symbol":               r.symbol,
                "market":               r.market,
                "run_date":             r.run_date,
                "raw_regime":           r.stock_regime.value,
                "stable_regime":        r.stable_regime.value,
                "prior_stable_regime":  r.prior_stable_regime.value,
                "confidence":           r.confidence,
                "regime_age_bars":      r.regime_age_bars,
                "stable_regime_age":    r.stable_regime_age,
                "regime_changed_today": r.regime_changed_today,
            })

        new_df = pd.DataFrame(rows)
        p      = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)

        if append and p.exists():
            existing = pd.read_parquet(p)
            combined = pd.concat([existing, new_df], ignore_index=True)
            combined.drop_duplicates(
                subset=["symbol", "run_date"], keep="last", inplace=True
            )
            combined.to_parquet(p, engine="pyarrow", compression="snappy", index=False)
        else:
            new_df.to_parquet(p, engine="pyarrow", compression="snappy", index=False)

        logger.info("RegimeStabiliser: saved history → '%s' (%d rows).", p, len(new_df))