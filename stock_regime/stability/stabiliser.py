"""
stock_regime/stability/stabiliser.py
======================================
Regime confirmation window + confidence smoothing + hysteresis.

Changes from previous version
------------------------------
- Added EWM confidence smoothing (smoothing_alpha from config)
- Added hysteresis: new regime needs smoothed confidence >
  current regime confidence + hysteresis_threshold to switch
- Added oscillation detection via persistence_window
- Added regime_switch_threshold: min smoothed confidence to accept
- StableRegimeResult gains: smoothed_confidence, oscillation_detected
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date
from typing import Optional

import pandas as pd

from ..src.models import StockRegime, StockRegimeResult

logger = logging.getLogger(__name__)


@dataclass
class StableRegimeResult:
    """StockRegimeResult extended with stability metadata."""
    # Original fields
    symbol:             str
    market:             str
    stock_regime:       StockRegime
    confidence:         float
    dimensional_scores: object
    regime_scores:      dict
    signals:            object
    indicators:         object
    error:              Optional[str] = None

    # Stability fields
    stable_regime:        StockRegime = StockRegime.UNCERTAIN
    prior_stable_regime:  StockRegime = StockRegime.UNCERTAIN
    regime_age_bars:      int         = 0
    stable_regime_age:    int         = 0
    regime_changed_today: bool        = False
    run_date:             Optional[date] = None

    # NEW: smoothing / hysteresis fields
    smoothed_confidence:  float = 0.0
    oscillation_detected: bool  = False

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
        smoothed_confidence:  float = 0.0,
        oscillation_detected: bool  = False,
    ) -> "StableRegimeResult":
        return cls(
            symbol               = result.symbol,
            market               = result.market,
            stock_regime         = result.stock_regime,
            confidence           = result.confidence,
            dimensional_scores   = result.dimensional_scores,
            regime_scores        = result.regime_scores,
            signals              = result.signals,
            indicators           = result.indicators,
            error                = result.error,
            stable_regime        = stable_regime,
            prior_stable_regime  = prior_stable_regime,
            regime_age_bars      = regime_age_bars,
            stable_regime_age    = stable_regime_age,
            regime_changed_today = regime_changed_today,
            run_date             = run_date,
            smoothed_confidence  = smoothed_confidence,
            oscillation_detected = oscillation_detected,
        )

    def is_valid(self) -> bool:
        return self.error is None

    def to_dict(self) -> dict:
        import math
        snap = self.indicators
        ds   = self.dimensional_scores

        def _clean(v):
            if v is None:
                return None
            try:
                return None if math.isnan(v) or math.isinf(v) else round(v, 4)
            except Exception:
                return None

        return {
            "symbol":               self.symbol,
            "market":               self.market,
            "stock_regime":         self.stock_regime.value,
            "stable_regime":        self.stable_regime.value,
            "prior_stable_regime":  self.prior_stable_regime.value,
            "confidence":           round(self.confidence, 4),
            "smoothed_confidence":  round(self.smoothed_confidence, 4),
            "regime_age_bars":      self.regime_age_bars,
            "stable_regime_age":    self.stable_regime_age,
            "regime_changed_today": self.regime_changed_today,
            "oscillation_detected": self.oscillation_detected,
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
                "roc_10":            _clean(snap.roc_10),
                "roc_21":            _clean(snap.roc_21),
                "rs_3m":             _clean(snap.rs_3m),
                "rs_trend":          _clean(snap.rs_trend),
            },
        }


@dataclass
class SymbolHistory:
    """In-memory record of a symbol's recent regime history."""
    symbol:              str
    raw_regimes:         list[str]   = field(default_factory=list)
    raw_confidences:     list[float] = field(default_factory=list)
    stable_regime:       str         = StockRegime.UNCERTAIN.value
    stable_age:          int         = 0
    smoothed_confidence: float       = 0.0


class RegimeStabiliser:
    """
    Applies confirmation window, EWM confidence smoothing, and hysteresis.

    Parameters
    ----------
    confirmation_bars : int
        Bars a raw regime must persist before becoming stable.
    uncertain_propagates : bool
        When False (default), UNCERTAIN never overwrites a stable regime.
    smoothing_enabled : bool
        Apply EWM smoothing to confidence before switching decisions.
    smoothing_alpha : float
        EWM decay factor (0 < alpha < 1). Lower = smoother.
    hysteresis_threshold : float
        New regime needs smoothed confidence > current + this to switch.
    persistence_window : int
        Look-back bars for oscillation detection.
    regime_switch_threshold : float
        Minimum smoothed confidence to accept a regime switch.
    """

    def __init__(
        self,
        confirmation_bars:      int   = 3,
        uncertain_propagates:   bool  = False,
        smoothing_enabled:      bool  = True,
        smoothing_alpha:        float = 0.30,
        hysteresis_threshold:   float = 0.05,
        persistence_window:     int   = 5,
        regime_switch_threshold: float = 0.55,
    ) -> None:
        self.confirmation_bars       = confirmation_bars
        self.uncertain_propagates    = uncertain_propagates
        self.smoothing_enabled       = smoothing_enabled
        self.smoothing_alpha         = smoothing_alpha
        self.hysteresis_threshold    = hysteresis_threshold
        self.persistence_window      = persistence_window
        self.regime_switch_threshold = regime_switch_threshold

    # ──────────────────────────────────────────────────────────────
    #  Public API
    # ──────────────────────────────────────────────────────────────

    def apply(
        self,
        today_results: list[StockRegimeResult],
        history:       dict[str, SymbolHistory],
        run_date:      Optional[date] = None,
    ) -> list[StableRegimeResult]:
        run_date = run_date or date.today()
        stable_results: list[StableRegimeResult] = []

        for result in today_results:
            sym  = result.symbol
            hist = history.get(sym, SymbolHistory(symbol=sym))
            history[sym] = hist

            stable = self._stabilise(result, hist, run_date)
            stable_results.append(stable)

            # Update history in-place
            hist.raw_regimes.append(result.stock_regime.value)
            hist.raw_confidences.append(result.confidence)
            keep = self.confirmation_bars + self.persistence_window + 1
            hist.raw_regimes     = hist.raw_regimes[-keep:]
            hist.raw_confidences = hist.raw_confidences[-keep:]
            hist.stable_regime       = stable.stable_regime.value
            hist.stable_age          = stable.stable_regime_age
            hist.smoothed_confidence = stable.smoothed_confidence

        changed = sum(1 for r in stable_results if r.regime_changed_today)
        oscillating = sum(1 for r in stable_results if r.oscillation_detected)
        logger.info(
            "RegimeStabiliser: %d symbols | %d changes | %d oscillating "
            "(confirmation=%d bars  smoothing=%s)",
            len(stable_results), changed, oscillating,
            self.confirmation_bars,
            "on" if self.smoothing_enabled else "off",
        )
        return stable_results

    # ──────────────────────────────────────────────────────────────
    #  Per-symbol logic
    # ──────────────────────────────────────────────────────────────

    def _stabilise(
        self,
        result:   StockRegimeResult,
        hist:     SymbolHistory,
        run_date: date,
    ) -> StableRegimeResult:
        raw_today   = result.stock_regime
        raw_conf    = result.confidence
        prior_stable = StockRegime(hist.stable_regime) if hist.stable_regime else StockRegime.UNCERTAIN

        # ── 1. EWM confidence smoothing ──────────────────────────────
        if self.smoothing_enabled and hist.smoothed_confidence > 0:
            alpha = self.smoothing_alpha
            smoothed_conf = alpha * raw_conf + (1 - alpha) * hist.smoothed_confidence
        else:
            smoothed_conf = raw_conf

        # ── 2. Consecutive-bar count for raw regime ──────────────────
        age = 1
        for prior_raw in reversed(hist.raw_regimes):
            if prior_raw == raw_today.value:
                age += 1
            else:
                break

        # ── 3. Oscillation detection ─────────────────────────────────
        recent_regimes = hist.raw_regimes[-(self.persistence_window):]
        unique_recent  = len(set(recent_regimes))
        oscillation    = (
            len(recent_regimes) >= self.persistence_window and
            unique_recent >= max(self.persistence_window // 2, 2)
        )

        # ── 4. Determine new stable regime ───────────────────────────
        if raw_today == StockRegime.UNCERTAIN and not self.uncertain_propagates:
            new_stable = prior_stable

        elif age >= self.confirmation_bars:
            # Hysteresis: only switch if smoothed confidence is meaningfully
            # higher than the current regime's smoothed confidence AND meets
            # the minimum switch threshold
            meets_threshold = smoothed_conf >= self.regime_switch_threshold
            hysteresis_ok   = (
                raw_today == prior_stable or
                smoothed_conf >= hist.smoothed_confidence + self.hysteresis_threshold
            )

            if meets_threshold and hysteresis_ok and not oscillation:
                new_stable = raw_today
            else:
                new_stable = prior_stable
        else:
            new_stable = prior_stable

        changed   = new_stable != prior_stable
        new_s_age = (hist.stable_age + 1) if not changed else 1

        if changed:
            logger.info(
                "%s regime_change: %s → %s  (raw=%s  age=%d  conf=%.2f  smooth=%.2f)",
                result.symbol,
                prior_stable.value,
                new_stable.value,
                raw_today.value,
                age,
                raw_conf,
                smoothed_conf,
            )
        elif oscillation:
            logger.debug(
                "%s oscillation_detected: %d unique regimes in last %d bars",
                result.symbol, unique_recent, self.persistence_window,
            )

        return StableRegimeResult.from_result(
            result               = result,
            stable_regime        = new_stable,
            prior_stable_regime  = prior_stable,
            regime_age_bars      = age,
            stable_regime_age    = new_s_age,
            regime_changed_today = changed,
            run_date             = run_date,
            smoothed_confidence  = round(smoothed_conf, 4),
            oscillation_detected = oscillation,
        )

    # ──────────────────────────────────────────────────────────────
    #  History I/O
    # ──────────────────────────────────────────────────────────────

    @staticmethod
    def load_history(path: str | None) -> dict[str, SymbolHistory]:
        from pathlib import Path
        if path is None:
            return {}
        p = Path(path)
        if not p.exists():
            logger.debug("No regime history at '%s' — starting fresh.", p)
            return {}

        df = pd.read_parquet(p)
        history: dict[str, SymbolHistory] = {}
        for symbol, sym_df in df.groupby("symbol"):
            sym_df = sym_df.sort_values("run_date")
            history[symbol] = SymbolHistory(
                symbol               = symbol,
                raw_regimes          = sym_df["raw_regime"].tolist(),
                raw_confidences      = sym_df["confidence"].tolist(),
                stable_regime        = sym_df["stable_regime"].iloc[-1],
                stable_age           = int(sym_df["stable_regime_age"].iloc[-1]),
                smoothed_confidence  = float(sym_df.get("smoothed_confidence", pd.Series([0.0])).iloc[-1]),
            )
        logger.debug("Loaded history for %d symbols.", len(history))
        return history

    @staticmethod
    def save_history(
        stable_results: list[StableRegimeResult],
        path: str,
        append: bool = True,
    ) -> None:
        from pathlib import Path
        rows = [{
            "symbol":               r.symbol,
            "market":               r.market,
            "run_date":             r.run_date,
            "raw_regime":           r.stock_regime.value,
            "stable_regime":        r.stable_regime.value,
            "prior_stable_regime":  r.prior_stable_regime.value,
            "confidence":           r.confidence,
            "smoothed_confidence":  r.smoothed_confidence,
            "regime_age_bars":      r.regime_age_bars,
            "stable_regime_age":    r.stable_regime_age,
            "regime_changed_today": r.regime_changed_today,
            "oscillation_detected": r.oscillation_detected,
        } for r in stable_results]

        new_df = pd.DataFrame(rows)
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)

        if append and p.exists():
            existing = pd.read_parquet(p)
            combined = pd.concat([existing, new_df], ignore_index=True)
            combined.drop_duplicates(subset=["symbol", "run_date"], keep="last", inplace=True)
            combined.to_parquet(p, engine="pyarrow", compression="snappy", index=False)
        else:
            new_df.to_parquet(p, engine="pyarrow", compression="snappy", index=False)

        logger.info("Saved regime history → '%s' (%d rows).", p, len(new_df))