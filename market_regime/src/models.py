"""
models.py — Data contracts for the Market Regime Engine
========================================================
All structured data flowing through the pipeline is expressed as
typed dataclasses or Enums.  Nothing outside this file should
define its own ad-hoc dicts for engine I/O.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


# ─────────────────────────────────────────────
#  Regime Enum
# ─────────────────────────────────────────────

class MarketRegime(str, Enum):
    """
    The five canonical market states the engine can classify.
    Inheriting from str makes JSON serialisation trivial.
    """
    BULLISH_TREND  = "BULLISH_TREND"
    BEARISH_TREND  = "BEARISH_TREND"
    SIDEWAYS       = "SIDEWAYS"
    VOLATILE       = "VOLATILE"
    QUIET          = "QUIET"
    UNCERTAIN      = "UNCERTAIN"   # fallback when confidence is too low


# ─────────────────────────────────────────────
#  Indicator snapshot (one row / one bar)
# ─────────────────────────────────────────────

@dataclass
class IndicatorSnapshot:
    """
    All computed indicator values for the most-recent bar.
    All fields are optional so callers can build the object
    incrementally and the engine can degrade gracefully when
    a value cannot be computed (e.g. not enough history).
    """
    # Price
    close: float = float("nan")

    # EMAs
    ema20:  Optional[float] = None
    ema50:  Optional[float] = None
    ema200: Optional[float] = None

    # EMA slopes  (% change over N bars, signed)
    ema20_slope:  Optional[float] = None
    ema50_slope:  Optional[float] = None

    # ADX (trend strength, 0-100)
    adx: Optional[float] = None

    # ATR (absolute volatility)
    atr:    Optional[float] = None
    atr_ma: Optional[float] = None   # rolling average of ATR

    # Volume
    volume:    Optional[float] = None
    volume_ma: Optional[float] = None   # rolling average of volume

    def is_complete(self) -> bool:
        """Return True only when every field has a valid value."""
        return all(
            v is not None
            for v in [
                self.ema20, self.ema50, self.ema200,
                self.ema20_slope, self.ema50_slope,
                self.adx, self.atr, self.atr_ma,
                self.volume, self.volume_ma,
            ]
        )


# ─────────────────────────────────────────────
#  Signals (boolean flags derived from snapshot)
# ─────────────────────────────────────────────

@dataclass
class RegimeSignals:
    """
    Human-readable boolean signals extracted from the indicator snapshot.
    Each signal corresponds to one market condition that the scorer uses
    as a feature.  Keeping signals separate from the score lets callers
    inspect *why* a regime was assigned.
    """
    # Trend direction
    price_above_ema200: bool = False
    price_below_ema200: bool = False
    ema20_above_ema50:  bool = False
    ema20_below_ema50:  bool = False

    # Trend strength
    adx_strong: bool = False   # ADX > strong_trend threshold
    adx_weak:   bool = False   # ADX < weak_trend  threshold

    # EMA flatness (used for SIDEWAYS)
    ema20_flat: bool = False
    ema50_flat: bool = False

    # Volatility
    atr_high: bool = False   # ATR / ATR_MA > volatile_ratio
    atr_low:  bool = False   # ATR / ATR_MA < quiet_ratio

    # Volume
    volume_confirms: bool = False   # volume > vol_MA * surge_ratio

    def to_dict(self) -> dict:
        """Serialise to a plain dict (for JSON output)."""
        return {
            "price_above_ema200": self.price_above_ema200,
            "price_below_ema200": self.price_below_ema200,
            "ema20_above_ema50":  self.ema20_above_ema50,
            "ema20_below_ema50":  self.ema20_below_ema50,
            "adx_strong":         self.adx_strong,
            "adx_weak":           self.adx_weak,
            "ema20_flat":         self.ema20_flat,
            "ema50_flat":         self.ema50_flat,
            "atr_high":           self.atr_high,
            "atr_low":            self.atr_low,
            "volume_confirms":    self.volume_confirms,
        }


# ─────────────────────────────────────────────
#  Final engine output
# ─────────────────────────────────────────────

@dataclass
class RegimeResult:
    """
    The structured output returned by the engine for a single bar.

    Attributes
    ----------
    regime:       The classified market regime.
    confidence:   Score in [0, 1]; higher = stronger regime match.
    signals:      The boolean features that drove the decision.
    scores:       Per-regime weighted scores (for transparency).
    indicator_snapshot: The raw computed indicator values.
    """
    regime:     MarketRegime
    confidence: float
    signals:    RegimeSignals
    scores:     dict = field(default_factory=dict)
    indicator_snapshot: Optional[IndicatorSnapshot] = None

    def to_dict(self) -> dict:
        """Serialise to the public JSON contract."""
        return {
            "regime":     self.regime.value,
            "confidence": round(self.confidence, 4),
            "signals":    self.signals.to_dict(),
            "scores":     {k: round(v, 4) for k, v in self.scores.items()},
        }
