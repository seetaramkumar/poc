"""
stock_regime/src/models.py
==========================
Typed data contracts for every layer of the Stock Regime Engine pipeline.

Nothing outside this file should define ad-hoc dicts for engine I/O.
All structured data flowing through the pipeline uses these types.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


# ─────────────────────────────────────────────────────────────
#  Stock Regime Enum
# ─────────────────────────────────────────────────────────────

class StockRegime(str, Enum):
    """
    Eight canonical stock-level market states.

    Inheriting from ``str`` makes JSON serialisation trivial and
    allows direct string comparison without calling ``.value``.
    """
    TREND_UP       = "TREND_UP"        # confirmed uptrend with momentum
    TREND_DOWN     = "TREND_DOWN"      # confirmed downtrend with momentum
    RANGE          = "RANGE"           # sideways / consolidation
    MOMENTUM       = "MOMENTUM"        # strong relative outperformance
    BREAKOUT_SETUP = "BREAKOUT_SETUP"  # volatility compression near key level
    VOLATILE       = "VOLATILE"        # elevated daily range vs history
    QUIET          = "QUIET"           # compressed daily range vs history
    UNCERTAIN      = "UNCERTAIN"       # no regime cleared min confidence


# ─────────────────────────────────────────────────────────────
#  Market Regime Input (consumed from Market Regime Engine)
# ─────────────────────────────────────────────────────────────

@dataclass
class MarketRegimeInput:
    """
    Structured container for the Market Regime Engine output that is
    consumed as context by the Stock Regime Engine.

    Parameters
    ----------
    regime :
        String value of the market regime (e.g. ``"BULLISH_TREND"``).
    confidence :
        Confidence score of the market regime in [0, 1].
    """
    regime:     str   = "UNCERTAIN"
    confidence: float = 0.0

    @classmethod
    def from_dict(cls, d: dict) -> "MarketRegimeInput":
        """Convenience constructor from the Market Regime Engine JSON output."""
        return cls(
            regime=d.get("regime", "UNCERTAIN"),
            confidence=float(d.get("confidence", 0.0)),
        )

    def __repr__(self) -> str:
        return f"MarketRegimeInput(regime={self.regime!r}, confidence={self.confidence:.2f})"


# ─────────────────────────────────────────────────────────────
#  Indicator Snapshot (one bar per stock)
# ─────────────────────────────────────────────────────────────

@dataclass
class StockIndicatorSnapshot:
    """
    All computed indicator values for the most-recent bar of one stock.

    Optional fields gracefully degrade when insufficient history exists
    (e.g. first 200 bars lack EMA-200, first 252 bars lack a full 52w high).
    """
    # Price
    close: float = float("nan")

    # Exponential Moving Averages
    ema20:  Optional[float] = None
    ema50:  Optional[float] = None
    ema200: Optional[float] = None

    # EMA slopes (% change over slope_window bars)
    ema20_slope: Optional[float] = None
    ema50_slope: Optional[float] = None

    # Trend strength
    adx: Optional[float] = None

    # Volatility
    atr:    Optional[float] = None   # current ATR
    atr_ma: Optional[float] = None   # rolling mean of ATR

    # Volume
    volume:    Optional[float] = None
    volume_ma: Optional[float] = None

    # Relative strength vs benchmark (ratio of returns)
    # > 1.0 → outperforming; < 1.0 → underperforming; None → no benchmark
    relative_strength: Optional[float] = None

    # Rolling high proxy for near-high breakout detection
    high_52w: Optional[float] = None

    def is_complete(self) -> bool:
        """Return ``True`` only when every core field has a valid value."""
        core_fields = [
            self.ema20, self.ema50, self.ema200,
            self.ema20_slope, self.ema50_slope,
            self.adx, self.atr, self.atr_ma,
            self.volume, self.volume_ma,
        ]
        return all(
            v is not None and not (isinstance(v, float) and math.isnan(v))
            for v in core_fields
        )

    def to_dict(self) -> dict:
        """Serialise to plain dict for persistence / JSON output."""
        return {
            "close":              self.close,
            "ema20":              self.ema20,
            "ema50":              self.ema50,
            "ema200":             self.ema200,
            "ema20_slope":        self.ema20_slope,
            "ema50_slope":        self.ema50_slope,
            "adx":                self.adx,
            "atr":                self.atr,
            "atr_ma":             self.atr_ma,
            "volume":             self.volume,
            "volume_ma":          self.volume_ma,
            "relative_strength":  self.relative_strength,
            "high_52w":           self.high_52w,
        }


# ─────────────────────────────────────────────────────────────
#  Signals (boolean flags derived from snapshot)
# ─────────────────────────────────────────────────────────────

@dataclass
class StockSignals:
    """
    Human-readable boolean signals extracted from a StockIndicatorSnapshot.

    Each field corresponds to one market condition consumed by the scorer.
    Keeping signals separate from scores lets callers inspect *why* a regime
    was assigned.
    """
    # Trend direction
    price_above_ema200: bool = False
    price_below_ema200: bool = False
    ema20_above_ema50:  bool = False
    ema20_below_ema50:  bool = False

    # EMA flatness (sideways / range detection)
    ema20_flat: bool = False
    ema50_flat: bool = False

    # Trend strength
    adx_strong: bool = False   # ADX ≥ strong_trend threshold
    adx_weak:   bool = False   # ADX < weak_trend threshold

    # Volatility regime
    atr_high:       bool = False   # ATR/ATR_MA ≥ volatile_ratio
    atr_low:        bool = False   # ATR/ATR_MA ≤ quiet_ratio
    atr_compressed: bool = False   # ATR/ATR_MA ≤ compressed_ratio (setup)
    atr_expanding:  bool = False   # ATR/ATR_MA > expanding_ratio

    # Volume
    volume_confirmed: bool = False  # volume ≥ vol_MA × surge_ratio

    # Relative strength
    rs_positive: bool = False   # RS ≥ positive threshold
    rs_negative: bool = False   # RS ≤ negative threshold
    rs_strong:   bool = False   # RS ≥ strong threshold

    # Breakout location
    price_near_52w_high: bool = False   # close within near_high_pct of high

    def to_dict(self) -> dict:
        """Serialise to plain dict for persistence / JSON output."""
        return {
            "price_above_ema200":  self.price_above_ema200,
            "price_below_ema200":  self.price_below_ema200,
            "ema20_above_ema50":   self.ema20_above_ema50,
            "ema20_below_ema50":   self.ema20_below_ema50,
            "ema20_flat":          self.ema20_flat,
            "ema50_flat":          self.ema50_flat,
            "adx_strong":          self.adx_strong,
            "adx_weak":            self.adx_weak,
            "atr_high":            self.atr_high,
            "atr_low":             self.atr_low,
            "atr_compressed":      self.atr_compressed,
            "atr_expanding":       self.atr_expanding,
            "volume_confirmed":    self.volume_confirmed,
            "rs_positive":         self.rs_positive,
            "rs_negative":         self.rs_negative,
            "rs_strong":           self.rs_strong,
            "price_near_52w_high": self.price_near_52w_high,
        }


# ─────────────────────────────────────────────────────────────
#  Dimensional Scores (for ranking, not classification)
# ─────────────────────────────────────────────────────────────

@dataclass
class DimensionalScores:
    """
    Three orthogonal scores that describe a stock's current character.

    These scores are computed separately from the regime classification
    scores and are intended for ranking and downstream consumers
    (strategy router, risk engine).

    All scores are in [0, 1].
    """
    trend:      float = 0.0   # EMA alignment + ADX + RS; high = strong uptrend
    momentum:   float = 0.0   # RS + volume + ADX + ATR expansion
    volatility: float = 0.0   # ATR ratio normalised to [0, 1]

    def to_dict(self) -> dict:
        return {
            "trend":      round(self.trend,      4),
            "momentum":   round(self.momentum,   4),
            "volatility": round(self.volatility, 4),
        }


# ─────────────────────────────────────────────────────────────
#  Final engine output (one per stock per run)
# ─────────────────────────────────────────────────────────────

@dataclass
class StockRegimeResult:
    """
    Structured output returned by the engine for a single stock.

    This object is the primary data contract between the Stock Regime
    Engine and all downstream consumers (strategy router, signal engine,
    risk engine, persistence layer).

    Attributes
    ----------
    symbol :
        Stock ticker / identifier.
    market :
        Universe label, e.g. ``"NIFTY500"`` or ``"SP500"``.
    stock_regime :
        The classified stock regime.
    confidence :
        Score of the winning regime in [0, 1], post-context adjustment.
    dimensional_scores :
        Trend / momentum / volatility scores for ranking.
    regime_scores :
        Raw weighted score for every regime (transparency).
    signals :
        Boolean flags that drove the classification.
    indicators :
        Raw computed indicator values (for inspection / persistence).
    error :
        Set to a non-empty string if processing failed for this stock.
    """
    symbol:            str
    market:            str
    stock_regime:      StockRegime
    confidence:        float
    dimensional_scores: DimensionalScores  = field(default_factory=DimensionalScores)
    regime_scores:     dict[str, float]    = field(default_factory=dict)
    signals:           StockSignals        = field(default_factory=StockSignals)
    indicators:        StockIndicatorSnapshot = field(default_factory=StockIndicatorSnapshot)
    error:             Optional[str]       = None

    def is_valid(self) -> bool:
        """Return ``True`` when the result was produced without error."""
        return self.error is None

    def to_dict(self) -> dict:
        """Serialise to the public JSON contract."""
        snap = self.indicators

        def _clean(v: Optional[float]) -> Optional[float]:
            """Replace NaN / inf with None so json.dumps never chokes."""
            if v is None:
                return None
            try:
                return None if math.isnan(v) or math.isinf(v) else round(v, 4)
            except (TypeError, ValueError):
                return None

        return {
            "symbol":        self.symbol,
            "market":        self.market,
            "stock_regime":  self.stock_regime.value,
            "confidence":    round(self.confidence, 4),
            "scores":        self.dimensional_scores.to_dict(),
            "regime_scores": {k: round(v, 4) for k, v in self.regime_scores.items()},
            "signals":       self.signals.to_dict(),
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
