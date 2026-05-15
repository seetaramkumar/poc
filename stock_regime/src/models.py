"""
stock_regime/src/models.py
==========================
Typed data contracts for every layer of the Stock Regime Engine pipeline.

Changes from previous version
------------------------------
- StockIndicatorSnapshot: added roc_10, roc_21, acceleration, rs_1m/3m/6m,
  rs_trend, higher_highs_count, ema_distance_pct
- StockSignals: added roc_positive, roc_accelerating, rs_improving,
  rs_weakening, higher_highs, ema_extended
- ContinuousScores dataclass (NEW) — replaces binary dimensional scoring
- DimensionalScores: now wraps ContinuousScores; backward-compatible
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class StockRegime(str, Enum):
    TREND_UP       = "TREND_UP"
    TREND_DOWN     = "TREND_DOWN"
    RANGE          = "RANGE"
    MOMENTUM       = "MOMENTUM"
    BREAKOUT_SETUP = "BREAKOUT_SETUP"
    VOLATILE       = "VOLATILE"
    QUIET          = "QUIET"
    UNCERTAIN      = "UNCERTAIN"


@dataclass
class MarketRegimeInput:
    regime:     str   = "UNCERTAIN"
    confidence: float = 0.0

    @classmethod
    def from_dict(cls, d: dict) -> "MarketRegimeInput":
        return cls(
            regime=d.get("regime", "UNCERTAIN"),
            confidence=float(d.get("confidence", 0.0)),
        )

    def __repr__(self) -> str:
        return f"MarketRegimeInput(regime={self.regime!r}, confidence={self.confidence:.2f})"


@dataclass
class StockIndicatorSnapshot:
    """
    All computed indicator values for one bar.

    New fields added for continuous scoring and trend/momentum separation:
    roc_10, roc_21, acceleration  — rate-of-change and momentum acceleration
    rs_1m, rs_3m, rs_6m           — multi-period relative strength
    rs_trend                       — slope of rs_3m (improving vs weakening)
    higher_highs_count             — trend quality (count of higher highs)
    ema_distance_pct               — (close - ema200) / ema200
    """
    # Core
    close: float = float("nan")

    # EMAs
    ema20:  Optional[float] = None
    ema50:  Optional[float] = None
    ema200: Optional[float] = None

    # EMA slopes
    ema20_slope: Optional[float] = None
    ema50_slope: Optional[float] = None

    # Trend strength
    adx: Optional[float] = None

    # Volatility
    atr:    Optional[float] = None
    atr_ma: Optional[float] = None

    # Volume
    volume:    Optional[float] = None
    volume_ma: Optional[float] = None

    # Legacy RS (kept for compatibility)
    relative_strength: Optional[float] = None

    # Rolling high
    high_52w: Optional[float] = None

    # NEW: Rate of change (momentum separation)
    roc_10:       Optional[float] = None
    roc_21:       Optional[float] = None
    acceleration: Optional[float] = None   # roc_10 - roc_21

    # NEW: Multi-period RS
    rs_1m:   Optional[float] = None   # 21-bar
    rs_3m:   Optional[float] = None   # 63-bar (replaces relative_strength going forward)
    rs_6m:   Optional[float] = None   # 126-bar
    rs_trend: Optional[float] = None  # slope of rs_3m over last N bars

    # NEW: Trend quality
    higher_highs_count: Optional[int]   = None
    ema_distance_pct:   Optional[float] = None   # (close - ema200) / ema200

    def is_complete(self) -> bool:
        core = [
            self.ema20, self.ema50, self.ema200,
            self.ema20_slope, self.ema50_slope,
            self.adx, self.atr, self.atr_ma,
            self.volume, self.volume_ma,
        ]
        return all(
            v is not None and not (isinstance(v, float) and math.isnan(v))
            for v in core
        )

    def to_dict(self) -> dict:
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
            "roc_10":             self.roc_10,
            "roc_21":             self.roc_21,
            "acceleration":       self.acceleration,
            "rs_1m":              self.rs_1m,
            "rs_3m":              self.rs_3m,
            "rs_6m":              self.rs_6m,
            "rs_trend":           self.rs_trend,
            "higher_highs_count": self.higher_highs_count,
            "ema_distance_pct":   self.ema_distance_pct,
        }


@dataclass
class StockSignals:
    """Boolean signals. Extended with momentum and trend-quality signals."""
    # Trend direction
    price_above_ema200: bool = False
    price_below_ema200: bool = False
    ema20_above_ema50:  bool = False
    ema20_below_ema50:  bool = False

    # EMA flatness
    ema20_flat: bool = False
    ema50_flat: bool = False

    # Trend strength
    adx_strong: bool = False
    adx_weak:   bool = False

    # Volatility
    atr_high:       bool = False
    atr_low:        bool = False
    atr_compressed: bool = False
    atr_expanding:  bool = False

    # Volume
    volume_confirmed: bool = False

    # RS (legacy)
    rs_positive: bool = False
    rs_negative: bool = False
    rs_strong:   bool = False

    # Breakout
    price_near_52w_high: bool = False

    # NEW: Momentum signals
    roc_positive:     bool = False
    roc_accelerating: bool = False

    # NEW: RS improvement
    rs_improving: bool = False
    rs_weakening: bool = False

    # NEW: Trend quality
    higher_highs: bool = False
    ema_extended: bool = False

    def to_dict(self) -> dict:
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
            "roc_positive":        self.roc_positive,
            "roc_accelerating":    self.roc_accelerating,
            "rs_improving":        self.rs_improving,
            "rs_weakening":        self.rs_weakening,
            "higher_highs":        self.higher_highs,
            "ema_extended":        self.ema_extended,
        }


@dataclass
class ContinuousScores:
    """
    Continuous normalized component scores in [0, 1] computed directly
    from raw indicator values — NOT from boolean signals.

    These drive the dimensional scores so rankings have realistic gradient
    rather than binary saturation (trend=1.0 / momentum=1.0 too often).

    Calibration reference
    ---------------------
    adx_score:           0 at ADX=0,  0.5 at ADX=25, 1.0 at ADX=50
    ema_alignment_score: 0.33 per bullish EMA pair (3 pairs total)
    ema_distance_score:  0 at distance=0, 1.0 at distance=+20%  (capped)
    atr_expansion_score: 0 at ratio=0.5,  0.5 at ratio=1.0, 1.0 at ratio=2.0
    rs_score:            0 at RS=0.90,     0.5 at RS=1.0,   1.0 at RS=1.10
    rs_trend_score:      0=falling,  0.5=flat,  1.0=strongly rising
    roc_score:           0 at roc_10=-5%,  0.5 at 0%,       1.0 at +5%
    volume_score:        0 at vol/vol_ma=0.5, 0.5 at 1.0,   1.0 at 2.0
    """
    adx_score:           float = 0.0
    ema_alignment_score: float = 0.0
    ema_distance_score:  float = 0.0
    atr_expansion_score: float = 0.5   # neutral default
    rs_score:            float = 0.5   # neutral default
    rs_trend_score:      float = 0.5   # neutral default
    roc_score:           float = 0.5   # neutral default
    volume_score:        float = 0.0

    def to_dict(self) -> dict:
        return {
            "adx_score":           round(self.adx_score,           4),
            "ema_alignment_score": round(self.ema_alignment_score,  4),
            "ema_distance_score":  round(self.ema_distance_score,   4),
            "atr_expansion_score": round(self.atr_expansion_score,  4),
            "rs_score":            round(self.rs_score,             4),
            "rs_trend_score":      round(self.rs_trend_score,       4),
            "roc_score":           round(self.roc_score,            4),
            "volume_score":        round(self.volume_score,         4),
        }


@dataclass
class DimensionalScores:
    """
    Three orthogonal scores [0, 1] for ranking. Now derived from
    ContinuousScores — no longer binary-saturated.
    """
    trend:      float = 0.0
    momentum:   float = 0.0
    volatility: float = 0.0
    continuous: Optional[ContinuousScores] = field(default=None, compare=False)

    def to_dict(self) -> dict:
        d = {
            "trend":      round(self.trend,      4),
            "momentum":   round(self.momentum,   4),
            "volatility": round(self.volatility, 4),
        }
        if self.continuous is not None:
            d["continuous"] = self.continuous.to_dict()
        return d


@dataclass
class StockRegimeResult:
    symbol:             str
    market:             str
    stock_regime:       StockRegime
    confidence:         float
    dimensional_scores: DimensionalScores      = field(default_factory=DimensionalScores)
    regime_scores:      dict[str, float]       = field(default_factory=dict)
    signals:            StockSignals           = field(default_factory=StockSignals)
    indicators:         StockIndicatorSnapshot = field(default_factory=StockIndicatorSnapshot)
    error:              Optional[str]          = None

    def is_valid(self) -> bool:
        return self.error is None

    def to_dict(self) -> dict:
        snap = self.indicators

        def _clean(v):
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
                "roc_10":            _clean(snap.roc_10),
                "roc_21":            _clean(snap.roc_21),
                "acceleration":      _clean(snap.acceleration),
                "rs_3m":             _clean(snap.rs_3m),
                "rs_trend":          _clean(snap.rs_trend),
            },
        }