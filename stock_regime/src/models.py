"""
stock_regime/src/models.py — Extended with Phase 1+2 indicator fields.

Phase 1 additions to StockIndicatorSnapshot:
  candle_instability, reversal_frequency, gap_frequency, wickiness_score

Phase 2 additions to StockIndicatorSnapshot:
  bb_width, directional_efficiency, ema_spread

Phase 1 additions to StockSignals:
  volatile_instability  — composite erratic-market signal
  candle_erratic        — high candle instability
  high_reversal_freq    — direction reversals too frequent
  range_bound           — directional efficiency low (Phase 2)
  bb_compressed         — Bollinger bands tight (Phase 2)
  ema_compressed        — EMA20/50 converged (Phase 2)
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

    def __repr__(self):
        return f"MarketRegimeInput(regime={self.regime!r}, confidence={self.confidence:.2f})"


@dataclass
class StockIndicatorSnapshot:
    # Core price
    close: float = float("nan")

    # EMAs
    ema20:  Optional[float] = None
    ema50:  Optional[float] = None
    ema200: Optional[float] = None
    ema20_slope: Optional[float] = None
    ema50_slope: Optional[float] = None

    # Strength / volatility
    adx:    Optional[float] = None
    atr:    Optional[float] = None
    atr_ma: Optional[float] = None

    # Volume
    volume:    Optional[float] = None
    volume_ma: Optional[float] = None

    # RS (legacy alias kept)
    relative_strength: Optional[float] = None
    high_52w:          Optional[float] = None

    # ROC / momentum
    roc_10:       Optional[float] = None
    roc_21:       Optional[float] = None
    acceleration: Optional[float] = None

    # Multi-period RS
    rs_1m:    Optional[float] = None
    rs_3m:    Optional[float] = None
    rs_6m:    Optional[float] = None
    rs_trend: Optional[float] = None

    # Trend quality
    higher_highs_count: Optional[int]   = None
    ema_distance_pct:   Optional[float] = None

    # Phase 1 — volatility instability
    candle_instability: Optional[float] = None   # erratic move ratio
    reversal_frequency: Optional[float] = None   # direction-flip fraction
    gap_frequency:      Optional[float] = None   # open-gap fraction
    wickiness_score:    Optional[float] = None   # wick-to-range ratio

    # Phase 2 — range detection
    bb_width:               Optional[float] = None   # Bollinger bandwidth
    directional_efficiency: Optional[float] = None   # DER [0=ranging, 1=trending]
    ema_spread:             Optional[float] = None   # (ema20-ema50)/close

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
            "close": self.close, "ema20": self.ema20, "ema50": self.ema50,
            "ema200": self.ema200, "ema20_slope": self.ema20_slope,
            "ema50_slope": self.ema50_slope, "adx": self.adx,
            "atr": self.atr, "atr_ma": self.atr_ma,
            "volume": self.volume, "volume_ma": self.volume_ma,
            "relative_strength": self.relative_strength, "high_52w": self.high_52w,
            "roc_10": self.roc_10, "roc_21": self.roc_21,
            "acceleration": self.acceleration,
            "rs_1m": self.rs_1m, "rs_3m": self.rs_3m, "rs_6m": self.rs_6m,
            "rs_trend": self.rs_trend,
            "higher_highs_count": self.higher_highs_count,
            "ema_distance_pct": self.ema_distance_pct,
            # Phase 1
            "candle_instability": self.candle_instability,
            "reversal_frequency": self.reversal_frequency,
            "gap_frequency":      self.gap_frequency,
            "wickiness_score":    self.wickiness_score,
            # Phase 2
            "bb_width":               self.bb_width,
            "directional_efficiency": self.directional_efficiency,
            "ema_spread":             self.ema_spread,
        }


@dataclass
class StockSignals:
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

    # RS
    rs_positive: bool = False
    rs_negative: bool = False
    rs_strong:   bool = False

    # Breakout
    price_near_52w_high: bool = False

    # Momentum
    roc_positive:     bool = False
    roc_accelerating: bool = False
    rs_improving:     bool = False
    rs_weakening:     bool = False
    higher_highs:     bool = False
    ema_extended:     bool = False

    # Phase 1 — volatility instability (NEW)
    candle_erratic:       bool = False   # candle_instability > threshold
    high_reversal_freq:   bool = False   # reversal_frequency > threshold
    volatile_instability: bool = False   # composite: erratic + reversals + gaps

    # Phase 2 — range detection (NEW)
    range_bound:    bool = False   # directional_efficiency < threshold
    bb_compressed:  bool = False   # bb_width < bb_narrow_threshold
    ema_compressed: bool = False   # |ema_spread| < ema_compressed_threshold

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
            # Phase 1
            "candle_erratic":       self.candle_erratic,
            "high_reversal_freq":   self.high_reversal_freq,
            "volatile_instability": self.volatile_instability,
            # Phase 2
            "range_bound":    self.range_bound,
            "bb_compressed":  self.bb_compressed,
            "ema_compressed": self.ema_compressed,
        }


@dataclass
class ContinuousScores:
    adx_score:           float = 0.0
    ema_alignment_score: float = 0.0
    ema_distance_score:  float = 0.5
    atr_expansion_score: float = 0.5
    rs_score:            float = 0.5
    rs_trend_score:      float = 0.5
    roc_score:           float = 0.5
    volume_score:        float = 0.0
    # Phase 1
    instability_score:   float = 0.0   # higher = more erratic/unstable
    # Phase 2
    ranging_score:       float = 0.0   # higher = more range-bound

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
            "instability_score":   round(self.instability_score,    4),
            "ranging_score":       round(self.ranging_score,        4),
        }


@dataclass
class DimensionalScores:
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

        def _c(v):
            if v is None: return None
            try: return None if math.isnan(v) or math.isinf(v) else round(v, 4)
            except: return None

        return {
            "symbol":        self.symbol,
            "market":        self.market,
            "stock_regime":  self.stock_regime.value,
            "confidence":    round(self.confidence, 4),
            "scores":        self.dimensional_scores.to_dict(),
            "regime_scores": {k: round(v, 4) for k, v in self.regime_scores.items()},
            "signals":       self.signals.to_dict(),
            "indicators": {
                "close": _c(snap.close), "ema20": _c(snap.ema20),
                "ema50": _c(snap.ema50), "ema200": _c(snap.ema200),
                "adx": _c(snap.adx), "atr": _c(snap.atr),
                "relative_strength": _c(snap.relative_strength),
                "roc_10": _c(snap.roc_10), "rs_3m": _c(snap.rs_3m),
                "candle_instability": _c(snap.candle_instability),
                "reversal_frequency": _c(snap.reversal_frequency),
                "bb_width": _c(snap.bb_width),
                "directional_efficiency": _c(snap.directional_efficiency),
            },
        }