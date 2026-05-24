"""
stock_regime/src/config_loader.py
==================================
Reads config/config.yaml and exposes a typed dot-accessible object.

Key defensive change
---------------------
_NS.__getattr__ now returns None for missing keys instead of raising
AttributeError.  This prevents the engine from crashing when a config key
is absent (e.g. after a partial yaml.dump() that dropped some keys).

All required numeric thresholds also have explicit hardcoded defaults in
StockEngineConfig.thresholds_with_defaults so the engine can always fall
back to safe values even with a minimal config file.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Optional

import yaml

_DEFAULT_CONFIG_PATH = (
    Path(__file__).parent.parent / "config" / "config.yaml"
)


def _load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(
            f"Config file not found: {path}\n"
            "Set STOCK_REGIME_CONFIG env var to override."
        )
    with open(path) as fh:
        return yaml.safe_load(fh)


class _NS:
    """
    Nested dict → dot-accessible attributes.

    Returns None for missing attributes instead of raising AttributeError.
    This makes the config resilient to partial yaml files.
    """
    def __init__(self, data: dict[str, Any]) -> None:
        for k, v in data.items():
            setattr(self, k, _NS(v) if isinstance(v, dict) else v)

    def __getattr__(self, name: str):
        # Called only when normal attribute lookup fails.
        # Return None so callers can use `getattr(thr, 'key', default)` safely.
        return None

    def __repr__(self) -> str:
        return f"<Config {self.__dict__}>"

    def get(self, key: str, default=None):
        """dict-like get() with default."""
        v = getattr(self, key)
        return default if v is None else v


# ─────────────────────────────────────────────────────────────────────────────
#  Hardcoded defaults for every threshold the engine reads.
#  These are used when the YAML is missing a key.
# ─────────────────────────────────────────────────────────────────────────────
_THRESHOLD_DEFAULTS: dict[str, Any] = {
    # ADX
    "adx_strong_trend": 25.0,
    "adx_weak_trend":   18.0,

    # ATR
    "atr_ma_period":        20,
    "atr_volatile_ratio":   1.30,
    "atr_quiet_ratio":      0.70,
    "atr_compressed_ratio": 0.75,
    "atr_expanding_ratio":  1.00,

    # EMA slope
    "ema_slope_window":   5,
    "ema_flat_threshold": 0.001,

    # Volume
    "volume_surge_ratio": 1.50,

    # Relative Strength
    "rs_positive_threshold":         1.02,
    "rs_negative_threshold":         0.98,
    "rs_strong_threshold":           1.05,
    "rs_trend_positive_threshold":   0.0005,

    # Breakout
    "near_high_pct": 0.03,

    # ROC / momentum
    "acceleration_threshold": 0.5,

    # Trend quality
    "higher_highs_window":    20,
    "higher_highs_min_count":  5,
    "ema_extended_pct":       0.10,

    # Phase 3 — volatility instability
    "volatility_instability_composite_threshold": 0.40,
    "volatility_instability_severe_threshold":    0.55,
    "volatility_instability_window":              20,
    "candle_instability_threshold":              0.50,
    "reversal_frequency_threshold":              0.55,
    "gap_frequency_threshold":                   0.15,
    "gap_threshold_pct":                         0.01,
    "wickiness_threshold":                       0.45,

    # Phase 2 — range detection
    "bb_period":                20,
    "bb_std":                    2.0,
    "bb_narrow_threshold":      0.04,
    "der_window":               14,
    "der_range_threshold":      0.35,
    "ema_compressed_threshold": 0.005,
}


class _ThresholdsProxy:
    """
    Wraps a _NS config object and transparently falls back to
    _THRESHOLD_DEFAULTS for any missing key.

    Usage in indicators.py / signals.py is unchanged — just access
    thr.some_key as before. If the YAML is missing the key, the
    hardcoded default is returned silently.
    """

    def __init__(self, ns: _NS) -> None:
        self._ns = ns

    def __getattr__(self, name: str):
        # 1. Try the loaded YAML value
        val = getattr(self._ns, name)
        if val is not None:
            return val
        # 2. Fall back to hardcoded default
        if name in _THRESHOLD_DEFAULTS:
            return _THRESHOLD_DEFAULTS[name]
        # 3. Unknown key — return None (better than AttributeError crash)
        return None

    def get(self, key: str, default=None):
        v = getattr(self, key)
        return default if v is None else v


class MarketContextConfig:
    def __init__(self, raw: dict[str, Any]) -> None:
        self.alignment_boost:      float = float(raw.get("alignment_boost", 1.10))
        self.misalignment_penalty: float = float(raw.get("misalignment_penalty", 0.90))
        self.aligned_regimes: dict[str, list[str]] = raw.get("aligned_regimes", {})

    def is_aligned(self, market_regime: str, stock_regime: str) -> bool:
        return stock_regime in self.aligned_regimes.get(market_regime, [])


class StockEngineConfig:
    """
    Top-level configuration object for the Stock Regime Engine.

    All threshold access goes through _ThresholdsProxy which silently
    falls back to _THRESHOLD_DEFAULTS when a YAML key is absent.
    """

    def __init__(self, config_path: Optional[Path] = None) -> None:
        env_path = os.environ.get("STOCK_REGIME_CONFIG")
        path     = Path(env_path) if env_path else (config_path or _DEFAULT_CONFIG_PATH)
        raw      = _load_yaml(path)

        self.indicators  = _NS(raw.get("indicators", {}))
        # Wrap thresholds in the proxy for safe fallback access
        self.thresholds  = _ThresholdsProxy(_NS(raw.get("thresholds", {})))
        self.scoring     = _NS(raw.get("scoring", {}))
        self.dimensional = _NS(raw.get("dimensional_scores", {}))
        self.market_ctx  = MarketContextConfig(raw.get("market_context", {}))
        self.ranking     = _NS(raw.get("ranking", {}))

        # Optional sections — graceful defaults if missing
        self.filters           = _NS(raw["filters"])           if "filters"           in raw else _NS({})
        self.quality           = _NS(raw["quality"])           if "quality"           in raw else _NS({})
        self.stability         = _NS(raw["stability"])         if "stability"         in raw else _NS({})
        self.quality_engine    = _NS(raw["quality_engine"])    if "quality_engine"    in raw else _NS({})
        self.score_diagnostics = _NS(raw["score_diagnostics"]) if "score_diagnostics" in raw else _NS({})
        self.breadth_engine    = _NS(raw["breadth_engine"])    if "breadth_engine"    in raw else _NS({})
        self.sector_engine     = _NS(raw["sector_engine"])     if "sector_engine"     in raw else _NS({})
        self.strategy_router   = _NS(raw["strategy_router"])   if "strategy_router"   in raw else _NS({})

    # ── Convenience accessors ──────────────────────────────────────────

    @property
    def ema_fast(self) -> int:
        v = getattr(self.indicators, "ema_periods", None)
        return int(v.fast) if v and v.fast is not None else 20

    @property
    def ema_mid(self) -> int:
        v = getattr(self.indicators, "ema_periods", None)
        return int(v.mid) if v and v.mid is not None else 50

    @property
    def ema_slow(self) -> int:
        v = getattr(self.indicators, "ema_periods", None)
        return int(v.slow) if v and v.slow is not None else 200

    @property
    def adx_period(self) -> int:
        v = getattr(self.indicators, "adx_period", None)
        return int(v) if v is not None else 14

    @property
    def atr_period(self) -> int:
        v = getattr(self.indicators, "atr_period", None)
        return int(v) if v is not None else 14

    @property
    def volume_ma_period(self) -> int:
        v = getattr(self.indicators, "volume_ma_period", None)
        return int(v) if v is not None else 20

    @property
    def rs_period(self) -> int:
        v = getattr(self.indicators, "rs_period", None)
        return int(v) if v is not None else 63

    @property
    def high_period(self) -> int:
        v = getattr(self.indicators, "high_period", None)
        return int(v) if v is not None else 252

    @property
    def min_confidence(self) -> float:
        v = getattr(self.scoring, "min_confidence", None)
        return float(v) if v is not None else 0.50