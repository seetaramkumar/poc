"""
stock_regime/src/config_loader.py
==================================
Reads config/config.yaml and exposes a typed dot-accessible object.

Updated to load: filters, quality, stability, quality_engine,
score_diagnostics sections added in the improvement roadmap.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml

_DEFAULT_CONFIG_PATH = (
    Path(__file__).parent.parent / "config" / "config.yaml"
)


def _load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(
            f"Config file not found: {path}\n"
            "Set STOCK_REGIME_CONFIG to override."
        )
    with open(path) as fh:
        return yaml.safe_load(fh)


class _NS:
    """Nested dict → dot-accessible attributes."""
    def __init__(self, data: dict[str, Any]) -> None:
        for k, v in data.items():
            setattr(self, k, _NS(v) if isinstance(v, dict) else v)

    def __repr__(self) -> str:
        return f"<Config {self.__dict__}>"


class MarketContextConfig:
    def __init__(self, raw: dict[str, Any]) -> None:
        self.alignment_boost:      float = float(raw["alignment_boost"])
        self.misalignment_penalty: float = float(raw["misalignment_penalty"])
        self.aligned_regimes: dict[str, list[str]] = raw.get("aligned_regimes", {})

    def is_aligned(self, market_regime: str, stock_regime: str) -> bool:
        return stock_regime in self.aligned_regimes.get(market_regime, [])


class StockEngineConfig:
    """
    Top-level configuration object for the Stock Regime Engine.

    Sections loaded
    ---------------
    indicators, thresholds, scoring, dimensional_scores,
    market_context, ranking,
    filters, quality, stability,          (Phase 1-3)
    quality_engine, score_diagnostics      (Phase 7, 10)
    """

    def __init__(self, config_path: Path | None = None) -> None:
        env_path = os.environ.get("STOCK_REGIME_CONFIG")
        path     = Path(env_path) if env_path else (config_path or _DEFAULT_CONFIG_PATH)
        raw      = _load_yaml(path)

        self.indicators  = _NS(raw["indicators"])
        self.thresholds  = _NS(raw["thresholds"])
        self.scoring     = _NS(raw["scoring"])
        self.dimensional = _NS(raw["dimensional_scores"])
        self.market_ctx  = MarketContextConfig(raw["market_context"])
        self.ranking     = _NS(raw["ranking"])

        # Optional sections — graceful defaults if missing
        self.filters          = _NS(raw["filters"])          if "filters"          in raw else _NS({})
        self.quality          = _NS(raw["quality"])          if "quality"          in raw else _NS({})
        self.stability        = _NS(raw["stability"])        if "stability"        in raw else _NS({})
        self.quality_engine   = _NS(raw["quality_engine"])   if "quality_engine"   in raw else _NS({})
        self.score_diagnostics= _NS(raw["score_diagnostics"])if "score_diagnostics"in raw else _NS({})

    # ── Convenience accessors ───────────────────────────────────────

    @property
    def ema_fast(self)          -> int:   return self.indicators.ema_periods.fast
    @property
    def ema_mid(self)           -> int:   return self.indicators.ema_periods.mid
    @property
    def ema_slow(self)          -> int:   return self.indicators.ema_periods.slow
    @property
    def adx_period(self)        -> int:   return self.indicators.adx_period
    @property
    def atr_period(self)        -> int:   return self.indicators.atr_period
    @property
    def volume_ma_period(self)  -> int:   return self.indicators.volume_ma_period
    @property
    def rs_period(self)         -> int:   return self.indicators.rs_period
    @property
    def high_period(self)       -> int:   return self.indicators.high_period
    @property
    def min_confidence(self)    -> float: return self.scoring.min_confidence