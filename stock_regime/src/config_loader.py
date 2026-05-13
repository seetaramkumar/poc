"""
stock_regime/src/config_loader.py
==================================
Reads config/config.yaml once at startup and exposes a typed,
dot-accessible view of every threshold, weight, and parameter.

All other modules import from here — never from the YAML directly.
Supports custom config paths via the ``STOCK_REGIME_CONFIG`` env-var.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml


# Default config path; overridable via environment variable.
_DEFAULT_CONFIG_PATH = (
    Path(__file__).parent.parent / "config" / "config.yaml"
)


def _load_yaml(path: Path) -> dict[str, Any]:
    """Read and parse a YAML file; raise a clear error if missing."""
    if not path.exists():
        raise FileNotFoundError(
            f"Config file not found: {path}\n"
            "Set STOCK_REGIME_CONFIG to point to your config file."
        )
    with open(path) as fh:
        return yaml.safe_load(fh)


class _NS:
    """
    Converts a nested dict into dot-accessible attributes recursively.

    Example
    -------
    ns = _NS({"ema": {"fast": 20}})
    ns.ema.fast  # → 20
    """

    def __init__(self, data: dict[str, Any]) -> None:
        for key, value in data.items():
            setattr(self, key, _NS(value) if isinstance(value, dict) else value)

    def __repr__(self) -> str:  # pragma: no cover
        return f"<Config {self.__dict__}>"


class MarketContextConfig:
    """
    Wraps the market_context section of the config.

    Kept as a dedicated class (rather than _NS) because
    ``aligned_regimes`` is a dict[str, list[str]] that needs
    to stay as a plain dict for O(1) lookup at runtime.
    """

    def __init__(self, raw: dict[str, Any]) -> None:
        self.alignment_boost:      float = float(raw["alignment_boost"])
        self.misalignment_penalty: float = float(raw["misalignment_penalty"])
        # { "BULLISH_TREND": ["TREND_UP", "MOMENTUM", ...], ... }
        self.aligned_regimes: dict[str, list[str]] = raw.get("aligned_regimes", {})

    def is_aligned(self, market_regime: str, stock_regime: str) -> bool:
        """Return ``True`` when *stock_regime* aligns with *market_regime*."""
        return stock_regime in self.aligned_regimes.get(market_regime, [])


class StockEngineConfig:
    """
    Top-level configuration object for the Stock Regime Engine.

    Mirrors the structure of ``config/config.yaml``:

    Attributes
    ----------
    indicators  — EMA/ADX/ATR/Volume/RS periods
    thresholds  — All numeric cut-offs
    scoring     — Per-regime signal weights + min_confidence
    dimensional — Weights for trend/momentum dimensional scores
    market_ctx  — Context alignment boosts / penalties
    ranking     — top_n parameter
    """

    def __init__(self, config_path: Path | None = None) -> None:
        env_path = os.environ.get("STOCK_REGIME_CONFIG")
        path = Path(env_path) if env_path else (config_path or _DEFAULT_CONFIG_PATH)
        raw = _load_yaml(path)

        self.indicators  = _NS(raw["indicators"])
        self.thresholds  = _NS(raw["thresholds"])
        self.scoring     = _NS(raw["scoring"])
        self.dimensional = _NS(raw["dimensional_scores"])
        self.market_ctx  = MarketContextConfig(raw["market_context"])
        self.ranking     = _NS(raw["ranking"])
        self.filters     = _NS(raw["filters"])     if "filters"   in raw else _NS({})
        self.quality     = _NS(raw["quality"])     if "quality"   in raw else _NS({})
        self.stability   = _NS(raw["stability"])   if "stability" in raw else _NS({})

    # ── Convenience accessors (avoids deep-dot chains in hot paths) ──

    @property
    def ema_fast(self) -> int:
        return self.indicators.ema_periods.fast

    @property
    def ema_mid(self) -> int:
        return self.indicators.ema_periods.mid

    @property
    def ema_slow(self) -> int:
        return self.indicators.ema_periods.slow

    @property
    def adx_period(self) -> int:
        return self.indicators.adx_period

    @property
    def atr_period(self) -> int:
        return self.indicators.atr_period

    @property
    def volume_ma_period(self) -> int:
        return self.indicators.volume_ma_period

    @property
    def rs_period(self) -> int:
        return self.indicators.rs_period

    @property
    def high_period(self) -> int:
        return self.indicators.high_period

    @property
    def min_confidence(self) -> float:
        return self.scoring.min_confidence


    @property
    def has_filters(self) -> bool:
        return hasattr(self, "filters")

    @property
    def has_quality(self) -> bool:
        return hasattr(self, "quality")

    @property
    def has_stability(self) -> bool:
        return hasattr(self, "stability")