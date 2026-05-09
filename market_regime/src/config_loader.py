"""
config_loader.py — Load and expose engine configuration
=========================================================
Reads config/config.yaml once at startup and provides a
typed, attribute-friendly view of every threshold and parameter.
All other modules import from here — never from the YAML directly.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict

import yaml


# Default path; can be overridden via the REGIME_CONFIG env-var.
_DEFAULT_CONFIG_PATH = Path(__file__).parent.parent / "config" / "config.yaml"


def _load_yaml(path: Path) -> Dict[str, Any]:
    """Read and parse a YAML file; raise a clear error if missing."""
    if not path.exists():
        raise FileNotFoundError(
            f"Config file not found: {path}\n"
            "Set the REGIME_CONFIG environment variable to point to your config."
        )
    with open(path, "r") as fh:
        return yaml.safe_load(fh)


class _NestedNamespace:
    """
    Converts a nested dict into dot-accessible attributes.

    Example
    -------
    ns = _NestedNamespace({"a": {"b": 1}})
    ns.a.b  # → 1
    """

    def __init__(self, data: Dict[str, Any]) -> None:
        for key, value in data.items():
            if isinstance(value, dict):
                setattr(self, key, _NestedNamespace(value))
            else:
                setattr(self, key, value)

    def __repr__(self) -> str:  # pragma: no cover
        return f"<Config {self.__dict__}>"


class EngineConfig:
    """
    Top-level configuration object.

    Attributes (mirrors config.yaml structure)
    -------------------------------------------
    indicators  – EMA periods, ADX/ATR periods, volume MA period
    thresholds  – All numeric cut-offs
    scoring     – Per-regime signal weights and min_confidence
    """

    def __init__(self, config_path: Path | None = None) -> None:
        env_path = os.environ.get("REGIME_CONFIG")
        path = Path(env_path) if env_path else (config_path or _DEFAULT_CONFIG_PATH)
        raw = _load_yaml(path)

        self.indicators  = _NestedNamespace(raw["indicators"])
        self.thresholds  = _NestedNamespace(raw["thresholds"])
        self.scoring     = _NestedNamespace(raw["scoring"])

    # ── convenience accessors (avoids deep-dot chains in hot paths) ──

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
