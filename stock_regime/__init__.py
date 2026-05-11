"""
stock_regime/src/__init__.py
=============================
Public surface of the Stock Regime Engine.

Callers need only import from here:

    from stock_regime.src import StockRegimeEngine
    from stock_regime.src.models import MarketRegimeInput, StockRegime

Everything else (indicators, signals, scorer, classifier, ranker,
persistence, config_loader) is an implementation detail.
"""

from .engine import StockRegimeEngine
from .models import (
    DimensionalScores,
    MarketRegimeInput,
    StockIndicatorSnapshot,
    StockRegime,
    StockRegimeResult,
    StockSignals,
)

__all__ = [
    # Engine
    "StockRegimeEngine",
    # Models
    "MarketRegimeInput",
    "StockRegime",
    "StockRegimeResult",
    "StockSignals",
    "StockIndicatorSnapshot",
    "DimensionalScores",
]
