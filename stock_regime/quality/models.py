"""
stock_regime/quality/models.py
================================
Typed data contracts for the data quality validation layer.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from enum import Enum
from typing import Optional


class AnomalySeverity(str, Enum):
    WARNING = "WARNING"   # log and continue; data is usable
    ERROR   = "ERROR"     # log; attempt to correct then continue
    FATAL   = "FATAL"     # exclude symbol from this run entirely


@dataclass
class Anomaly:
    """A single data quality issue found in one symbol's OHLCV series."""
    symbol:    str
    check:     str               # e.g. "inverted_ohlc" | "price_spike" | "zero_close"
    severity:  AnomalySeverity
    bar_date:  Optional[date]    # which bar triggered it (None = whole-series check)
    detail:    str               # human-readable description
    value:     float             # the offending value
    expected:  str               # what was expected (e.g. "high >= low")

    def to_dict(self) -> dict:
        return {
            "symbol":   self.symbol,
            "check":    self.check,
            "severity": self.severity.value,
            "bar_date": str(self.bar_date) if self.bar_date else None,
            "detail":   self.detail,
            "value":    round(self.value, 6),
            "expected": self.expected,
        }


@dataclass
class QualityReport:
    """
    Output of DataQualityValidator.validate().

    Attributes
    ----------
    clean :
        Symbols that passed all checks (or had only WARNING/ERROR anomalies
        that were corrected), ready for indicator computation.
    excluded :
        Symbols with FATAL anomalies — removed from this run.
    anomalies :
        All anomalies found (WARNING + ERROR + FATAL), for logging/persistence.
    corrections :
        Description of any in-place corrections made to the clean DataFrames.
    """
    clean:       dict[str, object]           = field(default_factory=dict)  # pd.DataFrame
    excluded:    dict[str, list[Anomaly]]    = field(default_factory=dict)
    anomalies:   dict[str, list[Anomaly]]   = field(default_factory=dict)
    corrections: dict[str, list[str]]        = field(default_factory=dict)

    @property
    def clean_count(self)    -> int: return len(self.clean)
    @property
    def excluded_count(self) -> int: return len(self.excluded)
    @property
    def warning_count(self)  -> int:
        return sum(
            1 for anoms in self.anomalies.values()
            for a in anoms if a.severity == AnomalySeverity.WARNING
        )

    def all_anomalies_as_records(self) -> list[dict]:
        records = []
        for sym, anoms in self.anomalies.items():
            for a in anoms:
                records.append(a.to_dict())
        return records