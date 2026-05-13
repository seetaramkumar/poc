"""
stock_regime/filters/models.py
================================
Typed data contracts for the universe filter layer.

FilterResult is the primary output of UniverseFilter.apply().
It carries accepted data, rejection reasons, and a run summary —
enough information to persist, log, and audit every filtering decision.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Optional

import pandas as pd


# ─────────────────────────────────────────────────────────────────────────────
#  Filter identifiers
# ─────────────────────────────────────────────────────────────────────────────

# Canonical filter names — used in logs, parquet, and the diagnostics report.
FILTER_HISTORY   = "history"
FILTER_PRICE     = "price"
FILTER_LIQUIDITY = "liquidity"


# ─────────────────────────────────────────────────────────────────────────────
#  Rejection reason
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class FilterReason:
    """
    A single reason why a symbol was rejected by one filter.

    A symbol may accumulate multiple FilterReasons if it fails more than
    one check.  All checks are run even after the first failure so the
    diagnostics report can show the full picture.

    Attributes
    ----------
    filter_name :
        Which filter raised this reason (``"history"``, ``"price"``,
        ``"liquidity"``).
    check :
        The specific check within that filter (e.g. ``"min_bars"``,
        ``"min_price"``, ``"min_adv"``).
    reason :
        Human-readable description of what failed.
    metric :
        The actual value that failed (e.g. ADV in Crore INR).
    threshold :
        The threshold it needed to meet.
    """
    filter_name: str
    check:       str
    reason:      str
    metric:      float
    threshold:   float

    def to_dict(self) -> dict:
        return {
            "filter_name": self.filter_name,
            "check":       self.check,
            "reason":      self.reason,
            "metric":      round(self.metric, 4),
            "threshold":   round(self.threshold, 4),
        }


# ─────────────────────────────────────────────────────────────────────────────
#  Filter summary (aggregate stats for one filter run)
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class FilterSummary:
    """
    Aggregate statistics for one UniverseFilter.apply() call.

    Used in the diagnostics report and persisted to parquet.
    """
    universe:        str
    run_date:        date
    input_count:     int
    accepted_count:  int
    rejected_count:  int

    # Breakdown by which filter caused the rejection
    # (a symbol may be counted in multiple buckets if it failed multiple filters)
    rejected_by_history:   int = 0
    rejected_by_price:     int = 0
    rejected_by_liquidity: int = 0

    high_rejection_warning: bool = False   # True when rejection_pct > configured threshold

    @property
    def accepted_pct(self) -> float:
        return self.accepted_count / max(self.input_count, 1) * 100

    @property
    def rejected_pct(self) -> float:
        return self.rejected_count / max(self.input_count, 1) * 100

    def to_dict(self) -> dict:
        return {
            "universe":              self.universe,
            "run_date":              self.run_date,
            "input_count":           self.input_count,
            "accepted_count":        self.accepted_count,
            "rejected_count":        self.rejected_count,
            "accepted_pct":          round(self.accepted_pct, 2),
            "rejected_pct":          round(self.rejected_pct, 2),
            "rejected_by_history":   self.rejected_by_history,
            "rejected_by_price":     self.rejected_by_price,
            "rejected_by_liquidity": self.rejected_by_liquidity,
            "high_rejection_warning":self.high_rejection_warning,
        }


# ─────────────────────────────────────────────────────────────────────────────
#  Primary filter output
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class FilterResult:
    """
    Output of UniverseFilter.apply().

    Attributes
    ----------
    accepted :
        Symbols that passed all filters, ready for the engine.
    rejected :
        Symbols that failed at least one filter, with all failure reasons.
    summary :
        Aggregate statistics for this filter run.
    """
    accepted: dict[str, pd.DataFrame]            = field(default_factory=dict)
    rejected: dict[str, list[FilterReason]]      = field(default_factory=dict)
    summary:  Optional[FilterSummary]            = None

    def rejected_symbols_as_records(self) -> list[dict]:
        """Flatten rejected symbols and reasons into a list of dicts for parquet."""
        records = []
        for symbol, reasons in self.rejected.items():
            for r in reasons:
                row = {"symbol": symbol}
                row.update(r.to_dict())
                records.append(row)
        return records