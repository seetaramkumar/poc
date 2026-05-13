"""
stock_regime/filters/universe_filter.py
=========================================
Orchestrates the three-stage filter pipeline.

Pipeline order (cheapest first)
--------------------------------
1. HistoryFilter   — minimum bars, staleness, data gaps
2. PriceFilter     — minimum price, circuit-breaker detection
3. LiquidityFilter — ADV, zero-volume ratio, volume CV

ALL three filters are run for every symbol even after the first failure.
This allows the diagnostics report to show the full rejection picture
(e.g. a stock failed both history and liquidity — don't just log one).

The orchestrator is the ONLY piece that the pipeline calls directly.
Individual filter classes are never imported outside this package.
"""

from __future__ import annotations

import logging
from datetime import date
from pathlib import Path
from typing import Optional

import pandas as pd

from .history_filter   import HistoryFilter
from .liquidity_filter import LiquidityFilter
from .models           import FilterReason, FilterResult, FilterSummary
from .price_filter     import PriceFilter

logger = logging.getLogger(__name__)


class UniverseFilter:
    """
    Applies the full filter pipeline to a dict of OHLCV DataFrames.

    Parameters
    ----------
    history_filter :
        Configured HistoryFilter instance.
    price_filter :
        Configured PriceFilter instance.
    liquidity_filter :
        Configured LiquidityFilter instance.
    universe :
        Universe label used in logging and the FilterSummary (e.g. "NIFTY500").
    high_rejection_warning_pct :
        Log a warning when the rejection rate exceeds this percentage.
    """

    def __init__(
        self,
        history_filter:              HistoryFilter,
        price_filter:                PriceFilter,
        liquidity_filter:            LiquidityFilter,
        universe:                    str   = "UNKNOWN",
        high_rejection_warning_pct:  float = 30.0,
    ) -> None:
        self._history   = history_filter
        self._price     = price_filter
        self._liquidity = liquidity_filter
        self._universe  = universe
        self._warn_pct  = high_rejection_warning_pct

    # ──────────────────────────────────────────────────────────────────────────
    #  Primary API
    # ──────────────────────────────────────────────────────────────────────────

    def apply(
        self,
        stock_data: dict[str, pd.DataFrame],
        run_date:   Optional[date] = None,
    ) -> FilterResult:
        """
        Apply all filters to *stock_data*.

        Parameters
        ----------
        stock_data :
            ``{symbol: ohlcv_df}`` mapping from DataManager.
        run_date :
            Date label for the FilterSummary.  Defaults to today.

        Returns
        -------
        FilterResult
            Contains accepted DataFrames, rejection reasons, and summary stats.
        """
        run_date = run_date or date.today()
        accepted: dict[str, pd.DataFrame]       = {}
        rejected: dict[str, list[FilterReason]] = {}

        for symbol, df in stock_data.items():
            all_reasons: list[FilterReason] = []

            # Run all three filters regardless of prior failures
            all_reasons += self._history.check(symbol, df)
            all_reasons += self._price.check(symbol, df)
            all_reasons += self._liquidity.check(symbol, df)

            if all_reasons:
                rejected[symbol] = all_reasons
            else:
                accepted[symbol] = df

        summary = self._build_summary(accepted, rejected, run_date)
        self._log_summary(summary)

        return FilterResult(
            accepted = accepted,
            rejected = rejected,
            summary  = summary,
        )

    # ──────────────────────────────────────────────────────────────────────────
    #  Factory method — build from config.yaml section
    # ──────────────────────────────────────────────────────────────────────────

    @classmethod
    def from_config(
        cls,
        config: object,       # StockEngineConfig
        universe: str,
        exchange: str = "NSE",
    ) -> "UniverseFilter":
        """
        Build a UniverseFilter from the ``filters:`` section of config.yaml.

        Parameters
        ----------
        config :
            StockEngineConfig (loaded from config.yaml).
        universe :
            Universe label for logging.
        exchange :
            Exchange identifier ("NSE" | "NYSE" | "NASDAQ") used to select
            the correct price/liquidity thresholds.
        """
        f   = config.filters
        exc = exchange.upper()

        # History filter — same thresholds for all exchanges
        history = HistoryFilter(
            min_bars          = int(f.history.min_bars),
            max_gap_days      = int(f.history.max_gap_days),
            max_stale_days    = int(f.history.max_stale_days),
            gap_lookback_days = int(getattr(f.history, "gap_lookback_days", 90)),
        )

        # Price filter — exchange-specific thresholds
        price_cfg = getattr(f.price, exc, f.price.NSE)  # fallback to NSE
        price = PriceFilter(
            min_price             = float(getattr(price_cfg, "min_price_inr",
                                           getattr(price_cfg, "min_price_usd", 10.0))),
            max_price             = getattr(price_cfg, "max_price_inr",
                                    getattr(price_cfg, "max_price_usd", None)),
            exchange              = exc,
            max_circuit_days      = int(getattr(f.price, "max_circuit_days", 3)),
            circuit_lookback_days = int(getattr(f.price, "circuit_lookback_days", 90)),
            circuit_is_fatal      = bool(getattr(f.price, "circuit_is_fatal", False)),
        )

        # Liquidity filter — exchange-specific thresholds
        liq_cfg    = getattr(f.liquidity, exc, f.liquidity.NSE)
        in_crore   = exc == "NSE"
        min_adv    = float(
            getattr(liq_cfg, "min_adv_cr",     None) or
            getattr(liq_cfg, "min_adv_usd_mn", 5.0)
        )
        liquidity = LiquidityFilter(
            min_adv               = min_adv,
            adv_in_crore          = in_crore,
            adv_period            = int(liq_cfg.adv_period),
            max_zero_volume_ratio = float(liq_cfg.max_zero_volume_ratio),
            max_volume_cv         = float(getattr(liq_cfg, "max_volume_cv", 5.0)),
            zero_volume_is_fatal  = True,
            volume_cv_is_fatal    = False,
        )

        return cls(
            history_filter             = history,
            price_filter               = price,
            liquidity_filter           = liquidity,
            universe                   = universe,
            high_rejection_warning_pct = float(
                getattr(f, "high_rejection_warning_pct", 30.0)
            ),
        )

    # ──────────────────────────────────────────────────────────────────────────
    #  Private helpers
    # ──────────────────────────────────────────────────────────────────────────

    def _build_summary(
        self,
        accepted: dict,
        rejected: dict[str, list[FilterReason]],
        run_date: date,
    ) -> FilterSummary:
        """Count per-filter rejection tallies for the summary."""
        from .models import FILTER_HISTORY, FILTER_PRICE, FILTER_LIQUIDITY

        n_hist = n_price = n_liq = 0
        for reasons in rejected.values():
            names = {r.filter_name for r in reasons}
            if FILTER_HISTORY   in names: n_hist  += 1
            if FILTER_PRICE     in names: n_price += 1
            if FILTER_LIQUIDITY in names: n_liq   += 1

        total      = len(accepted) + len(rejected)
        rej_pct    = len(rejected) / max(total, 1) * 100
        warn_flag  = rej_pct > self._warn_pct

        return FilterSummary(
            universe                = self._universe,
            run_date                = run_date,
            input_count             = total,
            accepted_count          = len(accepted),
            rejected_count          = len(rejected),
            rejected_by_history     = n_hist,
            rejected_by_price       = n_price,
            rejected_by_liquidity   = n_liq,
            high_rejection_warning  = warn_flag,
        )

    def _log_summary(self, s: FilterSummary) -> None:
        pct_str = f"{s.rejected_pct:.1f}%"
        logger.info(
            "UniverseFilter [%s]: %d → %d accepted (%s rejected) | "
            "history=%d  price=%d  liquidity=%d",
            s.universe,
            s.input_count,
            s.accepted_count,
            pct_str,
            s.rejected_by_history,
            s.rejected_by_price,
            s.rejected_by_liquidity,
        )
        if s.high_rejection_warning:
            logger.warning(
                "UniverseFilter [%s]: rejection rate %.1f%% exceeds warning "
                "threshold %.1f%% — check data feed or loosen thresholds.",
                s.universe, s.rejected_pct, self._warn_pct,
            )