"""
trading_data/examples/basic_usage.py
--------------------------------------
Runnable examples covering every public method of the DataManager.

Run with:
    python -m trading_data.examples.basic_usage

Or import individual functions for use in notebooks:
    from trading_data.examples.basic_usage import example_indices
"""

from __future__ import annotations

import sys
from pathlib import Path

# Allow running directly from any working directory
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pandas as pd

from trading_data import (
    DataManager,
    DataManagerConfig,
    FetchResult,
    Provider,
)
from trading_data.exceptions import DataLayerError


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _print_df(df: pd.DataFrame, title: str = "") -> None:
    if title:
        print(f"\n{'─' * 60}")
        print(f"  {title}")
        print(f"{'─' * 60}")
    if df.empty:
        print("  (empty DataFrame)")
    else:
        print(f"  Shape : {df.shape}")
        print(f"  Cols  : {df.columns.tolist()}")
        print(f"  Range : {df.index[0].date()} → {df.index[-1].date()}")
        print(f"  dtypes: {df.dtypes.to_dict()}")
        print()
        print(df.tail(3).to_string())


# ─────────────────────────────────────────────────────────────────────────────
# Example 1 — Market Regime Engine compatible fetch (primary use-case)
# ─────────────────────────────────────────────────────────────────────────────

def example_market_regime_engine() -> pd.DataFrame:
    """
    Fetch Nifty 50 OHLCV exactly as the Market Regime Engine expects.

    The returned DataFrame has:
    - lowercase columns: open, high, low, close, volume
    - tz-naive DatetimeIndex named "date"
    - float64 for price columns, int64 for volume
    """
    print("\n" + "=" * 60)
    print("  EXAMPLE 1 — Market Regime Engine Compatible Fetch")
    print("=" * 60)

    manager = DataManager()          # uses Yahoo Finance + local parquet cache

    df = manager.get_daily_data(
        symbol="^NSEI",              # Nifty 50 (Yahoo ticker)
        start="2020-01-01",
        end="2025-01-01",
    )

    _print_df(df, "Nifty 50 daily OHLCV (2020–2025)")

    # Verify schema for Market Regime Engine
    required_cols = {"open", "high", "low", "close", "volume"}
    assert required_cols.issubset(set(df.columns)), "Missing required columns!"
    assert isinstance(df.index, pd.DatetimeIndex),  "Index must be DatetimeIndex!"
    assert df.index.tz is None,                      "Index must be tz-naive!"
    print("\n  ✓ Schema validated — ready for Market Regime Engine")

    return df


# ─────────────────────────────────────────────────────────────────────────────
# Example 2 — Canonical symbol resolution
# ─────────────────────────────────────────────────────────────────────────────

def example_canonical_symbols() -> None:
    """
    Demonstrate that canonical names ("NIFTY50", "SP500") are resolved
    automatically to provider-specific tickers.
    """
    print("\n" + "=" * 60)
    print("  EXAMPLE 2 — Canonical Symbol Resolution")
    print("=" * 60)

    manager = DataManager()

    pairs = [
        ("NIFTY50",  "Nifty 50 via canonical name"),
        ("^NSEI",    "Nifty 50 via Yahoo ticker"),
        ("SP500",    "S&P 500 via canonical name"),
        ("^GSPC",    "S&P 500 via Yahoo ticker"),
    ]

    for sym, label in pairs:
        try:
            df = manager.get_daily_data(sym, start="2024-01-01", end="2024-06-30")
            print(f"\n  {label}  ({sym!r})")
            print(f"    rows={len(df)}  last_close={df['close'].iloc[-1]:.2f}")
        except DataLayerError as exc:
            print(f"\n  {label}  ({sym!r})  → ERROR: {exc}")


# ─────────────────────────────────────────────────────────────────────────────
# Example 3 — Indices
# ─────────────────────────────────────────────────────────────────────────────

def example_indices() -> dict[str, pd.DataFrame]:
    """Fetch multiple global indices."""
    print("\n" + "=" * 60)
    print("  EXAMPLE 3 — Global Indices")
    print("=" * 60)

    manager  = DataManager()
    indices  = {
        "Nifty 50":    "NIFTY50",
        "S&P 500":     "SP500",
        "NASDAQ":      "NASDAQ",
        "Nifty Bank":  "BANKNIFTY",
    }
    results: dict[str, pd.DataFrame] = {}

    for label, sym in indices.items():
        try:
            df = manager.get_daily_data(sym, start="2023-01-01", end="2024-12-31")
            results[label] = df
            print(f"  {label:15s}  rows={len(df):4d}  last={df['close'].iloc[-1]:>12.2f}")
        except DataLayerError as exc:
            print(f"  {label:15s}  ERROR: {exc}")

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Example 4 — Individual stocks
# ─────────────────────────────────────────────────────────────────────────────

def example_stocks() -> None:
    """Fetch NSE and US equity OHLCV data."""
    print("\n" + "=" * 60)
    print("  EXAMPLE 4 — Equity Stocks")
    print("=" * 60)

    manager = DataManager()
    stocks  = [
        ("RELIANCE",    "Reliance Industries (NSE)"),
        ("TCS",         "TCS (NSE)"),
        ("AAPL",        "Apple (NASDAQ)"),
        ("MSFT",        "Microsoft (NASDAQ)"),
    ]

    for sym, label in stocks:
        try:
            df = manager.get_daily_data(sym, start="2022-01-01", end="2024-12-31")
            print(f"  {label:30s}  rows={len(df):4d}  last={df['close'].iloc[-1]:>10.2f}")
        except DataLayerError as exc:
            print(f"  {label:30s}  ERROR: {exc}")


# ─────────────────────────────────────────────────────────────────────────────
# Example 5 — fetch_latest()
# ─────────────────────────────────────────────────────────────────────────────

def example_fetch_latest() -> None:
    """Retrieve the most recent available trading bar."""
    print("\n" + "=" * 60)
    print("  EXAMPLE 5 — Latest Bar")
    print("=" * 60)

    manager = DataManager()

    for sym in ["NIFTY50", "AAPL"]:
        try:
            df = manager.fetch_latest(sym)
            if df.empty:
                print(f"  {sym}: no data returned")
            else:
                row = df.iloc[0]
                print(
                    f"  {sym:10s}  date={df.index[0].date()}  "
                    f"O={row['open']:.2f}  H={row['high']:.2f}  "
                    f"L={row['low']:.2f}  C={row['close']:.2f}"
                )
        except DataLayerError as exc:
            print(f"  {sym}: ERROR: {exc}")


# ─────────────────────────────────────────────────────────────────────────────
# Example 6 — fetch_multiple_symbols()
# ─────────────────────────────────────────────────────────────────────────────

def example_batch_fetch() -> dict[str, FetchResult]:
    """Fetch multiple symbols in a single call with error isolation."""
    print("\n" + "=" * 60)
    print("  EXAMPLE 6 — Batch Fetch (fetch_multiple_symbols)")
    print("=" * 60)

    manager = DataManager()
    symbols = ["NIFTY50", "AAPL", "TCS", "MSFT", "GOOGL"]

    results = manager.fetch_multiple_symbols(
        symbols=symbols,
        start="2023-01-01",
        end="2024-12-31",
    )

    print(f"\n  Fetched {len(results)} symbols:\n")
    for sym, result in results.items():
        status = "✓ OK   " if result.success else f"✗ FAIL ({result.error[:40]})"
        source = " [cache]" if result.from_cache else ""
        print(f"    {sym:12s} {status}  rows={result.rows:4d}{source}")

    # Build a combined panel (MultiIndex)
    frames = {
        sym: r.data
        for sym, r in results.items()
        if r.success and not r.data.empty
    }
    if frames:
        panel = pd.concat(frames, axis=1)
        print(f"\n  Combined panel shape: {panel.shape}")
        print(panel["close"].tail(3).to_string())

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Example 7 — Custom configuration (no cache, more retries)
# ─────────────────────────────────────────────────────────────────────────────

def example_custom_config() -> None:
    """Show how to configure the DataManager for different environments."""
    print("\n" + "=" * 60)
    print("  EXAMPLE 7 — Custom Configuration")
    print("=" * 60)

    # Production config: cache in /tmp, aggressive retries
    prod_config = DataManagerConfig(
        default_provider=Provider.YAHOO,
        cache_enabled=True,
        cache_dir="/tmp/ohlcv_cache",
        cache_max_age_days=1,
        retry_attempts=5,
        retry_backoff_seconds=2.0,
    )
    prod_manager = DataManager(config=prod_config)
    print(f"  Production manager: {prod_manager}")

    # Research config: no cache, single attempt (fail fast)
    research_config = DataManagerConfig(
        cache_enabled=False,
        retry_attempts=1,
    )
    research_manager = DataManager(config=research_config)
    print(f"  Research manager:   {research_manager}")

    # Fetch with the research manager
    df = research_manager.get_daily_data("AAPL", "2024-01-01", "2024-06-30")
    print(f"\n  AAPL (no cache)  rows={len(df)}  last={df['close'].iloc[-1]:.2f}")


# ─────────────────────────────────────────────────────────────────────────────
# Example 8 — Cache management
# ─────────────────────────────────────────────────────────────────────────────

def example_cache_management() -> None:
    """Demonstrate cache introspection and management."""
    print("\n" + "=" * 60)
    print("  EXAMPLE 8 — Cache Management")
    print("=" * 60)

    manager = DataManager()

    # Warm the cache
    manager.get_daily_data("NIFTY50", "2023-01-01", "2024-12-31")
    manager.get_daily_data("AAPL",    "2023-01-01", "2024-12-31")

    # Stats
    stats = manager.cache_stats()
    print(f"\n  Cache stats: {stats}")

    # Force refresh (bypass cache)
    df = manager.get_daily_data(
        "NIFTY50", "2023-01-01", "2024-12-31", refresh=True
    )
    print(f"\n  Force-refreshed NIFTY50  rows={len(df)}")


# ─────────────────────────────────────────────────────────────────────────────
# Example 9 — List available symbols
# ─────────────────────────────────────────────────────────────────────────────

def example_list_symbols() -> None:
    """Print all symbols in the canonical registry."""
    print("\n" + "=" * 60)
    print("  EXAMPLE 9 — Symbol Registry")
    print("=" * 60)

    manager = DataManager()
    symbols = manager.list_symbols()
    print(f"\n  {len(symbols)} registered symbols:")
    for sym in symbols:
        print(f"    {sym}")


# ─────────────────────────────────────────────────────────────────────────────
# Example 10 — Market Regime Engine integration pattern
# ─────────────────────────────────────────────────────────────────────────────

def example_regime_engine_integration() -> None:
    """
    Show exactly how to wire the DataManager into a Market Regime Engine.

    The engine typically expects:
    - A DataFrame with open/high/low/close/volume
    - DatetimeIndex (tz-naive)
    - Sorted ascending
    - No NaN rows in OHLC columns
    """
    print("\n" + "=" * 60)
    print("  EXAMPLE 10 — Market Regime Engine Integration")
    print("=" * 60)

    manager = DataManager(
        config=DataManagerConfig(
            cache_enabled=True,
            cache_dir=".cache/regime_engine",
            cache_max_age_days=1,
        )
    )

    # Fetch primary index
    nifty = manager.get_daily_data("^NSEI", start="2020-01-01", end="2025-01-01")

    # Verify all invariants expected by the regime engine
    checks = {
        "Lowercase columns":    set(nifty.columns) >= {"open", "high", "low", "close", "volume"},
        "DatetimeIndex":        isinstance(nifty.index, pd.DatetimeIndex),
        "TZ-naive":             nifty.index.tz is None,
        "Ascending":            nifty.index.is_monotonic_increasing,
        "No NaN in OHLC":       not nifty[["open","high","low","close"]].isnull().any().any(),
        "Float prices":         nifty["close"].dtype == "float64",
        "Int volume":           nifty["volume"].dtype == "int64",
    }

    print("\n  Schema validation:")
    all_ok = True
    for check, passed in checks.items():
        mark   = "✓" if passed else "✗"
        all_ok = all_ok and passed
        print(f"    {mark}  {check}")

    if all_ok:
        print("\n  ✓ DataFrame is fully compatible with Market Regime Engine")
    else:
        print("\n  ✗ Some checks failed — review the data pipeline")

    _print_df(nifty, "Final Nifty 50 frame for regime engine")


# ─────────────────────────────────────────────────────────────────────────────
# Main runner
# ─────────────────────────────────────────────────────────────────────────────

def run_all() -> None:
    """Execute all examples sequentially."""
    example_market_regime_engine()
    example_canonical_symbols()
    example_indices()
    example_stocks()
    example_fetch_latest()
    example_batch_fetch()
    example_custom_config()
    example_cache_management()
    example_list_symbols()
    example_regime_engine_integration()
    print("\n" + "=" * 60)
    print("  All examples complete.")
    print("=" * 60)


if __name__ == "__main__":
    import logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
        datefmt="%H:%M:%S",
    )
    run_all()
