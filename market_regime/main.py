"""
market_regime/main.py
======================
Standalone demo for the Market Regime Engine.

Previously this file used a DataAdapter class that lived in src/.
That adapter has been removed — data sourcing is now handled by
DataManager from the trading_data package (a sibling module).

The Market Regime Engine itself is unchanged.  Only this demo
script is updated to call DataManager directly.

Run from the project root:
    cd algo_platform/
    python market_regime/main.py

For the full pipeline (market + stocks together):
    python runner/main.py
"""

from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

# ── Ensure project root is on sys.path ───────────────────────────────────────
ROOT = Path(__file__).parent.parent   # algo_platform/
sys.path.insert(0, str(ROOT))

from market_regime.src import MarketRegimeEngine   # noqa: E402

warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────────────────────────────────────
#  Option A: Real data via DataManager (recommended)
# ─────────────────────────────────────────────────────────────────────────────

def run_with_real_data(symbol: str = "NIFTY50") -> None:
    """
    Fetch live OHLCV data via DataManager and classify the latest bar.

    DataManager handles:
      - symbol resolution  (e.g. "NIFTY50" → "^NSEI")
      - local parquet cache (stale after 1 day)
      - retry / back-off on network failure

    Parameters
    ----------
    symbol :
        Canonical name ("NIFTY50", "SP500") or Yahoo ticker ("^NSEI").
    """
    from trading_data import DataManager, DataManagerConfig

    print(f"\n  Fetching {symbol} via DataManager …")

    manager = DataManager(
        config=DataManagerConfig(
            cache_enabled      = True,
            cache_dir          = ".cache/ohlcv",
            cache_max_age_days = 1,
        )
    )

    df = manager.get_daily_data(symbol, start="2020-01-01", end="2025-01-01")
    print(f"  {symbol}: {len(df)} bars  "
          f"({df.index[0].date()} → {df.index[-1].date()})")

    engine = MarketRegimeEngine()
    result = engine.analyze(df)

    print("\n  Latest-bar classification:")
    print(json.dumps(result.to_dict(), indent=2))


# ─────────────────────────────────────────────────────────────────────────────
#  Option B: Synthetic data (offline demo, no network required)
# ─────────────────────────────────────────────────────────────────────────────

def _make_synthetic(
    n_bars: int = 600,
    seed: int   = 42,
) -> pd.DataFrame:
    """Generate synthetic NIFTY-like OHLCV data across five regime phases."""
    rng   = np.random.default_rng(seed)
    dates = pd.bdate_range(end="2024-12-31", periods=n_bars)
    close = np.empty(n_bars)
    price = 18_000.0

    for i in range(n_bars):
        if   i < 150: drift, vol = 0.0003, 0.004    # quiet drift
        elif i < 300: drift, vol = 0.0010, 0.010    # strong bull
        elif i < 400: drift, vol = 0.0,    0.025    # volatile
        elif i < 500: drift, vol = -0.0008, 0.012   # bear
        else:         drift, vol = 0.0001, 0.005    # sideways

        price    = max(price * (1 + rng.normal(drift, vol)), 5_000)
        close[i] = price

    dr    = close * rng.uniform(0.005, 0.02, n_bars)
    high  = close + dr * rng.uniform(0.3, 0.7, n_bars)
    low   = close - dr * rng.uniform(0.3, 0.7, n_bars)
    open_ = low   + (high - low) * rng.uniform(0, 1, n_bars)
    vol   = rng.integers(8_000_000, 20_000_000, n_bars).astype(float)

    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": vol},
        index=dates,
    )


def run_with_synthetic_data() -> None:
    """Run the demo entirely offline using generated data."""
    print("=" * 60)
    print("  Market Regime Engine — Synthetic Data Demo")
    print("=" * 60)

    df = _make_synthetic(n_bars=600)
    print(f"\n✓ Generated {len(df)} bars  "
          f"({df.index[0].date()} → {df.index[-1].date()})")

    engine = MarketRegimeEngine()

    # ── Latest bar ─────────────────────────────────────────────────
    print("\n─── Latest-bar Classification ───")
    result = engine.analyze(df)
    print(json.dumps(result.to_dict(), indent=2))

    # ── Spot-checks ────────────────────────────────────────────────
    print("\n─── Spot-checks (one bar per synthetic phase) ───")
    checks = {
        "Bar 100 (quiet)":    100,
        "Bar 250 (bull)":     250,
        "Bar 350 (volatile)": 350,
        "Bar 450 (bear)":     450,
        "Bar 550 (sideways)": 550,
    }
    for label, idx in checks.items():
        r = engine.analyze(df, row_index=idx)
        s = r.indicator_snapshot
        print(
            f"  {label:<25} → {r.regime.value:<15}  "
            f"conf={r.confidence:.2f}  ADX={s.adx:.1f}"
        )

    # ── Rolling distribution ────────────────────────────────────────
    print("\n─── Rolling Regime Distribution ───")
    all_results = engine.analyze_rolling(df)
    counts: dict[str, int] = {}
    for r in all_results:
        counts[r.regime.value] = counts.get(r.regime.value, 0) + 1
    total = len(all_results)
    for regime, count in sorted(counts.items(), key=lambda x: -x[1]):
        bar = "█" * int(count / total * 35)
        print(f"  {regime:<16} {bar:<35} {count:>4}  ({count/total*100:5.1f}%)")

    print("\n  Note: for real data, call run_with_real_data('NIFTY50')")
    print("  Note: for the full pipeline, run:  python runner/main.py")
    print("=" * 60)


# ─────────────────────────────────────────────────────────────────────────────
#  Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Market Regime Engine demo")
    parser.add_argument(
        "--real", action="store_true",
        help="Fetch real data via DataManager (requires network)",
    )
    parser.add_argument(
        "--symbol", default="NIFTY50",
        help="Symbol to fetch when --real is set (default: NIFTY50)",
    )
    args = parser.parse_args()

    if args.real:
        run_with_real_data(args.symbol)
    else:
        run_with_synthetic_data()