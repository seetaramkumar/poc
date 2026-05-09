"""
main.py — Market Regime Engine with live market data
=====================================================
Fetches real OHLCV data via the trading_data DataManager and runs
the regime engine on it.

Data source priority
--------------------
1. trading_data.DataManager  (Yahoo Finance + local parquet cache)
2. Synthetic fallback         (offline / CI environments only)

Usage
-----
    # Analyse NIFTY 50 (last 3 years)
    python main.py

    # Analyse a different symbol
    python main.py --symbol SP500

    # Force a live fetch, bypass cache
    python main.py --symbol AAPL --refresh

    # Offline mode: use synthetic data (no network required)
    python main.py --synthetic
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import warnings
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

# ── Project root on path ──────────────────────────────────────────
ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from src.engine import MarketRegimeEngine          # noqa: E402
from src.data_adapter import DataAdapter   # noqa: E402

warnings.filterwarnings("ignore")
logging.basicConfig(
    level=logging.WARNING,
    format="%(levelname)-8s  %(name)s  %(message)s",
)


# ══════════════════════════════════════════════════════════════════
#  Synthetic fallback (offline / testing only)
# ══════════════════════════════════════════════════════════════════

def _synthetic_nifty(n_bars: int = 600, seed: int = 42) -> pd.DataFrame:
    """
    Generate synthetic NIFTY-like OHLCV data.
    Used only when --synthetic flag is passed or network is unavailable.
    """
    rng   = np.random.default_rng(seed)
    today = pd.Timestamp(date.today())
    end_bd = today if today.weekday() < 5 else today - pd.offsets.BDay(1)
    dates = pd.bdate_range(end=end_bd, periods=n_bars)
    close = np.empty(n_bars)
    price = 18_000.0

    for i in range(n_bars):
        if   i < 150: drift, vol = 0.0003, 0.004
        elif i < 300: drift, vol = 0.0010, 0.010
        elif i < 400: drift, vol = 0.0000, 0.025
        elif i < 500: drift, vol = -0.0008, 0.012
        else:         drift, vol = 0.0001, 0.005

        price    = max(price * (1 + rng.normal(drift, vol)), 5_000)
        close[i] = price

    rng2      = np.random.default_rng(seed + 1)
    rng3      = np.random.default_rng(seed + 2)
    d_range   = close * rng2.uniform(0.005, 0.02, n_bars)
    hi        = close + d_range * rng2.uniform(0.3, 0.7, n_bars)
    lo        = close - d_range * rng2.uniform(0.3, 0.7, n_bars)
    op        = lo + (hi - lo) * rng3.uniform(0, 1, n_bars)
    vol_data  = rng3.integers(8_000_000, 20_000_000, n_bars).astype(float)

    return pd.DataFrame(
        {"open": op, "high": hi, "low": lo, "close": close, "volume": vol_data},
        index=dates,
    )


# ══════════════════════════════════════════════════════════════════
#  Data loading
# ══════════════════════════════════════════════════════════════════

def load_data(symbol: str, start: str, end: str, refresh: bool) -> pd.DataFrame:
    """Fetch via DataAdapter (Yahoo Finance + parquet cache)."""
    adapter = DataAdapter(cache_dir=str(ROOT / ".cache" / "ohlcv"))
    return adapter.fetch(symbol, start=start, end=end, refresh=refresh)


# ══════════════════════════════════════════════════════════════════
#  Helpers
# ══════════════════════════════════════════════════════════════════

def _section(title: str) -> None:
    print(f"\n{'─' * 60}\n  {title}\n{'─' * 60}")


def _snap_line(label: str, value, fmt: str = ",.2f") -> None:
    val = f"{value:{fmt}}" if value is not None else "N/A"
    print(f"    {label:<12}: {val:>14}")


# ══════════════════════════════════════════════════════════════════
#  Main
# ══════════════════════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser(description="Market Regime Engine")
    parser.add_argument("--symbol",    default="NIFTY50",
                        help="Symbol to analyse (default: NIFTY50)")
    parser.add_argument("--start",     default=None,
                        help="Start date YYYY-MM-DD (default: 3 years ago)")
    parser.add_argument("--end",       default=None,
                        help="End date YYYY-MM-DD (default: today)")
    parser.add_argument("--refresh",   action="store_true",
                        help="Force live fetch, bypass cache")
    parser.add_argument("--synthetic", action="store_true",
                        help="Use synthetic data — no network required")
    args = parser.parse_args()

    end_date   = args.end   or date.today().isoformat()
    start_date = args.start or (date.today() - timedelta(days=3 * 365)).isoformat()

    print("=" * 60)
    print("  Market Regime Engine")
    print("=" * 60)

    # ── Load data ─────────────────────────────────────────────────
    if args.synthetic:
        df     = _synthetic_nifty()
        source = "Synthetic NIFTY (offline mode)"
        print("\n  Mode   : SYNTHETIC")
    else:
        print(f"\n  Fetching {args.symbol}  {start_date} → {end_date} …")
        try:
            df     = load_data(args.symbol, start_date, end_date, args.refresh)
            source = f"{args.symbol} via Yahoo Finance / parquet cache"
        except Exception as exc:
            print(f"\n  ✗ Fetch failed: {exc}")
            print("  Falling back to synthetic data.\n")
            df     = _synthetic_nifty()
            source = "Synthetic NIFTY (network fallback)"

    print(f"\n  Source : {source}")
    print(f"  Bars   : {len(df)}")
    print(f"  Range  : {df.index[0].date()} → {df.index[-1].date()}")
    print(f"  Close  : {df['close'].iloc[0]:,.2f} → {df['close'].iloc[-1]:,.2f}")

    # ── Engine ────────────────────────────────────────────────────
    engine = MarketRegimeEngine()

    # ── Latest-bar classification ─────────────────────────────────
    _section("Latest-bar Classification")
    result = engine.analyze(df)
    print(json.dumps(result.to_dict(), indent=2))

    # ── Indicator snapshot ────────────────────────────────────────
    snap = result.indicator_snapshot
    _section("Indicator Snapshot (latest bar)")
    _snap_line("Close",     snap.close)
    _snap_line("EMA 20",    snap.ema20)
    _snap_line("EMA 50",    snap.ema50)
    _snap_line("EMA 200",   snap.ema200)
    _snap_line("ADX",       snap.adx,       fmt=".2f")
    _snap_line("ATR",       snap.atr)
    _snap_line("ATR MA",    snap.atr_ma)
    _snap_line("Volume",    snap.volume,    fmt=",.0f")
    _snap_line("Volume MA", snap.volume_ma, fmt=",.0f")

    # ── Rolling regime distribution ───────────────────────────────
    _section("Rolling Classification — Regime Distribution")
    all_results   = engine.analyze_rolling(df)
    counts: dict[str, int] = {}
    for r in all_results:
        counts[r.regime.value] = counts.get(r.regime.value, 0) + 1

    total = len(all_results)
    for regime, count in sorted(counts.items(), key=lambda x: -x[1]):
        bar = "█" * int(count / total * 40)
        pct = count / total * 100
        print(f"  {regime:<16} {bar:<40} {count:>4} bars ({pct:5.1f} %)")

    # ── Spot-check: evenly-spaced bars ────────────────────────────
    _section("Spot-check: Sample Bars Across History")
    n       = len(df)
    indices = [int(n * p) for p in (0.2, 0.4, 0.6, 0.8, 0.95)]
    for idx in indices:
        idx = min(idx, n - 1)
        d   = df.index[idx].date()
        res = engine.analyze(df, row_index=idx)
        s   = res.indicator_snapshot
        atr_ratio = (s.atr / s.atr_ma) if (s.atr and s.atr_ma) else 0.0
        print(
            f"  Bar {idx:>4}  ({d})  |  {res.regime.value:<15} "
            f"conf={res.confidence:.2f}  ADX={s.adx:.1f}  ATR/MA={atr_ratio:.2f}"
        )

    print("\n" + "=" * 60)
    print("  Done.")
    print("=" * 60)


if __name__ == "__main__":
    main()
