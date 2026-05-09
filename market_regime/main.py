"""
main.py — Sample usage of the Market Regime Engine
===================================================
Generates synthetic NIFTY-like OHLCV data and runs the engine on it,
demonstrating:

  • Single-bar classification (latest bar)
  • JSON output
  • Rolling classification over the full history
  • How to plug in real CSV data
"""

from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

# ── Make sure the project root is on the path ─────────────────────
ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from src import MarketRegimeEngine  # noqa: E402

warnings.filterwarnings("ignore")


# ══════════════════════════════════════════════════════════════════
#  1.  Synthetic data generator
#      Produces realistic-ish OHLCV data for NIFTY 50.
# ══════════════════════════════════════════════════════════════════

def generate_synthetic_nifty(
    n_bars: int = 600,
    seed: int   = 42,
) -> pd.DataFrame:
    """
    Build a DataFrame of synthetic daily NIFTY OHLCV data.

    The series passes through four rough market regimes so you can
    see the engine classify each of them:
      bars   0-149  → Quiet bullish drift
      bars 150-299  → Strong bullish trend
      bars 300-399  → Volatile (sharp swings)
      bars 400-499  → Bearish trend
      bars 500-599  → Sideways / consolidation
    """
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(end="2024-12-31", periods=n_bars)

    close = np.empty(n_bars)
    price = 18_000.0

    for i in range(n_bars):
        if i < 150:                  # Quiet bullish drift
            drift = 0.0003
            vol   = 0.004
        elif i < 300:                # Strong bullish trend
            drift = 0.0010
            vol   = 0.010
        elif i < 400:                # Volatile
            drift = 0.0
            vol   = 0.025
        elif i < 500:                # Bearish trend
            drift = -0.0008
            vol   = 0.012
        else:                        # Sideways
            drift = 0.0001
            vol   = 0.005

        ret   = rng.normal(drift, vol)
        price = max(price * (1 + ret), 5_000)
        close[i] = price

    # Realistic OHLC from close
    daily_range = close * rng.uniform(0.005, 0.02, n_bars)
    high   = close + daily_range * rng.uniform(0.3, 0.7, n_bars)
    low    = close - daily_range * rng.uniform(0.3, 0.7, n_bars)
    open_  = low   + (high - low) * rng.uniform(0, 1, n_bars)
    volume = rng.integers(8_000_000, 20_000_000, n_bars).astype(float)

    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
        index=dates,
    )


# ══════════════════════════════════════════════════════════════════
#  2.  Main demo
# ══════════════════════════════════════════════════════════════════

def main() -> None:
    print("=" * 60)
    print("  Market Regime Engine — Demo")
    print("=" * 60)

    # ── Generate data ─────────────────────────────────────────────
    df = generate_synthetic_nifty(n_bars=600)
    print(f"\n✓ Generated {len(df)} bars of synthetic NIFTY data")
    print(f"  Date range : {df.index[0].date()} → {df.index[-1].date()}")
    print(f"  Close range: {df['close'].min():,.0f} → {df['close'].max():,.0f}")

    # ── Instantiate engine ────────────────────────────────────────
    engine = MarketRegimeEngine()

    # ── Single-bar classification (latest bar) ────────────────────
    print("\n" + "─" * 60)
    print("  Latest-bar Classification")
    print("─" * 60)
    result = engine.analyze(df)
    print(json.dumps(result.to_dict(), indent=2))

    # ── Snapshot values for inspection ────────────────────────────
    snap = result.indicator_snapshot
    print("\n  Indicator Snapshot (latest bar)")
    print(f"    Close   : {snap.close:,.2f}")
    print(f"    EMA20   : {snap.ema20:,.2f}")
    print(f"    EMA50   : {snap.ema50:,.2f}")
    print(f"    EMA200  : {snap.ema200:,.2f}")
    print(f"    ADX     : {snap.adx:.2f}")
    print(f"    ATR     : {snap.atr:,.2f}")
    print(f"    ATR_MA  : {snap.atr_ma:,.2f}")
    print(f"    Volume  : {snap.volume:,.0f}")
    print(f"    Vol_MA  : {snap.volume_ma:,.0f}")

    # ── Spot-check five specific bars (one per regime region) ──────
    print("\n" + "─" * 60)
    print("  Spot-check: One Bar per Synthetic Regime Region")
    print("─" * 60)

    checkpoints = {
        "Bar 100  (Quiet drift)  ": 100,
        "Bar 250  (Strong bull)  ": 250,
        "Bar 350  (Volatile)     ": 350,
        "Bar 450  (Bearish)      ": 450,
        "Bar 550  (Sideways)     ": 550,
    }

    for label, idx in checkpoints.items():
        res = engine.analyze(df, row_index=idx)
        s = res.indicator_snapshot
        print(
            f"  {label} | {res.regime.value:<15} "
            f"conf={res.confidence:.2f} | "
            f"ADX={s.adx:.1f}  ATR/ATR_MA="
            f"{(s.atr / s.atr_ma) if (s.atr and s.atr_ma) else 0:.2f}"
        )

    # ── Rolling classification summary ─────────────────────────────
    print("\n" + "─" * 60)
    print("  Rolling Classification — Regime Distribution")
    print("─" * 60)

    all_results = engine.analyze_rolling(df)
    regime_counts: dict = {}
    for r in all_results:
        key = r.regime.value
        regime_counts[key] = regime_counts.get(key, 0) + 1

    total = len(all_results)
    for regime, count in sorted(regime_counts.items(), key=lambda x: -x[1]):
        bar  = "█" * int(count / total * 40)
        pct  = count / total * 100
        print(f"  {regime:<16} {bar:<40} {count:>4} bars ({pct:5.1f} %)")

    # ── How to use real CSV data ───────────────────────────────────
    print("\n" + "─" * 60)
    print("  Using Real CSV Data")
    print("─" * 60)
    print("""
  # Your CSV must have columns: date, open, high, low, close, volume
  df_real = pd.read_csv(
      "nifty_daily.csv",
      parse_dates=["date"],
      index_col="date",
  )
  result = engine.analyze(df_real)
  print(json.dumps(result.to_dict(), indent=2))
""")

    print("=" * 60)
    print("  Done.")
    print("=" * 60)


if __name__ == "__main__":
    main()
