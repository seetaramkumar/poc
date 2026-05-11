"""
stock_regime/main.py
=====================
Runnable demo for the Stock Regime Engine.

Generates synthetic OHLCV data for a small universe of stocks and a
benchmark index, runs the full pipeline, and prints a structured summary
to the console.

Run with:
    python main.py

To use real data from the trading_data layer, see the section at the
bottom of this file: "Wiring with trading_data.DataManager".
"""

from __future__ import annotations

import json
import logging
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT.parent))

from stock_regime.src import StockRegimeEngine
from stock_regime.src.models import MarketRegimeInput
from stock_regime.src.logging_config import configure_logging

warnings.filterwarnings("ignore")


# ══════════════════════════════════════════════════════════════════
#  1. Synthetic data generator
# ══════════════════════════════════════════════════════════════════

def _make_df(
    n_bars: int = 500,
    drift: float = 0.001,
    vol: float = 0.010,
    seed: int = 0,
    start_price: float = 18_000.0,
) -> pd.DataFrame:
    """Generate synthetic daily OHLCV data."""
    rng   = np.random.default_rng(seed)
    dates = pd.bdate_range(end="2024-12-31", periods=n_bars)
    close = np.empty(n_bars)
    price = start_price

    for i in range(n_bars):
        price = max(price * (1 + rng.normal(drift, vol)), 100.0)
        close[i] = price

    rng2        = np.random.default_rng(seed + 100)
    daily_range = close * 0.012
    high   = close + daily_range * rng2.uniform(0.3, 0.7, n_bars)
    low    = close - daily_range * rng2.uniform(0.3, 0.7, n_bars)
    open_  = low + (high - low) * rng2.uniform(0.0, 1.0, n_bars)
    volume = rng2.integers(5_000_000, 25_000_000, n_bars).astype(float)

    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
        index=dates,
    )


def generate_universe() -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    """
    Create a synthetic stock universe and a benchmark index.

    Returns
    -------
    tuple[dict, pd.DataFrame]
        (stock_data, benchmark_df)
    """
    stock_specs = {
        # symbol       drift     vol    seed  start_price  description
        "INFY":      (  0.0018,  0.008, 10,   1_800.0),   # Strong uptrend
        "RELIANCE":  (  0.0012,  0.009, 20,   2_500.0),   # Moderate uptrend
        "TCS":       (  0.0005,  0.006, 30,   3_200.0),   # Quiet / range
        "HDFCBANK":  ( -0.0010,  0.010, 40,   1_500.0),   # Mild downtrend
        "ICICIBANK": ( -0.0018,  0.009, 50,   1_000.0),   # Downtrend
        "WIPRO":     (  0.0000,  0.035, 60,     450.0),   # Volatile
        "LTIM":      (  0.0022,  0.009, 70,   5_000.0),   # Strong momentum
        "AXISBANK":  (  0.0001,  0.003, 80,     900.0),   # Quiet
        "SUNPHARMA": (  0.0008,  0.012, 90,   1_200.0),   # Moderate
        "BAJFINANCE":(  0.0015,  0.011,100,   7_000.0),   # Uptrend
    }

    stock_data: dict[str, pd.DataFrame] = {}
    for symbol, (drift, vol, seed, start) in stock_specs.items():
        stock_data[symbol] = _make_df(drift=drift, vol=vol, seed=seed, start_price=start)

    # Benchmark: Nifty-like index, moderate positive drift
    benchmark_df = _make_df(drift=0.0008, vol=0.008, seed=999, start_price=18_000.0)

    return stock_data, benchmark_df


# ══════════════════════════════════════════════════════════════════
#  2. Main demo
# ══════════════════════════════════════════════════════════════════

def main() -> None:
    # ── Logging ──────────────────────────────────────────────────
    logger = configure_logging(log_dir="output/logs")
    logger.info("Stock Regime Engine demo starting.")

    print("=" * 70)
    print("  Stock Regime Engine — Demo")
    print("=" * 70)

    # ── Synthetic universe ────────────────────────────────────────
    stock_data, benchmark_df = generate_universe()
    print(f"\n✓ Generated {len(stock_data)} synthetic stocks + benchmark")
    print(f"  Date range : {list(stock_data.values())[0].index[0].date()} "
          f"→ {list(stock_data.values())[0].index[-1].date()}")

    # ── Market regime context (from Market Regime Engine output) ──
    market_regime = MarketRegimeInput.from_dict({
        "regime":     "BULLISH_TREND",
        "confidence": 0.82,
    })
    print(f"\n  Market Regime : {market_regime.regime} "
          f"(confidence: {market_regime.confidence:.0%})")

    # ── Instantiate engine ────────────────────────────────────────
    engine = StockRegimeEngine(output_dir="output")

    # ── Run full universe classification ──────────────────────────
    print("\n" + "─" * 70)
    print("  Classifying universe …")
    print("─" * 70)

    results = engine.analyze_universe(
        stock_data     = stock_data,
        market_regime  = market_regime,
        benchmark_data = benchmark_df,
        market_label   = "NIFTY500",
        persist        = True,
    )

    # ── Per-stock summary table ───────────────────────────────────
    print(f"\n  {'Symbol':<12} {'Regime':<18} {'Conf':>6}  "
          f"{'Trend':>6}  {'Mom':>6}  {'Vol':>6}  {'RS':>7}")
    print("  " + "─" * 65)

    for r in results:
        ds = r.dimensional_scores
        rs = r.indicators.relative_strength
        rs_str = f"{rs:.3f}" if rs is not None else "  N/A "
        regime_str = r.stock_regime.value
        mark = "✗" if not r.is_valid() else " "
        print(
            f"  {mark}{r.symbol:<11} {regime_str:<18} {r.confidence:>6.2f}  "
            f"{ds.trend:>6.2f}  {ds.momentum:>6.2f}  {ds.volatility:>6.2f}  {rs_str:>7}"
        )

    # ── Regime distribution ───────────────────────────────────────
    print("\n" + "─" * 70)
    print("  Regime Distribution")
    print("─" * 70)

    counts: dict[str, int] = {}
    for r in results:
        key = r.stock_regime.value
        counts[key] = counts.get(key, 0) + 1

    total = len(results)
    for regime, count in sorted(counts.items(), key=lambda x: -x[1]):
        bar = "█" * int(count / total * 30)
        pct = count / total * 100
        print(f"  {regime:<18}  {bar:<30}  {count:>3}  ({pct:5.1f}%)")

    # ── Rankings ──────────────────────────────────────────────────
    rankings = engine.get_rankings(results)

    print("\n" + "─" * 70)
    print("  Top 5 by Trend Strength")
    print("─" * 70)
    for entry in rankings.strongest_trends[:5]:
        print(f"  #{entry.rank}  {entry.symbol:<12} {entry.stock_regime:<18}  "
              f"trend={entry.score:.3f}  conf={entry.confidence:.2f}")

    print("\n" + "─" * 70)
    print("  Top 5 by Momentum")
    print("─" * 70)
    for entry in rankings.strongest_momentum[:5]:
        print(f"  #{entry.rank}  {entry.symbol:<12} {entry.stock_regime:<18}  "
              f"momentum={entry.score:.3f}  conf={entry.confidence:.2f}")

    print("\n" + "─" * 70)
    print("  Top 5 by Volatility Expansion")
    print("─" * 70)
    for entry in rankings.highest_volatility[:5]:
        print(f"  #{entry.rank}  {entry.symbol:<12} {entry.stock_regime:<18}  "
              f"volatility={entry.score:.3f}  conf={entry.confidence:.2f}")

    # ── Detailed JSON output for one stock ────────────────────────
    print("\n" + "─" * 70)
    print("  Detailed JSON — INFY")
    print("─" * 70)
    infy_result = next((r for r in results if r.symbol == "INFY"), None)
    if infy_result:
        print(json.dumps(infy_result.to_dict(), indent=2))

    # ── Output location ───────────────────────────────────────────
    print("\n" + "─" * 70)
    print("  Persisted Output Files")
    print("─" * 70)
    for p in sorted(Path("output").rglob("*.parquet")):
        size_kb = p.stat().st_size / 1024
        print(f"  {str(p):<60}  {size_kb:6.1f} KB")

    print("\n" + "=" * 70)
    print("  Done.")
    print("=" * 70)

    logger.info("Stock Regime Engine demo complete.")


# ══════════════════════════════════════════════════════════════════
#  3. Wiring with trading_data.DataManager  (reference pattern)
# ══════════════════════════════════════════════════════════════════

REAL_DATA_EXAMPLE = '''
# ── How to wire trading_data into the Stock Regime Engine ─────────────────
#
#    from trading_data import DataManager
#    from market_regime.src.engine import MarketRegimeEngine
#    from stock_regime.src import StockRegimeEngine
#    from stock_regime.src.models import MarketRegimeInput
#
#    # 1. Fetch benchmark (Nifty 50 or S&P 500)
#    dm = DataManager()
#    benchmark_df = dm.get_daily_data("NIFTY50", start="2020-01-01", end="2025-01-01")
#
#    # 2. Get market regime context
#    market_engine = MarketRegimeEngine()
#    market_result = market_engine.analyze(benchmark_df)
#    market_ctx = MarketRegimeInput.from_dict(market_result.to_dict())
#
#    # 3. Fetch universe stocks
#    symbols = ["INFY", "RELIANCE", "TCS", "HDFCBANK", "WIPRO"]
#    fetch_results = dm.fetch_multiple_symbols(
#        symbols, start="2020-01-01", end="2025-01-01"
#    )
#    stock_data = {
#        sym: res.data
#        for sym, res in fetch_results.items()
#        if res.success
#    }
#
#    # 4. Run Stock Regime Engine
#    stock_engine = StockRegimeEngine(output_dir="output")
#    results = stock_engine.analyze_universe(
#        stock_data     = stock_data,
#        market_regime  = market_ctx,
#        benchmark_data = benchmark_df,
#        market_label   = "NIFTY500",
#        persist        = True,
#    )
'''


if __name__ == "__main__":
    main()
