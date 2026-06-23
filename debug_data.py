#!/usr/bin/env python
"""Debug script to check data fetching and staleness."""

import sys
from datetime import date, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from trading_data import DataManager, DataManagerConfig
from stock_regime.filters.history_filter import HistoryFilter

# Fetch some sample stocks to check data freshness
manager = DataManager(config=DataManagerConfig(cache_enabled=True))

# Test with a benchmark and a few symbols
test_symbols = ["^NSEI", "INFY", "TCS", "RELIANCE"]

history_filter = HistoryFilter(
    min_bars=300,
    max_gap_days=5,
    max_stale_days=3,
    gap_lookback_days=90,
)

print("=" * 70)
print("DATA FRESHNESS DIAGNOSTIC")
print("=" * 70)
print()

for symbol in test_symbols:
    try:
        df = manager.get_daily_data(symbol, start="2021-01-01", end="today")
        
        if df.empty:
            print(f"❌ {symbol:15} | No data fetched")
            continue
        
        last_bar = df.index[-1].date()
        today = date.today()
        age_days = (today - last_bar).days
        n_bars = len(df)
        
        # Check history filter
        reasons = history_filter.check(symbol, df)
        status = "✓ PASS" if not reasons else "✗ FAIL"
        
        print(f"{status}  {symbol:15} | Bars={n_bars:4d} | Last={last_bar} | Age={age_days} days", end="")
        
        if reasons:
            print(f" | Reason: {reasons[0].check}")
        else:
            print()
            
    except Exception as e:
        print(f"❌ {symbol:15} | Error: {type(e).__name__}: {e}")

print()
print("=" * 70)
print("CONFIG SETTINGS")
print("=" * 70)
print(f"max_stale_days: {history_filter.max_stale_days}")
print(f"max_gap_days:   {history_filter.max_gap_days}")
print(f"min_bars:       {history_filter.min_bars}")
print()
print(f"Today's date: {date.today()}")
print()
