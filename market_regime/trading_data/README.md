# trading_data — Modular Data Sourcing Layer

Production-grade OHLCV data layer for algorithmic trading systems.  
Pluggable provider architecture · Local parquet cache · Market Regime Engine compatible.

---

## Quick Start

```python
from trading_data import DataManager

manager = DataManager()

df = manager.get_daily_data(
    symbol="^NSEI",        # or "NIFTY50" — canonical names work too
    start="2020-01-01",
    end="2025-01-01",
)
# df is immediately consumable by the Market Regime Engine
```

**Returned DataFrame schema** (guaranteed):

| Column   | dtype   | Description              |
|----------|---------|--------------------------|
| `open`   | float64 | Opening price            |
| `high`   | float64 | Session high             |
| `low`    | float64 | Session low              |
| `close`  | float64 | Closing / adjusted close |
| `volume` | int64   | Volume traded            |

- **Index**: `DatetimeIndex` named `"date"`, tz-naive, ascending.
- **No NaN** in any OHLC column.

---

## Project Structure

```
trading_data/
│
├── __init__.py                  ← Public surface; re-exports everything
├── manager.py                   ← DataManager (primary entry point)
├── models.py                    ← Shared types, enums, dataclasses
├── exceptions.py                ← Exception hierarchy
│
├── providers/
│   ├── __init__.py
│   ├── base.py                  ← BaseDataProvider (abstract)
│   ├── yahoo.py                 ← YahooFinanceProvider  ✓ implemented
│   ├── zerodha.py               ← ZerodhaProvider       ◑ stub
│   ├── polygon.py               ← PolygonProvider       ◑ stub
│   └── ibkr.py                  ← IBKRProvider          ◑ stub
│
├── cache/
│   ├── __init__.py
│   └── parquet_cache.py         ← ParquetCache
│
├── symbols/
│   ├── __init__.py
│   └── mapper.py                ← SymbolMapper
│
├── examples/
│   └── basic_usage.py           ← 10 runnable examples
│
└── requirements.txt
```

---

## Module Responsibilities

### `manager.py` — DataManager

The **single public entry point** for all consumers.

| Responsibility | How |
|---|---|
| Symbol resolution | Delegates to `SymbolMapper` |
| Cache-first reads | Checks `ParquetCache` before network |
| Partial cache merging | Merges cached ranges with live data |
| Retry / back-off | Exponential back-off around provider calls |
| Provider routing | Instantiates the correct `BaseDataProvider` subclass |
| Batch fetching | Calls `fetch_multiple_symbols()`; isolates per-symbol errors |
| Cache invalidation | `clear_cache()`, `refresh=True` parameter |

### `providers/base.py` — BaseDataProvider

Abstract base class that **defines the contract** every provider must honour.

- `fetch_ohlcv(symbol, start, end, interval)` — **must** override
- `fetch_latest(symbol)` — default implementation (uses `fetch_ohlcv`)
- `fetch_multiple_symbols(symbols, …)` — default sequential implementation
- `validate_connection()` — optional override
- `normalise(df, symbol, provider)` — **static utility**; all providers call this to guarantee output schema

The `normalise()` method is the schema contract enforcer:
1. Lower-cases all column names
2. Handles yfinance MultiIndex columns
3. Selects only `open, high, low, close, volume`
4. Enforces `float64` / `int64` dtypes
5. Drops all-NaN OHLC rows
6. Sets a tz-naive `DatetimeIndex` named `"date"`

### `providers/yahoo.py` — YahooFinanceProvider

Active provider using `yfinance`.

- Uses `yf.download()` for both single and batch fetches
- Overrides `fetch_multiple_symbols()` to exploit `yf.download`'s native multi-ticker support (faster than sequential for large lists)
- `auto_adjust=True` by default (split + dividend adjusted prices)
- Covers: NSE (`.NS`), BSE (`.BO`), US equities, global indices (`^NSEI`, `^GSPC`, etc.)

### `providers/zerodha.py` / `polygon.py` / `ibkr.py` — Stubs

Full interface skeletons with:
- Complete docstrings showing exactly what needs to be implemented
- Step-by-step implementation notes referencing the correct SDK calls
- `ConfigurationError` raised on missing credentials at construction time

### `cache/parquet_cache.py` — ParquetCache

| Feature | Detail |
|---|---|
| Format | Apache Parquet (via PyArrow) |
| Compression | Snappy |
| Staleness | Configurable `max_age_days` |
| Path scheme | `<root>/<provider>/<interval>/<symbol>.parquet` |
| Thread safety | POSIX atomic writes |
| Degradation | Stale files served with warning if network fails |

Methods: `read()`, `write()`, `is_stale()`, `invalidate()`, `clear_all()`, `stats()`

### `symbols/mapper.py` — SymbolMapper

Translates user-facing canonical names to provider-specific tickers.

```python
mapper = SymbolMapper()
info = mapper.resolve("NIFTY50", Provider.YAHOO)
# info.provider_symbol == "^NSEI"
# info.asset_class     == AssetClass.INDEX
# info.exchange        == "NSE"
# info.currency        == "INR"
```

- Static registry covers 13 symbols (indices + major equities)
- **Passthrough strategy**: unknown symbols are forwarded unchanged
- Runtime registration via `mapper.register(canonical, entry)`
- Custom registry JSON file mountable at construction

### `models.py` — Shared Types

| Type | Purpose |
|---|---|
| `OHLCVFrame` | Type alias for `pd.DataFrame` |
| `DateLike` | Union of `str | date | datetime` |
| `AssetClass` | `INDEX, EQUITY, ETF, CRYPTO, FX` |
| `Provider` | `YAHOO, ZERODHA, POLYGON, IBKR` |
| `SymbolInfo` | Canonical + provider ticker + metadata |
| `FetchResult` | Data + success + error + cache flag |
| `DataManagerConfig` | All DataManager settings in one place |

### `exceptions.py` — Exception Hierarchy

```
DataLayerError
├── ProviderError(provider, symbol, reason)
├── SymbolNotFoundError(symbol, provider)
├── CacheError
└── ConfigurationError(provider, detail)
```

---

## All Public Methods

```python
manager = DataManager()

# Primary (cache-aware)
df  = manager.get_daily_data(symbol, start, end, interval="1d", refresh=False)
df  = manager.fetch_latest(symbol, lookback_days=5)
res = manager.fetch_multiple_symbols(symbols, start, end, interval="1d")

# Raw (bypass cache)
df  = manager.fetch_ohlcv(symbol, start, end, interval="1d")

# Utility
syms = manager.list_symbols()
ok   = manager.validate_provider()
info = manager.cache_stats()
n    = manager.clear_cache()
```

---

## Configuration

```python
from trading_data import DataManager, DataManagerConfig, Provider

config = DataManagerConfig(
    default_provider      = Provider.YAHOO,   # switch to POLYGON / IBKR / ZERODHA
    cache_enabled         = True,
    cache_dir             = ".cache/ohlcv",
    cache_max_age_days    = 1,
    retry_attempts        = 3,
    retry_backoff_seconds = 1.0,             # doubles each attempt
)

manager = DataManager(config=config)
```

---

## Symbol Reference

| Canonical | Yahoo ticker | Exchange | Asset class |
|-----------|-------------|----------|-------------|
| `NIFTY50` | `^NSEI`     | NSE      | Index       |
| `BANKNIFTY` | `^NSEBANK` | NSE    | Index       |
| `SP500`   | `^GSPC`     | NYSE     | Index       |
| `NASDAQ`  | `^IXIC`     | NASDAQ   | Index       |
| `DOW`     | `^DJI`      | NYSE     | Index       |
| `NIFTYIT` | `^CNXIT`    | NSE      | Index       |
| `RELIANCE` | `RELIANCE.NS` | NSE  | Equity      |
| `TCS`     | `TCS.NS`    | NSE      | Equity      |
| `INFY`    | `INFY.NS`   | NSE      | Equity      |
| `HDFCBANK` | `HDFCBANK.NS` | NSE  | Equity      |
| `AAPL`    | `AAPL`      | NASDAQ   | Equity      |
| `MSFT`    | `MSFT`      | NASDAQ   | Equity      |
| `GOOGL`   | `GOOGL`     | NASDAQ   | Equity      |

Unknown symbols are passed through to the provider unchanged.

---

## Adding a New Provider

1. Create `trading_data/providers/myprovider.py`
2. Subclass `BaseDataProvider`
3. Implement `name` property and `fetch_ohlcv()`
4. Always call `self.normalise(raw_df, symbol, self.name)` before returning
5. Add `Provider.MYPROVIDER` to `models.py`
6. Add the provider symbol mapping to `SymbolMapper._STATIC_REGISTRY`
7. Add a factory case in `manager._build_provider()`

```python
class MyProvider(BaseDataProvider):
    @property
    def name(self) -> str:
        return "myprovider"

    def fetch_ohlcv(self, symbol, start, end, interval="1d"):
        raw = ...           # call your API
        return self.normalise(raw, symbol=symbol, provider=self.name)
```

---

## Installation

```bash
pip install -r trading_data/requirements.txt
```

Optional provider extras:
```bash
pip install kiteconnect           # Zerodha
pip install polygon-api-client    # Polygon.io
pip install ib_insync             # Interactive Brokers
```

---

## Recommended Future Extensions

| Extension | Where to add |
|---|---|
| Rate-limit token bucket | `providers/base.py` — `_rate_limit()` hook |
| Redis / Memcached cache | New `cache/redis_cache.py` implementing same interface as `ParquetCache` |
| Async batch downloads | `providers/yahoo_async.py` using `asyncio` + `aiohttp` |
| Intraday / tick data | Extend `BaseDataProvider.fetch_ohlcv(interval="1m"/"5m")` |
| Corporate actions | New `fetch_splits()` / `fetch_dividends()` on base |
| Data quality checks | `trading_data/validation.py` with gap detection, OHLC consistency |
| Metrics / observability | Prometheus counters in `DataManager._fetch_with_retry()` |
| Multi-provider fallback | `DataManager` tries `[YAHOO, POLYGON]` in order on failure |
| Custom symbol universe | Mount a JSON file via `SymbolMapper(custom_registry_path=...)` |
| Cloud cache (S3/GCS) | New `cache/s3_cache.py` with same `read()`/`write()` interface |
