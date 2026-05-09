# Market Regime Engine

A modular, scoring-based Python engine that classifies daily stock market conditions into one of five regimes using technical indicators. Built for production use, backtesting pipelines, and easy extensibility.

---

## Table of Contents

- [Overview](#overview)
- [Regimes](#regimes)
- [Project Structure](#project-structure)
- [Module Responsibilities](#module-responsibilities)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Output Format](#output-format)
- [Architecture Deep Dive](#architecture-deep-dive)
- [Configuration Reference](#configuration-reference)
- [Indicators Used](#indicators-used)
- [Regime Classification Logic](#regime-classification-logic)
- [Scoring System](#scoring-system)
- [Extending the Engine](#extending-the-engine)
- [Running the Demo](#running-the-demo)
- [Running Tests](#running-tests)
- [Using Real Market Data](#using-real-market-data)
- [Design Decisions](#design-decisions)

---

## Overview

The Market Regime Engine takes **daily OHLCV data** (e.g. NIFTY 50, S&P 500) and produces a structured JSON output telling you which market regime is currently active and how confident the engine is in that assessment.

Instead of brittle if/else branching, each regime is given a **continuous weighted score** derived from boolean signals. The highest-scoring regime wins, making the engine naturally express partial matches and degrade gracefully when the market is ambiguous.

```
OHLCV DataFrame
      │
      ▼
IndicatorCalculator  →  IndicatorSnapshot  (EMA20/50/200, ADX, ATR, Volume MA)
      │
      ▼
SignalExtractor      →  RegimeSignals      (boolean flags, e.g. adx_strong=True)
      │
      ▼
RegimeScorer         →  scores dict        (weighted float per regime)
      │
      ▼
RegimeClassifier     →  RegimeResult       (regime, confidence, signals, scores)
```

---

## Regimes

| Regime | Description |
|---|---|
| `BULLISH_TREND` | Price above EMA200, EMA20 > EMA50, strong ADX |
| `BEARISH_TREND` | Price below EMA200, EMA20 < EMA50, strong ADX |
| `SIDEWAYS` | ADX low, EMAs flat and crossing |
| `VOLATILE` | ATR significantly above its moving average |
| `QUIET` | ATR significantly below its moving average |
| `UNCERTAIN` | No regime scored above the minimum confidence threshold |

---

## Project Structure

```
market_regime_engine/
├── config/
│   └── config.yaml          # All numeric thresholds and scoring weights
├── src/
│   ├── __init__.py          # Public API: exposes MarketRegimeEngine
│   ├── models.py            # Typed data contracts (dataclasses + Enum)
│   ├── config_loader.py     # Reads config.yaml into dot-accessible Python object
│   ├── indicators.py        # Computes EMA, ADX, ATR, Volume MA via pandas-ta
│   ├── signals.py           # Converts indicator values → boolean flags
│   ├── scorer.py            # Weighted dot-product scoring per regime
│   ├── classifier.py        # Picks winning regime with tiebreaking
│   └── engine.py            # Orchestrator — wires all layers together
├── tests/
│   └── test_engine.py       # 20 pytest unit and integration tests
├── main.py                  # Runnable demo with synthetic NIFTY data
├── requirements.txt
└── README.md
```

---

## Module Responsibilities

### `config/config.yaml`
The single source of truth for every numeric threshold and scoring weight. No magic numbers anywhere in the code. Edit this file to retune the engine without touching Python.

### `src/models.py`
Defines all structured data types used across the pipeline:
- `MarketRegime` — Enum of the five regimes plus `UNCERTAIN`
- `IndicatorSnapshot` — Numeric indicator values for one bar
- `RegimeSignals` — Boolean flags derived from the snapshot
- `RegimeResult` — Final output (regime, confidence, signals, scores)

### `src/config_loader.py`
Reads `config.yaml` and exposes it as a dot-accessible `EngineConfig` object. Respects the `REGIME_CONFIG` environment variable for custom config paths.

### `src/indicators.py`
Takes a raw OHLCV DataFrame and computes all technical indicators using `pandas-ta`. Returns an `IndicatorSnapshot` for any requested row. This is the only module that touches `pandas-ta`.

### `src/signals.py`
Contains all threshold comparisons. Converts numeric indicator values into named boolean signals (`adx_strong`, `atr_high`, `ema20_flat`, etc.). Pure functions — no side effects.

### `src/scorer.py`
Computes a weighted score in `[0, 1]` for each regime by taking the dot-product of configured weights and boolean signal values. The scores make it transparent *how strongly* each regime matches the current data.

### `src/classifier.py`
Selects the winning regime from the scored dict. Applies a tiebreaking priority order and rejects the winner as `UNCERTAIN` if its score falls below the configured minimum confidence.

### `src/engine.py`
The only class callers need to import. Constructs and wires together all internal modules. Exposes three methods: `analyze()`, `analyze_to_json()`, and `analyze_rolling()`.

### `tests/test_engine.py`
Twenty pytest tests covering every layer independently (unit tests) and the full pipeline end-to-end (integration tests).

---

## Installation

**Requirements:** Python 3.10+

```bash
# Clone or download the project
cd market_regime_engine

# Install dependencies
pip install -r requirements.txt
```

`requirements.txt` contents:
```
pandas>=2.0.0
pandas-ta>=0.3.14b
numpy>=1.24.0
pyyaml>=6.0
pytest>=7.0.0
```

---

## Quick Start

```python
import pandas as pd
from src import MarketRegimeEngine

# Load your OHLCV data (columns must be: open, high, low, close, volume)
df = pd.read_csv("nifty_daily.csv", parse_dates=["date"], index_col="date")

# Instantiate the engine (loads config automatically)
engine = MarketRegimeEngine()

# Classify the most recent bar
result = engine.analyze(df)

# Print structured JSON output
import json
print(json.dumps(result.to_dict(), indent=2))
```

---

## Output Format

```json
{
  "regime": "BULLISH_TREND",
  "confidence": 0.8,
  "signals": {
    "price_above_ema200": true,
    "price_below_ema200": false,
    "ema20_above_ema50": true,
    "ema20_below_ema50": false,
    "adx_strong": true,
    "adx_weak": false,
    "ema20_flat": false,
    "ema50_flat": false,
    "atr_high": false,
    "atr_low": false,
    "volume_confirms": false
  },
  "scores": {
    "BULLISH_TREND": 0.8,
    "BEARISH_TREND": 0.0,
    "SIDEWAYS": 0.0,
    "VOLATILE": 0.25,
    "QUIET": 0.0
  }
}
```

| Field | Type | Description |
|---|---|---|
| `regime` | string | Winning market regime |
| `confidence` | float [0–1] | Score of the winning regime |
| `signals` | object | Boolean flags that drove the decision |
| `scores` | object | Raw weighted score for every regime |

---

## Architecture Deep Dive

### Data Flow

```
pd.DataFrame (OHLCV)
        │
        ▼
┌──────────────────────┐
│  IndicatorCalculator │  pandas-ta: EMA, ADX, ATR, Volume MA
└──────────┬───────────┘
           │ IndicatorSnapshot
           ▼
┌──────────────────────┐
│   SignalExtractor    │  threshold comparisons → True/False flags
└──────────┬───────────┘
           │ RegimeSignals
           ▼
┌──────────────────────┐
│    RegimeScorer      │  weight × signal for each regime
└──────────┬───────────┘
           │ dict[MarketRegime, float]
           ▼
┌──────────────────────┐
│  RegimeClassifier    │  argmax + tiebreaking + min confidence
└──────────┬───────────┘
           │ RegimeResult
           ▼
     JSON / Python object
```

### Layer Isolation

Each layer has strict boundaries about what it is allowed to know:

| Layer | Allowed to know | Not allowed to know |
|---|---|---|
| `indicators.py` | Config periods, pandas-ta | Thresholds, weights |
| `signals.py` | Config thresholds | Weights, regime names |
| `scorer.py` | Config weights, signal names | Thresholds, indicator values |
| `classifier.py` | Config min confidence | Weights, thresholds, indicator values |
| `engine.py` | All modules | Internal implementation details |

This means you can swap any layer independently. For example, replace `signals.py` with an ML-based signal extractor and nothing else changes.

### Tiebreaking Priority

When two regimes score equally, the classifier uses this fixed priority:

```
VOLATILE > BEARISH_TREND > BULLISH_TREND > SIDEWAYS > QUIET
```

Extreme regimes take precedence so they are not silenced by overlapping trend signals.

---

## Configuration Reference

All values live in `config/config.yaml`. Use the `REGIME_CONFIG` environment variable to point to a different file:

```bash
REGIME_CONFIG=/path/to/my_config.yaml python main.py
```

### `indicators` section

```yaml
indicators:
  ema_periods:
    fast: 20      # EMA-20 period
    mid:  50      # EMA-50 period
    slow: 200     # EMA-200 period
  adx_period: 14  # ADX/DMI look-back window
  atr_period:  14 # ATR look-back window
  volume_ma_period: 20  # Rolling volume average window
```

### `thresholds` section

```yaml
thresholds:
  adx_strong_trend:   25.0  # ADX above this → trend is strong
  adx_weak_trend:     18.0  # ADX below this → market is directionless
  atr_ma_period:       20   # Periods for ATR's own moving average
  atr_volatile_ratio:  1.30 # ATR / ATR_MA > this → VOLATILE signal
  atr_quiet_ratio:     0.70 # ATR / ATR_MA < this → QUIET signal
  ema_slope_window:    5    # Bars to look back when measuring EMA slope
  ema_flat_threshold:  0.001 # |slope| < this → EMA is considered flat
  volume_surge_ratio:  1.50 # volume / vol_MA > this → volume confirms move
```

### `scoring` section

Weights per regime must sum to `1.0`. Each true signal contributes `weight × 1.0`.

```yaml
scoring:
  bullish:
    price_above_ema200: 0.30
    ema20_above_ema50:  0.25
    adx_strong:         0.25
    volume_confirms:    0.20

  bearish:
    price_below_ema200: 0.30
    ema20_below_ema50:  0.25
    adx_strong:         0.25
    volume_confirms:    0.20

  sideways:
    adx_weak:           0.40
    ema20_flat:         0.30
    ema50_flat:         0.30

  volatile:
    atr_high:           0.60
    adx_strong:         0.40

  quiet:
    atr_low:            0.70
    adx_weak:           0.30

  min_confidence: 0.50  # Below this → UNCERTAIN
```

---

## Indicators Used

| Indicator | Purpose | Supplied by |
|---|---|---|
| EMA 20 | Short-term trend / momentum | `pandas-ta` |
| EMA 50 | Medium-term trend | `pandas-ta` |
| EMA 200 | Macro trend anchor | `pandas-ta` |
| ADX (14) | Trend strength (0–100, direction-agnostic) | `pandas-ta` |
| ATR (14) | Absolute daily volatility in price points | `pandas-ta` |
| ATR MA (20) | Smoothed ATR baseline for ratio comparison | `pandas` rolling mean |
| Volume MA (20) | Average volume baseline | `pandas` rolling mean |
| EMA slope | % change of EMA over N bars (flatness test) | `pandas` pct_change |

---

## Regime Classification Logic

### BULLISH_TREND
The market is in a confirmed uptrend with momentum.

| Condition | Weight |
|---|---|
| `close > EMA200` | 0.30 |
| `EMA20 > EMA50` | 0.25 |
| `ADX > 25` | 0.25 |
| `Volume > Volume_MA × 1.5` | 0.20 |

### BEARISH_TREND
The market is in a confirmed downtrend with momentum.

| Condition | Weight |
|---|---|
| `close < EMA200` | 0.30 |
| `EMA20 < EMA50` | 0.25 |
| `ADX > 25` | 0.25 |
| `Volume > Volume_MA × 1.5` | 0.20 |

### SIDEWAYS
No clear direction; market is consolidating.

| Condition | Weight |
|---|---|
| `ADX < 18` | 0.40 |
| `EMA20 slope ≈ 0` | 0.30 |
| `EMA50 slope ≈ 0` | 0.30 |

### VOLATILE
Daily range is significantly elevated relative to recent history.

| Condition | Weight |
|---|---|
| `ATR / ATR_MA > 1.30` | 0.60 |
| `ADX > 25` | 0.40 |

### QUIET
Daily range is significantly compressed relative to recent history.

| Condition | Weight |
|---|---|
| `ATR / ATR_MA < 0.70` | 0.70 |
| `ADX < 18` | 0.30 |

---

## Scoring System

The scoring system replaces traditional if/else branching with a continuous weighted sum, giving several advantages:

**Transparency** — The `scores` dict in the output shows how strongly each regime matched, not just the winner.

**Graceful degradation** — A market that is 70% bullish and 30% sideways will score appropriately rather than forcing a binary choice.

**Tunability** — Change weights in `config.yaml` without touching any Python code.

**UNCERTAIN fallback** — If no regime clears the `min_confidence` threshold (default 0.50), the result is `UNCERTAIN` rather than a forced low-confidence label.

### Score calculation example

Suppose the signals for the latest bar are:

```
price_above_ema200 = True
ema20_above_ema50  = True
adx_strong         = True
volume_confirms    = False
```

Bullish score = `0.30 × 1 + 0.25 × 1 + 0.25 × 1 + 0.20 × 0` = **0.80**

The engine returns `BULLISH_TREND` with `confidence: 0.80`.

---

## Extending the Engine

### Adding a new signal

1. Add a boolean field to `RegimeSignals` in `models.py`:
   ```python
   rsi_oversold: bool = False
   ```

2. Compute and set it in `signals.py`:
   ```python
   if snap.rsi is not None:
       sig.rsi_oversold = snap.rsi < 30
   ```

3. Compute the new indicator in `indicators.py`:
   ```python
   df["rsi"] = ta.rsi(df["close"], length=14)
   ```

4. Add it to `IndicatorSnapshot` in `models.py`:
   ```python
   rsi: Optional[float] = None
   ```

5. Add a weight to the relevant regime in `config.yaml`:
   ```yaml
   bullish:
     rsi_oversold: 0.10
     # adjust other weights to keep sum = 1.0
   ```

6. Reference it in `scorer.py`:
   ```python
   scores[MarketRegime.BULLISH_TREND] = (
       ...
       + sc.bullish.rsi_oversold * int(sig.rsi_oversold)
   )
   ```

### Adding a new regime

1. Add it to the `MarketRegime` enum in `models.py`
2. Add a weights block for it in `config.yaml`
3. Add the scoring line in `scorer.py`
4. Add it to `_TIEBREAK_PRIORITY` in `classifier.py`

### Custom config path

```python
from pathlib import Path
from src import MarketRegimeEngine

engine = MarketRegimeEngine(config_path=Path("my_configs/aggressive.yaml"))
```

Or via environment variable:

```bash
REGIME_CONFIG=my_configs/aggressive.yaml python main.py
```

---

## Running the Demo

```bash
python main.py
```

The demo generates 600 bars of synthetic NIFTY-like data across five distinct market phases and demonstrates single-bar classification, per-regime spot checks, and a rolling classification summary.

Expected output excerpt:

```
✓ Generated 600 bars of synthetic NIFTY data
  Date range : 2022-09-14 → 2024-12-31

Latest-bar Classification
{
  "regime": "BEARISH_TREND",
  "confidence": 0.55,
  "signals": { ... },
  "scores": { ... }
}

Spot-check: One Bar per Synthetic Regime Region
  Bar 250  (Strong bull)   | UNCERTAIN       conf=0.30 | ADX=18.7
  Bar 350  (Volatile)      | SIDEWAYS        conf=0.70 | ADX=12.8
  Bar 450  (Bearish)       | BEARISH_TREND   conf=0.80 | ADX=26.1

Rolling Classification — Regime Distribution
  BULLISH_TREND    ████████████   191 bars (31.8%)
  UNCERTAIN        ███████████    178 bars (29.7%)
  BEARISH_TREND    █████████      139 bars (23.2%)
  SIDEWAYS         █████           89 bars (14.8%)
  VOLATILE                          3 bars ( 0.5%)
```

---

## Running Tests

```bash
pytest tests/ -v
```

The test suite covers:

- `IndicatorSnapshot` completeness checks
- `SignalExtractor` for all five signal types (bullish, bearish, flat, volatile, quiet)
- `RegimeScorer` boundary conditions (perfect score, all-false, weight bounds)
- `RegimeClassifier` winner selection and UNCERTAIN fallback
- `MarketRegimeEngine` end-to-end: JSON validity, confidence bounds, rolling length, error handling

```
======================== 20 passed in 6.88s ========================
```

---

## Using Real Market Data

Your CSV must contain these columns (case-insensitive): `date`, `open`, `high`, `low`, `close`, `volume`.

```python
import pandas as pd
import json
from src import MarketRegimeEngine

df = pd.read_csv(
    "nifty_daily.csv",
    parse_dates=["date"],
    index_col="date",
)

engine = MarketRegimeEngine()

# Latest bar
result = engine.analyze(df)
print(json.dumps(result.to_dict(), indent=2))

# All bars (for backtesting / charting)
all_results = engine.analyze_rolling(df)
regimes = [r.regime.value for r in all_results]
```

> **Note:** EMA-200 requires at least 200 bars of history. Bars before that threshold will return `UNCERTAIN`. For a full classification from day one, supply at least 220 bars (200 + warmup for ADX/ATR).

---

## Design Decisions

**Scoring over if/else** — A single failing condition should not disqualify a regime entirely. Weighted scoring means a strong bullish bar without a volume spike still gets `0.80` instead of `0`.

**No hardcoded numbers** — Every numeric threshold lives in `config.yaml`. The code is readable and self-documenting; the config is tunable without a code review.

**Strict layer isolation** — Each module has one job and is ignorant of everything else. This makes individual layers unit-testable in isolation and swappable without cascading changes.

**`UNCERTAIN` as a first-class regime** — It is better to say "I don't know" than to force a low-confidence label. `UNCERTAIN` signals to downstream consumers that they should not act on the regime.

**`pandas-ta` for indicators** — A well-maintained, vectorised library that computes all required indicators in a single pass over the DataFrame with no manual loop overhead.

**Dataclasses over dicts** — Typed dataclasses prevent silent key typos, provide IDE autocompletion, and make the data contract explicit and self-documenting.
