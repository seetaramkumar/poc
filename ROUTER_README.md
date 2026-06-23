# Strategy Router — Routing Rules Reference

## Files Changed

| File | Action |
|---|---|
| `stock_regime/strategy_router/router.py` | **Replace** |
| `stock_regime/config/config_router_additions.yaml` | **Merge** into `config.yaml` |
| `tests/test_strategy_router.py` | **New file** |

---

## Routing table

### Stock regime → Strategy (first-match rules)

| Stock Regime | Strategy | Blocked when |
|---|---|---|
| `TREND_UP` | `TREND_FOLLOWING` | Never (posture adjusts instead) |
| `MOMENTUM` | `MOMENTUM` | `EXTREME_DOWN` breadth only |
| `RANGE` | `MEAN_REVERSION` | Market is `VOLATILE` |
| `BREAKOUT_SETUP` | `BREAKOUT` | Market is `BEARISH_TREND` or `VOLATILE`; sector is `WEAKENING`/`WEAK`; breadth is `EXTREME_DOWN` |
| `TREND_DOWN` | `NO_TRADE` | Always |
| `VOLATILE` | `NO_TRADE` | Always |
| `QUIET` | `NO_TRADE` | Always |
| `UNCERTAIN` | `NO_TRADE` | Always |

---

## Per-strategy quality gates (by market regime)

Each strategy has its own min_quality and min_confidence thresholds.
Mean-reversion is designed for range conditions — it has the lowest gate.
Breakout requires the highest quality (breakouts must be clean setups).

```
TREND_FOLLOWING gates:
  BULLISH_TREND  quality≥0.40  conf≥0.50
  SIDEWAYS       quality≥0.45  conf≥0.52
  BEARISH_TREND  quality≥0.60  conf≥0.62   ← strict: RS leaders only
  VOLATILE       quality≥0.65  conf≥0.65   ← very strict

MOMENTUM gates:
  BULLISH_TREND  quality≥0.42  conf≥0.52
  BEARISH_TREND  quality≥0.62  conf≥0.64

MEAN_REVERSION gates:
  BULLISH_TREND  quality≥0.38  conf≥0.48
  SIDEWAYS       quality≥0.36  conf≥0.46   ← lowest: preferred strategy here
  BEARISH_TREND  quality≥0.42  conf≥0.50   ← allowed (range persists in bear)
  VOLATILE       quality≥0.80  conf≥0.80   ← effectively blocked

BREAKOUT gates:
  BULLISH_TREND  quality≥0.50  conf≥0.55
  SIDEWAYS       quality≥0.55  conf≥0.58
  BEARISH_TREND  quality≥0.99  conf≥0.99   ← blocked
  VOLATILE       quality≥0.99  conf≥0.99   ← blocked
```

---

## Risk posture per strategy × market

| Strategy | Market | Posture | Why |
|---|---|---|---|
| MEAN_REVERSION | SIDEWAYS | **NORMAL** | Preferred — full sizing |
| TREND_FOLLOWING | SIDEWAYS | DEFENSIVE | Against market character |
| TREND_FOLLOWING | BEARISH | DEFENSIVE | Only RS leaders pass |
| MEAN_REVERSION | BEARISH | DEFENSIVE | More uncertainty |
| Any | VOLATILE | CAPITAL_PRESERVATION | Elevated risk |

---

## PSM computation

```
PSM = base_psm
    × quality_mult      (0.30 → 1.0  based on quality vs min_quality_full)
    × breadth_mult      (0.75 → 1.25 based on breadth_score [0,1])
    × sector_mult       (0.65 → 1.15 based on 5-level sector state)
```

**Sector multipliers:**

| Sector State | Multiplier |
|---|---|
| LEADING | × 1.15 (+15%) |
| STRONG | × 1.07 (+7%) |
| NEUTRAL | × 1.00 (±0%) |
| WEAKENING | × 0.80 (−20%) |
| WEAK | × 0.65 (−35%) |

**Profile mapping:**

| PSM | Risk Profile |
|---|---|
| ≥ 1.15 | AGGRESSIVE |
| ≥ 0.80 | NORMAL |
| ≥ 0.45 | DEFENSIVE |
| ≥ 0.15 | CAPITAL_PRESERVATION |
| < 0.15 | OFF |

---

## Example output

```json
{
  "symbol": "BHEL.NS",
  "strategy": "BREAKOUT",
  "allowed": true,
  "risk_profile": "NORMAL",
  "position_size_multiplier": 0.94,
  "regime_context": "BREAKOUT_SETUP",
  "market_context": "BULLISH_TREND",
  "quality_score": 0.71,
  "breadth_state": "EXPANDING",
  "sector_state": "LEADING",
  "reason": [
    "BREAKOUT_SETUP: sector=LEADING breadth=EXPANDING market=BULLISH_TREND",
    "psm=0.94 [NORMAL] (base=1.00×q=0.86×b=1.09×s=1.15)"
  ]
}
```

```json
{
  "symbol": "YESBANK.NS",
  "strategy": "NO_TRADE",
  "allowed": false,
  "risk_profile": "OFF",
  "position_size_multiplier": 0.0,
  "regime_context": "TREND_UP",
  "market_context": "BEARISH_TREND",
  "quality_score": 0.42,
  "reason": [
    "quality=0.42 < 0.60 (TREND_FOLLOWING/BEARISH_TREND gate)"
  ]
}
```

---

## Diagnostics log example

```
───────────────────────────────────────────────────────
StrategyRouter: 487 routed | 312 allowed (64%) | market=BULLISH_TREND | breadth=EXPANDING
Strategy distribution:
  TREND_FOLLOWING     148 (30%)  ▓▓▓▓▓▓▓▓▓
  MEAN_REVERSION       89 (18%)  ▓▓▓▓▓
  MOMENTUM             51 (10%)  ▓▓▓
  BREAKOUT             24 (5%)   ▓
  NO_TRADE            175 (36%)  ▓▓▓▓▓▓▓▓▓▓▓
Top rejection reasons:
  regime_UNCERTAIN                       61 (13%)
  quality_TREND_FOLLOWING                48 (10%)
  regime_TREND_DOWN                      38 (8%)
  confidence_MOMENTUM                    19 (4%)
  regime_VOLATILE                         9 (2%)
Regime → strategy:
  BREAKOUT_SETUP   → BREAKOUT:24  NO_TRADE:8
  MOMENTUM         → MOMENTUM:51  NO_TRADE:6
  RANGE            → MEAN_REVERSION:89  NO_TRADE:4
  TREND_DOWN       → NO_TRADE:38
  TREND_UP         → TREND_FOLLOWING:148  NO_TRADE:48
  UNCERTAIN        → NO_TRADE:61
───────────────────────────────────────────────────────
```