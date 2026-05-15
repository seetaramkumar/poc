"""
stock_regime/quality_engine/opportunity_quality.py
====================================================
Evaluates the tradability and signal quality of each classified stock.

Five quality dimensions
-----------------------
1. liquidity_quality  : ADV adequacy, volume consistency
2. trend_quality      : EMA alignment cleanliness, higher-highs count
3. vol_health         : ATR in a healthy (not too high, not compressed) range
4. stability_quality  : regime age, smoothed confidence, oscillation penalty
5. tradability        : composite; penalises extended price, poor RS

All scores are in [0, 1]. The composite quality_score is a weighted sum.

Output
------
QualityScore dataclass — persisted to output/quality/
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class QualityScore:
    """Quality evaluation for one stock on one run date."""
    symbol:            str
    market:            str
    run_date:          date
    quality_score:     float   # composite [0, 1]
    liquidity_quality: float
    trend_quality:     float
    vol_health:        float
    stability_quality: float
    tradability:       float
    notes:             list[str]   # human-readable factor notes

    def to_dict(self) -> dict:
        return {
            "symbol":            self.symbol,
            "market":            self.market,
            "run_date":          str(self.run_date),
            "quality_score":     round(self.quality_score,     4),
            "liquidity_quality": round(self.liquidity_quality, 4),
            "trend_quality":     round(self.trend_quality,     4),
            "vol_health":        round(self.vol_health,        4),
            "stability_quality": round(self.stability_quality, 4),
            "tradability":       round(self.tradability,       4),
            "notes":             "; ".join(self.notes),
        }


class OpportunityQualityEngine:
    """
    Evaluates quality of stock regime signals.

    Parameters
    ----------
    weights : dict
        Weights for the five quality dimensions (should sum to 1.0).
    min_quality_score : float
        Stocks below this are flagged as low-quality in logs.
    """

    def __init__(
        self,
        weights:           Optional[dict] = None,
        min_quality_score: float          = 0.50,
    ) -> None:
        self.weights = weights or {
            "liquidity":   0.25,
            "trend_clean": 0.25,
            "vol_health":  0.20,
            "stability":   0.20,
            "tradability": 0.10,
        }
        self.min_quality_score = min_quality_score

    @classmethod
    def from_config(cls, config) -> "OpportunityQualityEngine":
        qcfg = getattr(config, "quality_engine", None)
        if qcfg is None:
            return cls()
        w = getattr(qcfg, "weights", None)
        weights = {
            "liquidity":   float(getattr(w, "liquidity",   0.25)),
            "trend_clean": float(getattr(w, "trend_clean", 0.25)),
            "vol_health":  float(getattr(w, "vol_health",  0.20)),
            "stability":   float(getattr(w, "stability",   0.20)),
            "tradability": float(getattr(w, "tradability", 0.10)),
        } if w else None
        return cls(
            weights=weights,
            min_quality_score=float(getattr(qcfg, "min_quality_score", 0.50)),
        )

    def evaluate_batch(
        self,
        stable_results: list,   # list[StableRegimeResult]
        stock_data:     dict[str, pd.DataFrame],
        run_date:       Optional[date] = None,
    ) -> list[QualityScore]:
        """
        Evaluate quality for all stable results.

        Parameters
        ----------
        stable_results :
            Output of RegimeStabiliser.apply().
        stock_data :
            Raw OHLCV DataFrames (post-filter) keyed by symbol.
        run_date :
            Date label for persistence.
        """
        run_date = run_date or date.today()
        scores   = []
        low_quality_count = 0

        for r in stable_results:
            if not r.is_valid():
                continue
            df    = stock_data.get(r.symbol)
            score = self._evaluate_one(r, df, run_date)
            scores.append(score)
            if score.quality_score < self.min_quality_score:
                low_quality_count += 1
                logger.debug(
                    "LOW_QUALITY %s: %.2f  [liq=%.2f trend=%.2f vol=%.2f stab=%.2f]",
                    r.symbol, score.quality_score,
                    score.liquidity_quality, score.trend_quality,
                    score.vol_health, score.stability_quality,
                )

        logger.info(
            "QualityEngine: %d evaluated | %d below min_quality=%.2f",
            len(scores), low_quality_count, self.min_quality_score,
        )
        return scores

    def persist(
        self,
        scores:     list[QualityScore],
        output_dir: str | Path,
        universe:   str = "UNKNOWN",
    ) -> Optional[Path]:
        if not scores:
            return None
        out  = Path(output_dir) / "quality" / str(scores[0].run_date)
        out.mkdir(parents=True, exist_ok=True)
        path = out / f"{universe.lower()}_quality_scores.parquet"
        pd.DataFrame([s.to_dict() for s in scores]).to_parquet(
            path, engine="pyarrow", compression="snappy", index=False,
        )
        logger.info("Quality scores [%s]: %d rows → '%s'.", universe, len(scores), path)
        return path

    # ──────────────────────────────────────────────────────────────
    #  Per-symbol evaluation
    # ──────────────────────────────────────────────────────────────

    def _evaluate_one(self, r, df: Optional[pd.DataFrame], run_date: date) -> QualityScore:
        notes: list[str] = []
        snap  = r.indicators
        ds    = r.dimensional_scores

        # ── 1. Liquidity quality ─────────────────────────────────────
        liq = self._liquidity_quality(snap, df, notes)

        # ── 2. Trend cleanliness ─────────────────────────────────────
        trend_q = self._trend_quality(snap, ds, notes)

        # ── 3. Volatility health ─────────────────────────────────────
        vol_h = self._vol_health(snap, notes)

        # ── 4. Stability quality ─────────────────────────────────────
        stab = self._stability_quality(r, notes)

        # ── 5. Tradability ───────────────────────────────────────────
        trad = self._tradability(snap, r, notes)

        composite = (
            self.weights["liquidity"]   * liq    +
            self.weights["trend_clean"] * trend_q +
            self.weights["vol_health"]  * vol_h  +
            self.weights["stability"]   * stab   +
            self.weights["tradability"] * trad
        )
        composite = round(min(max(composite, 0.0), 1.0), 4)

        return QualityScore(
            symbol            = r.symbol,
            market            = r.market,
            run_date          = run_date,
            quality_score     = composite,
            liquidity_quality = round(liq,    4),
            trend_quality     = round(trend_q, 4),
            vol_health        = round(vol_h,  4),
            stability_quality = round(stab,   4),
            tradability       = round(trad,   4),
            notes             = notes,
        )

    def _liquidity_quality(self, snap, df, notes: list[str]) -> float:
        score = 0.5  # default when no data
        if snap.volume is not None and snap.volume_ma is not None and snap.volume_ma > 0:
            ratio = snap.volume / snap.volume_ma
            # High volume relative to average = good liquidity signal
            score = min(ratio / 2.0, 1.0)
        if df is not None and len(df) >= 20:
            # Penalise high zero-volume day ratio
            zero_ratio = (df["volume"].tail(20) == 0).mean()
            if zero_ratio > 0.10:
                score *= (1 - zero_ratio)
                notes.append(f"zero_vol_ratio={zero_ratio:.2f}")
        return round(score, 4)

    def _trend_quality(self, snap, ds, notes: list[str]) -> float:
        score = ds.trend   # start from continuous trend score
        # Bonus for higher-highs structure
        if snap.higher_highs_count is not None:
            bonus = min(snap.higher_highs_count / 20.0, 0.20)
            score = min(score + bonus, 1.0)
        # Penalty when EMA too extended (overextended trends are lower quality)
        if snap.ema_distance_pct is not None and snap.ema_distance_pct > 0.20:
            score = max(score - 0.15, 0.0)
            notes.append(f"ema_extended={snap.ema_distance_pct:.2%}")
        return round(score, 4)

    def _vol_health(self, snap, notes: list[str]) -> float:
        """Healthy volatility = ATR/ATR_MA in [0.70, 1.30]. Too high or too low = lower quality."""
        if snap.atr is None or snap.atr_ma is None or snap.atr_ma <= 0:
            return 0.5
        ratio = snap.atr / snap.atr_ma
        # Peak quality at ratio=1.0; decays symmetrically
        deviation = abs(ratio - 1.0)
        score     = max(1.0 - deviation / 0.60, 0.0)
        if ratio > 1.50:
            notes.append(f"atr_elevated={ratio:.2f}")
        elif ratio < 0.60:
            notes.append(f"atr_compressed={ratio:.2f}")
        return round(score, 4)

    def _stability_quality(self, r, notes: list[str]) -> float:
        score = r.smoothed_confidence
        # Reward older regimes
        age_bonus = min(r.stable_regime_age / 20.0, 0.15)
        score     = min(score + age_bonus, 1.0)
        # Penalise oscillation
        if r.oscillation_detected:
            score = max(score - 0.20, 0.0)
            notes.append("oscillation_detected")
        # UNCERTAIN = 0 quality
        if r.stable_regime.value == "UNCERTAIN":
            score = 0.0
            notes.append("uncertain_regime")
        return round(score, 4)

    def _tradability(self, snap, r, notes: list[str]) -> float:
        score = 0.70  # base
        # Penalise if RS is weakening (would be selling into weakness)
        if hasattr(r.signals, "rs_weakening") and r.signals.rs_weakening:
            score -= 0.20
            notes.append("rs_weakening")
        # Reward if RS is improving
        if hasattr(r.signals, "rs_improving") and r.signals.rs_improving:
            score = min(score + 0.15, 1.0)
        # Penalise very high price extension
        if snap.ema_distance_pct is not None and snap.ema_distance_pct > 0.25:
            score = max(score - 0.15, 0.0)
        return round(min(max(score, 0.0), 1.0), 4)