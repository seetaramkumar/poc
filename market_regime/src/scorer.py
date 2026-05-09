"""
scorer.py — Weighted scoring for each market regime
====================================================
Instead of hard if/else branching, each regime gets a continuous
score in [0, 1] derived from the weighted sum of boolean signals.

This makes the engine:
  • transparent   — you can see how strongly each regime scored
  • extensible    — add new signals by adding entries to config.yaml
  • tunable       — change weights in config without touching code

Architecture note
-----------------
Scorer knows about scoring weights and how to combine signals.
It does NOT know about thresholds (that's SignalExtractor's job)
or about which regime wins (that's Classifier's job).
"""

from __future__ import annotations

from typing import Dict

from .config_loader import EngineConfig
from .models import MarketRegime, RegimeSignals


class RegimeScorer:
    """
    Computes a weighted confidence score for each candidate regime.

    Parameters
    ----------
    config : EngineConfig
    """

    def __init__(self, config: EngineConfig) -> None:
        self.cfg = config

    def score_all(self, signals: RegimeSignals) -> Dict[MarketRegime, float]:
        """
        Return a dict mapping every MarketRegime to its score in [0, 1].

        Each regime's score is the dot-product of its configured weights
        and the corresponding boolean signal values.

        Parameters
        ----------
        signals : RegimeSignals

        Returns
        -------
        dict[MarketRegime, float]
        """
        sig = signals   # alias for brevity
        sc  = self.cfg.scoring  # shorthand

        scores: Dict[MarketRegime, float] = {}

        # ── BULLISH_TREND ──────────────────────────────────────────
        scores[MarketRegime.BULLISH_TREND] = (
            sc.bullish.price_above_ema200 * int(sig.price_above_ema200) +
            sc.bullish.ema20_above_ema50  * int(sig.ema20_above_ema50)  +
            sc.bullish.adx_strong         * int(sig.adx_strong)         +
            sc.bullish.volume_confirms    * int(sig.volume_confirms)
        )

        # ── BEARISH_TREND ──────────────────────────────────────────
        scores[MarketRegime.BEARISH_TREND] = (
            sc.bearish.price_below_ema200 * int(sig.price_below_ema200) +
            sc.bearish.ema20_below_ema50  * int(sig.ema20_below_ema50)  +
            sc.bearish.adx_strong         * int(sig.adx_strong)         +
            sc.bearish.volume_confirms    * int(sig.volume_confirms)
        )

        # ── SIDEWAYS ───────────────────────────────────────────────
        scores[MarketRegime.SIDEWAYS] = (
            sc.sideways.adx_weak   * int(sig.adx_weak)   +
            sc.sideways.ema20_flat * int(sig.ema20_flat) +
            sc.sideways.ema50_flat * int(sig.ema50_flat)
        )

        # ── VOLATILE ───────────────────────────────────────────────
        scores[MarketRegime.VOLATILE] = (
            sc.volatile.atr_high   * int(sig.atr_high)   +
            sc.volatile.adx_strong * int(sig.adx_strong)
        )

        # ── QUIET ─────────────────────────────────────────────────
        scores[MarketRegime.QUIET] = (
            sc.quiet.atr_low  * int(sig.atr_low)  +
            sc.quiet.adx_weak * int(sig.adx_weak)
        )

        return scores
