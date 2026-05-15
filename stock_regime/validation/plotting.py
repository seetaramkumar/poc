"""
stock_regime/validation/plotting.py
=====================================
Reusable plotting utilities for validating regime classifications.

All functions return matplotlib Figure objects so they can be used
in notebooks or saved to disk — no display calls inside this module.

Usage (notebook):
    from stock_regime.validation.plotting import RegimePlotter
    plotter = RegimePlotter()
    fig = plotter.plot_regime_chart("INFY.NS", df, stable_result, quality_score)
    fig.savefig("output/validation/INFY_regime.png")
"""

from __future__ import annotations

import logging
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)

# Colour palette per regime
_REGIME_COLOURS = {
    "TREND_UP":       "#2ecc71",
    "TREND_DOWN":     "#e74c3c",
    "RANGE":          "#95a5a6",
    "MOMENTUM":       "#3498db",
    "BREAKOUT_SETUP": "#f39c12",
    "VOLATILE":       "#9b59b6",
    "QUIET":          "#1abc9c",
    "UNCERTAIN":      "#bdc3c7",
}


class RegimePlotter:
    """
    Produces regime validation charts from classification outputs.

    Parameters
    ----------
    figsize : tuple
        Default figure size (width, height) in inches.
    style : str
        Matplotlib style. "seaborn-v0_8-darkgrid" works on most installations.
    """

    def __init__(
        self,
        figsize: tuple = (16, 10),
        style:   str   = "seaborn-v0_8-darkgrid",
    ) -> None:
        self.figsize = figsize
        self.style   = style

    def plot_regime_chart(
        self,
        symbol:         str,
        ohlcv_df:       pd.DataFrame,
        stable_result,                  # StableRegimeResult
        quality_score=None,             # Optional[QualityScore]
        lookback_bars:  int = 120,
    ):
        """
        Four-panel validation chart for one symbol:
        Panel 1 — Price + EMAs + regime background shading
        Panel 2 — ADX
        Panel 3 — Continuous scores (trend, momentum)
        Panel 4 — Volume vs Volume MA

        Parameters
        ----------
        symbol : str
        ohlcv_df : pd.DataFrame
            Raw OHLCV with DatetimeIndex.
        stable_result : StableRegimeResult
        quality_score : QualityScore, optional
        lookback_bars : int
            How many bars to display.

        Returns
        -------
        matplotlib.figure.Figure
        """
        try:
            import matplotlib
            matplotlib.use("Agg")   # non-interactive backend; safe for scripts
            import matplotlib.pyplot as plt
            import matplotlib.patches as mpatches
        except ImportError:
            logger.error("matplotlib not installed. Run: pip install matplotlib")
            raise

        with plt.style.context(self.style):
            fig, axes = plt.subplots(4, 1, figsize=self.figsize,
                                     gridspec_kw={"height_ratios": [3, 1, 1, 1]},
                                     sharex=True)

        df    = ohlcv_df.tail(lookback_bars).copy()
        snap  = stable_result.indicators
        ds    = stable_result.dimensional_scores
        r_col = _REGIME_COLOURS.get(stable_result.stable_regime.value, "#bdc3c7")

        # ── Panel 1: Price + EMAs ──────────────────────────────────
        ax1 = axes[0]
        ax1.plot(df.index, df["close"], color="black", linewidth=1.2, label="Close")
        if snap.ema20  is not None: ax1.axhline(snap.ema20,  color="#3498db", linestyle="--", linewidth=0.8, label="EMA20")
        if snap.ema50  is not None: ax1.axhline(snap.ema50,  color="#e67e22", linestyle="--", linewidth=0.8, label="EMA50")
        if snap.ema200 is not None: ax1.axhline(snap.ema200, color="#e74c3c", linestyle="--", linewidth=1.2, label="EMA200")

        # Regime background shading on last bar
        ax1.axvspan(df.index[-3], df.index[-1], alpha=0.15, color=r_col)

        qual_str = f"  |  Quality: {quality_score.quality_score:.2f}" if quality_score else ""
        title = (
            f"{symbol}  —  Stable: {stable_result.stable_regime.value}"
            f"  (conf={stable_result.smoothed_confidence:.2f}  age={stable_result.stable_regime_age}d)"
            f"{qual_str}"
        )
        ax1.set_title(title, fontsize=11, fontweight="bold")
        ax1.legend(loc="upper left", fontsize=8)
        ax1.set_ylabel("Price")

        # ── Panel 2: ADX ──────────────────────────────────────────
        ax2 = axes[1]
        if snap.adx is not None:
            ax2.axhline(snap.adx, color="#8e44ad", linewidth=1.5, label=f"ADX={snap.adx:.1f}")
            ax2.axhline(25, color="grey", linestyle=":", linewidth=0.8)
            ax2.axhline(18, color="grey", linestyle=":", linewidth=0.8)
        ax2.set_ylabel("ADX")
        ax2.set_ylim(0, 60)
        ax2.legend(loc="upper left", fontsize=8)

        # ── Panel 3: Continuous scores ─────────────────────────────
        ax3 = axes[2]
        cs = ds.continuous
        scores_to_show = {
            "Trend":    ds.trend,
            "Momentum": ds.momentum,
        }
        if cs is not None:
            scores_to_show["ADX_c"]  = cs.adx_score
            scores_to_show["RS_c"]   = cs.rs_score
            scores_to_show["ROC_c"]  = cs.roc_score

        colours = ["#2ecc71", "#3498db", "#e74c3c", "#f39c12", "#9b59b6"]
        for (label, val), col in zip(scores_to_show.items(), colours):
            ax3.axhline(val, color=col, linewidth=1.2, label=f"{label}={val:.2f}")
        ax3.set_ylim(0, 1.05)
        ax3.set_ylabel("Scores")
        ax3.legend(loc="upper left", fontsize=7, ncol=3)
        ax3.axhline(0.5, color="grey", linestyle=":", linewidth=0.8)

        # ── Panel 4: Volume ────────────────────────────────────────
        ax4 = axes[3]
        ax4.bar(df.index, df["volume"], color="#95a5a6", alpha=0.7, label="Volume")
        if snap.volume_ma is not None:
            ax4.axhline(snap.volume_ma, color="#e74c3c", linewidth=1.0,
                        linestyle="--", label=f"Vol MA={snap.volume_ma:,.0f}")
        ax4.set_ylabel("Volume")
        ax4.legend(loc="upper left", fontsize=8)

        fig.tight_layout()
        return fig

    def plot_score_distribution(
        self,
        score_df: pd.DataFrame,
        score_col: str = "trend",
        universe:  str = "UNIVERSE",
    ):
        """
        Histogram of a score column across all stocks.

        Parameters
        ----------
        score_df : pd.DataFrame
            Must contain ``score_col`` column.
        score_col : str
            Column to plot.
        universe : str
            Label for the title.

        Returns
        -------
        matplotlib.figure.Figure
        """
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            raise

        with plt.style.context(self.style):
            fig, ax = plt.subplots(figsize=(10, 5))

        vals = score_df[score_col].dropna()
        ax.hist(vals, bins=20, color="#3498db", edgecolor="white", alpha=0.85)
        ax.axvline(vals.mean(),   color="#e74c3c", linestyle="--",
                   linewidth=1.5, label=f"Mean={vals.mean():.3f}")
        ax.axvline(vals.median(), color="#2ecc71", linestyle="--",
                   linewidth=1.5, label=f"Median={vals.median():.3f}")
        ax.axvline(vals.quantile(0.90), color="#f39c12", linestyle=":",
                   linewidth=1.5, label=f"P90={vals.quantile(0.90):.3f}")
        ax.set_title(f"{universe} — {score_col} Score Distribution (n={len(vals)})",
                     fontsize=11, fontweight="bold")
        ax.set_xlabel(score_col)
        ax.set_ylabel("Count")
        ax.legend(fontsize=9)
        fig.tight_layout()
        return fig

    def plot_regime_distribution(
        self,
        stable_results: list,
        universe: str = "UNIVERSE",
    ):
        """Bar chart of regime distribution."""
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            raise

        counts: dict[str, int] = {}
        for r in stable_results:
            k = r.stable_regime.value if hasattr(r, "stable_regime") else r.stock_regime.value
            counts[k] = counts.get(k, 0) + 1

        regimes = list(counts.keys())
        values  = list(counts.values())
        colours = [_REGIME_COLOURS.get(r, "#bdc3c7") for r in regimes]

        with plt.style.context(self.style):
            fig, ax = plt.subplots(figsize=(12, 5))
        bars = ax.bar(regimes, values, color=colours, edgecolor="white")
        ax.bar_label(bars, fmt="%d", fontsize=9)
        ax.set_title(f"{universe} — Regime Distribution (n={sum(values)})",
                     fontsize=11, fontweight="bold")
        ax.set_ylabel("Count")
        ax.set_xticklabels(regimes, rotation=20, ha="right")
        fig.tight_layout()
        return fig

    def save_all(
        self,
        symbol:         str,
        ohlcv_df:       pd.DataFrame,
        stable_result,
        output_dir:     str,
        quality_score=None,
    ) -> list[str]:
        """Save regime chart for one symbol. Returns list of saved paths."""
        from pathlib import Path
        out = Path(output_dir) / "validation"
        out.mkdir(parents=True, exist_ok=True)

        saved = []
        try:
            fig  = self.plot_regime_chart(symbol, ohlcv_df, stable_result, quality_score)
            path = out / f"{symbol.replace('.', '_')}_regime.png"
            fig.savefig(path, dpi=120, bbox_inches="tight")
            fig.clf()
            import matplotlib.pyplot as plt
            plt.close(fig)
            saved.append(str(path))
        except Exception as exc:
            logger.warning("Could not save chart for %s: %s", symbol, exc)
        return saved