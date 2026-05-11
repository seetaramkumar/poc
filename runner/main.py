"""
runner/main.py
==============
Runnable entry point for the full Algo Trading Pipeline.

Demonstrates:
  1. Programmatic API usage (recommended for production)
  2. Output inspection per universe
  3. How to filter results for downstream use

Run with:
    cd <project_root>
    python runner/main.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from runner.pipeline import AlgoTradingPipeline


def main() -> None:
    print("=" * 65)
    print("  Algo Trading Platform — Full Pipeline")
    print("=" * 65)

    # ── Instantiate pipeline ──────────────────────────────────────────
    # Config is loaded from runner/config/pipeline.yaml automatically.
    pipeline = AlgoTradingPipeline()

    # ── Run all configured universes ──────────────────────────────────
    # For a quick test: run only one universe.
    # For production: call pipeline.run() with no arguments.
    output = pipeline.run(
        universes=["NIFTY500"],   # remove to run SP500 as well
        persist=True,
    )

    # ── Print market regime per universe ─────────────────────────────
    print(f"\n{'─' * 65}")
    print("  Market Regime Summary")
    print(f"{'─' * 65}")
    for universe, regime_dict in output.market_results.items():
        print(f"\n  {universe}")
        print(f"    Regime     : {regime_dict['regime']}")
        print(f"    Confidence : {regime_dict['confidence']:.2%}")

    # ── Print stock classification table ─────────────────────────────
    for universe, results in output.stock_results.items():
        print(f"\n{'─' * 65}")
        print(f"  {universe} — Stock Classifications ({len(results)} stocks)")
        print(f"{'─' * 65}")
        print(f"  {'Symbol':<12} {'Regime':<18} {'Conf':>6}  "
              f"{'Trend':>6}  {'Mom':>6}  {'Vol':>6}  {'RS':>7}")
        print("  " + "─" * 58)
        for r in sorted(results, key=lambda x: -x.confidence):
            ds   = r.dimensional_scores
            rs   = r.indicators.relative_strength
            rs_s = f"{rs:.3f}" if rs is not None else "  N/A "
            mark = "✗ " if not r.is_valid() else "  "
            print(
                f"  {mark}{r.symbol:<10} {r.stock_regime.value:<18} "
                f"{r.confidence:>6.2f}  "
                f"{ds.trend:>6.2f}  {ds.momentum:>6.2f}  {ds.volatility:>6.2f}  {rs_s:>7}"
            )

    # ── Rankings ─────────────────────────────────────────────────────
    for universe, results in output.stock_results.items():
        rankings = pipeline._stock_engine.get_rankings(results)
        print(f"\n{'─' * 65}")
        print(f"  {universe} — Top 5 by Trend Strength")
        print(f"{'─' * 65}")
        for e in rankings.strongest_trends[:5]:
            print(f"  #{e.rank}  {e.symbol:<12} {e.stock_regime:<18}  "
                  f"trend={e.score:.3f}  conf={e.confidence:.2f}")

        print(f"\n  {universe} — Top 5 by Momentum")
        for e in rankings.strongest_momentum[:5]:
            print(f"  #{e.rank}  {e.symbol:<12} {e.stock_regime:<18}  "
                  f"momentum={e.score:.3f}  conf={e.confidence:.2f}")

    # ── Detailed JSON for one stock ───────────────────────────────────
    for universe, results in output.stock_results.items():
        if results:
            best = max(results, key=lambda r: r.confidence if r.is_valid() else 0)
            print(f"\n{'─' * 65}")
            print(f"  Detailed JSON — {best.symbol} ({universe})")
            print(f"{'─' * 65}")
            print(json.dumps(best.to_dict(), indent=2))
            break   # just show one

    # ── Persisted output files ────────────────────────────────────────
    print(f"\n{'─' * 65}")
    print("  Persisted Output Files")
    print(f"{'─' * 65}")
    output_root = Path(pipeline._cfg["output"]["root_dir"])
    parquet_files = sorted(output_root.rglob("*.parquet"))
    if parquet_files:
        for p in parquet_files:
            print(f"  {str(p):<60}  {p.stat().st_size / 1024:6.1f} KB")
    else:
        print("  (none — persist=False or no data returned)")

    print(f"\n  Run completed in {output.elapsed_seconds:.1f}s")
    print("=" * 65)


if __name__ == "__main__":
    main()
