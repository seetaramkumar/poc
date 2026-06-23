"""
tests/test_sector_intelligence.py
===================================
Tests for the Sector Intelligence Engine.

Covers:
- SectorMap: loading, normalisation, fallback, coverage
- SectorMetrics: computation correctness, field ranges
- SectorClassifier: all 5 states, ranking, edge cases
- SectorEngine: end-to-end pipeline, persistence
- Diagnostics: context map, router integration
- Router: 5-level sector state PSM adjustments

Run:
    pytest tests/test_sector_intelligence.py -v
"""

from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))


# ─────────────────────────────────────────────────────────────────────────────
#  Shared helpers
# ─────────────────────────────────────────────────────────────────────────────

def _make_stable(
    symbol:        str,
    raw_regime:    str,
    stable_regime: str,
    conf:          float = 0.72,
    above_ema200:  bool  = True,
    rs_3m:         float = 1.03,
    rs_trend:      float = 0.001,
    trend_score:   float = 0.65,
    mom_score:     float = 0.55,
):
    """Build a minimal StableRegimeResult for testing."""
    from stock_regime.stability.stabiliser import StableRegimeResult
    from stock_regime.src.models import (
        StockRegime, StockRegimeResult, DimensionalScores,
        StockSignals, StockIndicatorSnapshot,
    )
    sig = StockSignals(price_above_ema200=above_ema200)
    ind = StockIndicatorSnapshot(
        close=1000.0, ema20=980.0, ema50=940.0, ema200=850.0,
        rs_3m=rs_3m, rs_trend=rs_trend,
    )
    raw = StockRegimeResult(
        symbol=symbol, market="TEST",
        stock_regime=StockRegime(raw_regime), confidence=conf,
        dimensional_scores=DimensionalScores(
            trend=trend_score, momentum=mom_score, volatility=0.30,
        ),
        regime_scores={}, signals=sig, indicators=ind,
    )
    return StableRegimeResult.from_result(
        raw,
        stable_regime=StockRegime(stable_regime),
        prior_stable_regime=StockRegime.UNCERTAIN,
        regime_age_bars=8, stable_regime_age=8,
        regime_changed_today=False,
        smoothed_confidence=conf, oscillation_detected=False,
    )


def _make_quality(symbol: str, score: float):
    from stock_regime.quality_engine.opportunity_quality import QualityScore
    return QualityScore(
        symbol=symbol, market="TEST", run_date=date.today(),
        quality_score=score, liquidity_quality=0.70,
        trend_quality=0.70, vol_health=0.65,
        stability_quality=0.65, tradability=0.70, notes=[],
    )


# ─────────────────────────────────────────────────────────────────────────────
#  SectorMap tests
# ─────────────────────────────────────────────────────────────────────────────

class TestSectorMap:
    def test_empty_map_returns_unknown(self):
        from stock_regime.sector_engine.sector_map import SectorMap
        m = SectorMap()
        assert m.get_sector("ANYTHING.NS") == "UNKNOWN"

    def test_inline_overrides_applied(self):
        from stock_regime.sector_engine.sector_map import SectorMap
        m = SectorMap(inline_overrides={"INFY.NS": "Information Technology"})
        assert m.get_sector("INFY.NS") == "INFORMATION_TECHNOLOGY"

    def test_symbol_case_insensitive(self):
        from stock_regime.sector_engine.sector_map import SectorMap
        m = SectorMap(inline_overrides={"infy.ns": "IT"})
        assert m.get_sector("INFY.NS") == "IT"

    def test_sector_normalisation(self):
        from stock_regime.sector_engine.sector_map import SectorMap
        m = SectorMap(inline_overrides={
            "A": "Banking & Finance",
            "B": "oil / gas",
            "C": "Real Estate",
        })
        assert m.get_sector("A") == "BANKING_AND_FINANCE"
        assert m.get_sector("B") == "OIL_GAS"
        assert m.get_sector("C") == "REAL_ESTATE"

    def test_csv_loading(self, tmp_path):
        from stock_regime.sector_engine.sector_map import SectorMap
        csv = tmp_path / "test.csv"
        csv.write_text("symbol,sector,industry\n"
                       "INFY.NS,IT,IT_SERVICES\n"
                       "TCS.NS,IT,IT_SERVICES\n"
                       "HDFCBANK.NS,BANKING,PRIVATE_BANKS\n")
        m = SectorMap(csv_path=csv)
        assert m.get_sector("INFY.NS")    == "IT"
        assert m.get_sector("HDFCBANK.NS")== "BANKING"
        assert m.get_industry("INFY.NS")  == "IT_SERVICES"
        assert m.get_sector("UNKNOWN.NS") == "UNKNOWN"

    def test_csv_missing_column_graceful(self, tmp_path):
        from stock_regime.sector_engine.sector_map import SectorMap
        csv = tmp_path / "bad.csv"
        csv.write_text("symbol,name\nINFY.NS,Infosys\n")  # no 'sector' column
        m = SectorMap(csv_path=csv)
        assert len(m) == 0   # graceful: no mappings loaded

    def test_all_sectors_returns_unique(self):
        from stock_regime.sector_engine.sector_map import SectorMap
        m = SectorMap(inline_overrides={"A": "IT", "B": "IT", "C": "BANKING"})
        sectors = m.get_all_sectors()
        assert len(sectors) == 2
        assert "IT" in sectors
        assert "BANKING" in sectors

    def test_symbols_in_sector(self):
        from stock_regime.sector_engine.sector_map import SectorMap
        m = SectorMap(inline_overrides={"A": "IT", "B": "IT", "C": "BANKING"})
        it_syms = m.symbols_in_sector("IT")
        assert set(it_syms) == {"A", "B"}

    def test_from_project_root_missing_file(self, tmp_path):
        from stock_regime.sector_engine.sector_map import SectorMap
        # Should not raise even if file is absent
        m = SectorMap.from_project_root(tmp_path, "NIFTY500")
        assert m.get_sector("ANYTHING") == "UNKNOWN"

    def test_coverage_stats(self, tmp_path):
        from stock_regime.sector_engine.sector_map import SectorMap
        csv = tmp_path / "s.csv"
        csv.write_text("symbol,sector\nA,IT\nB,IT\nC,BANKING\n")
        m   = SectorMap(csv_path=csv)
        cov = m.coverage()
        assert cov["total_mapped"]   == 3
        assert cov["unique_sectors"] == 2


# ─────────────────────────────────────────────────────────────────────────────
#  SectorMetrics computation tests
# ─────────────────────────────────────────────────────────────────────────────

class TestSectorMetrics:
    def _bullish_results(self, n=8):
        return [_make_stable(f"S{i}", "TREND_UP", "TREND_UP",
                              above_ema200=True, rs_3m=1.05, trend_score=0.75)
                for i in range(n)]

    def _bearish_results(self, n=5):
        return [_make_stable(f"B{i}", "TREND_DOWN", "TREND_DOWN",
                              above_ema200=False, rs_3m=0.93, trend_score=0.30)
                for i in range(n)]

    def test_bullish_sector_pct(self):
        from stock_regime.sector_engine.metrics import compute_sector_metrics
        results = self._bullish_results(8)
        m = compute_sector_metrics("IT", "TEST", date.today(), results, {})
        assert m.pct_bullish == 100.0
        assert m.pct_bearish == 0.0

    def test_bearish_sector_pct(self):
        from stock_regime.sector_engine.metrics import compute_sector_metrics
        results = self._bearish_results(5)
        m = compute_sector_metrics("WEAK", "TEST", date.today(), results, {})
        assert m.pct_bearish == 100.0
        assert m.pct_bullish == 0.0

    def test_mixed_sector(self):
        from stock_regime.sector_engine.metrics import compute_sector_metrics
        results = self._bullish_results(6) + self._bearish_results(4)
        m = compute_sector_metrics("MIXED", "TEST", date.today(), results, {})
        assert m.pct_bullish == 60.0
        assert m.pct_bearish == 40.0
        assert abs(m.advance_decline_ratio - 1.5) < 0.01

    def test_ema200_participation(self):
        from stock_regime.sector_engine.metrics import compute_sector_metrics
        results = (
            [_make_stable(f"U{i}", "TREND_UP", "TREND_UP", above_ema200=True) for i in range(7)] +
            [_make_stable(f"D{i}", "RANGE",    "RANGE",    above_ema200=False) for i in range(3)]
        )
        m = compute_sector_metrics("T", "TEST", date.today(), results, {})
        assert m.pct_above_ema200 == 70.0

    def test_avg_quality_from_map(self):
        from stock_regime.sector_engine.metrics import compute_sector_metrics
        results   = [_make_stable(f"S{i}", "TREND_UP", "TREND_UP") for i in range(4)]
        q_map     = {f"S{i}": 0.80 for i in range(4)}
        m = compute_sector_metrics("T", "TEST", date.today(), results, q_map)
        assert abs(m.avg_quality_score - 0.80) < 0.001

    def test_rs_aggregation(self):
        from stock_regime.sector_engine.metrics import compute_sector_metrics
        results = [
            _make_stable("A", "TREND_UP", "TREND_UP", rs_3m=1.08),
            _make_stable("B", "TREND_UP", "TREND_UP", rs_3m=1.04),
            _make_stable("C", "RANGE",    "RANGE",    rs_3m=0.97),
        ]
        m = compute_sector_metrics("T", "TEST", date.today(), results, {})
        expected_rs = (1.08 + 1.04 + 0.97) / 3
        assert abs(m.avg_rs_3m - expected_rs) < 0.001

    def test_returns_none_for_empty(self):
        from stock_regime.sector_engine.metrics import compute_sector_metrics
        m = compute_sector_metrics("T", "TEST", date.today(), [], {})
        assert m is None

    def test_all_metric_fields_in_range(self):
        from stock_regime.sector_engine.metrics import compute_sector_metrics
        results = self._bullish_results(5) + self._bearish_results(3)
        m = compute_sector_metrics("T", "TEST", date.today(), results, {})
        assert 0.0 <= m.pct_bullish   <= 100.0
        assert 0.0 <= m.pct_bearish   <= 100.0
        assert 0.0 <= m.avg_trend_score <= 1.0
        assert 0.0 <= m.avg_momentum_score <= 1.0

    def test_composite_score_in_unit_interval(self):
        from stock_regime.sector_engine.metrics import (
            compute_sector_metrics, compute_composite_score
        )
        results = self._bullish_results(6) + self._bearish_results(2)
        m     = compute_sector_metrics("T", "TEST", date.today(), results, {})
        score = compute_composite_score(m)
        assert 0.0 <= score <= 1.0

    def test_all_bullish_composite_higher_than_all_bearish(self):
        from stock_regime.sector_engine.metrics import (
            compute_sector_metrics, compute_composite_score
        )
        m_bull = compute_sector_metrics("BULL", "T", date.today(), self._bullish_results(8), {})
        m_bear = compute_sector_metrics("BEAR", "T", date.today(), self._bearish_results(8), {})
        s_bull = compute_composite_score(m_bull)
        s_bear = compute_composite_score(m_bear)
        assert s_bull > s_bear, f"Bull={s_bull:.3f} should > Bear={s_bear:.3f}"


# ─────────────────────────────────────────────────────────────────────────────
#  SectorClassifier tests
# ─────────────────────────────────────────────────────────────────────────────

class TestSectorClassifier:
    def _classifier(self):
        from stock_regime.sector_engine.classifier import SectorClassifier
        return SectorClassifier(
            leading_threshold=0.65, leading_bull_pct=55.0,
            strong_threshold=0.50,  neutral_threshold=0.38,
            weakening_threshold=0.25, min_stocks=3,
        )

    def _metrics(self, sector, pct_bullish=60.0, pct_bearish=20.0,
                  pct_above=65.0, trend=0.65, mom=0.55, quality=0.70,
                  rs=1.05, stocks=8):
        from stock_regime.sector_engine.models import SectorMetrics
        return SectorMetrics(
            sector=sector, universe="TEST", run_date=date.today(),
            stock_count=stocks,
            pct_bullish=pct_bullish, pct_bearish=pct_bearish,
            pct_above_ema200=pct_above,
            momentum_participation=15.0, breakout_participation=5.0,
            advance_decline_ratio=pct_bullish/max(pct_bearish,1),
            avg_trend_score=trend, avg_momentum_score=mom,
            avg_quality_score=quality, avg_rs_3m=rs,
            avg_rs_trend=0.001, avg_confidence=0.72, pct_oscillating=5.0,
        )

    def test_leading_state(self):
        from stock_regime.sector_engine.models import SectorState
        clf = self._classifier()
        m   = self._metrics("IT", pct_bullish=70.0, trend=0.80, mom=0.75, rs=1.08)
        snap= clf.classify(m, composite_score=0.75)
        assert snap.state == SectorState.LEADING

    def test_leading_requires_breadth(self):
        """High composite but low pct_bullish → STRONG, not LEADING."""
        from stock_regime.sector_engine.models import SectorState
        clf = self._classifier()
        m   = self._metrics("IT", pct_bullish=40.0, trend=0.80, mom=0.75)
        snap= clf.classify(m, composite_score=0.72)
        assert snap.state == SectorState.STRONG

    def test_strong_state(self):
        from stock_regime.sector_engine.models import SectorState
        clf  = self._classifier()
        snap = clf.classify(self._metrics("BANKING"), composite_score=0.58)
        assert snap.state == SectorState.STRONG

    def test_neutral_state(self):
        from stock_regime.sector_engine.models import SectorState
        clf  = self._classifier()
        snap = clf.classify(self._metrics("FMCG"), composite_score=0.42)
        assert snap.state == SectorState.NEUTRAL

    def test_weakening_state(self):
        from stock_regime.sector_engine.models import SectorState
        clf  = self._classifier()
        m    = self._metrics("METALS", pct_bullish=30.0, pct_bearish=45.0, trend=0.30)
        snap = clf.classify(m, composite_score=0.28)
        assert snap.state == SectorState.WEAKENING

    def test_weak_state(self):
        from stock_regime.sector_engine.models import SectorState
        clf  = self._classifier()
        m    = self._metrics("REALTY", pct_bullish=10.0, pct_bearish=70.0, trend=0.15)
        snap = clf.classify(m, composite_score=0.12)
        assert snap.state == SectorState.WEAK

    def test_unknown_below_min_stocks(self):
        from stock_regime.sector_engine.models import SectorState
        clf  = self._classifier()
        m    = self._metrics("SMALL", stocks=2)
        snap = clf.classify(m, composite_score=0.80)
        assert snap.state == SectorState.UNKNOWN

    def test_batch_ranks_correctly(self):
        clf = self._classifier()
        metrics = [
            self._metrics("IT",      pct_bullish=75.0, trend=0.80),  # should be rank 1
            self._metrics("BANKING", pct_bullish=55.0, trend=0.60),
            self._metrics("FMCG",    pct_bullish=30.0, trend=0.40),  # should be rank 3
        ]
        from stock_regime.sector_engine.metrics import compute_composite_score
        comp_map = {m.sector: compute_composite_score(m) for m in metrics}
        snaps    = clf.classify_batch(metrics, comp_map)
        # Sort by rank
        ranked = sorted(snaps, key=lambda s: s.rank)
        assert ranked[0].sector == "IT"
        assert ranked[-1].sector == "FMCG"

    def test_rank_1_is_highest_composite(self):
        clf = self._classifier()
        from stock_regime.sector_engine.metrics import compute_composite_score
        metrics = [self._metrics(f"S{i}", pct_bullish=float(30+i*10)) for i in range(5)]
        comp_map = {m.sector: compute_composite_score(m) for m in metrics}
        snaps    = clf.classify_batch(metrics, comp_map)
        top      = min(snaps, key=lambda s: s.rank)
        assert top.composite_score == max(s.composite_score for s in snaps)


# ─────────────────────────────────────────────────────────────────────────────
#  SectorEngine end-to-end tests
# ─────────────────────────────────────────────────────────────────────────────

class TestSectorEngine:
    def _engine(self, csv_content: str = None, tmp_path=None):
        from stock_regime.sector_engine import SectorEngine, SectorMap, SectorClassifier
        if csv_content and tmp_path:
            p = tmp_path / "sectors.csv"
            p.write_text(csv_content)
            sm = SectorMap(csv_path=p)
        else:
            sm = SectorMap(inline_overrides={
                "INFY.NS": "IT", "TCS.NS": "IT", "WIPRO.NS": "IT",
                "HDFCBANK.NS": "BANKING", "ICICIBANK.NS": "BANKING", "AXISBANK.NS": "BANKING",
                "SUNPHARMA.NS": "PHARMA", "DRREDDY.NS": "PHARMA", "CIPLA.NS": "PHARMA",
            })
        clf = SectorClassifier()
        return SectorEngine(sector_map=sm, classifier=clf)

    def _results(self, regimes: dict):
        return [_make_stable(sym, r, r) for sym, r in regimes.items()]

    def _quality(self, symbols: list, score=0.70):
        return [_make_quality(sym, score) for sym in symbols]

    def test_compute_returns_snapshots(self):
        engine  = self._engine()
        symbols = {
            "INFY.NS": "TREND_UP", "TCS.NS": "TREND_UP", "WIPRO.NS": "MOMENTUM",
            "HDFCBANK.NS": "TREND_UP", "ICICIBANK.NS": "RANGE", "AXISBANK.NS": "TREND_DOWN",
            "SUNPHARMA.NS": "RANGE", "DRREDDY.NS": "RANGE", "CIPLA.NS": "QUIET",
        }
        results  = self._results(symbols)
        quality  = self._quality(list(symbols.keys()))
        snaps    = engine.compute(results, quality, "TEST")
        assert len(snaps) > 0
        for s in snaps:
            assert 0.0 <= s.composite_score <= 1.0
            assert s.rank >= 1

    def test_sectors_ranked_by_performance(self):
        """IT sector (all bullish) should rank above PHARMA (all bearish)."""
        engine = self._engine()
        results = self._results({
            "INFY.NS": "TREND_UP", "TCS.NS": "TREND_UP", "WIPRO.NS": "MOMENTUM",
            "SUNPHARMA.NS": "TREND_DOWN", "DRREDDY.NS": "TREND_DOWN", "CIPLA.NS": "TREND_DOWN",
        })
        quality = self._quality(["INFY.NS","TCS.NS","WIPRO.NS","SUNPHARMA.NS","DRREDDY.NS","CIPLA.NS"])
        snaps   = engine.compute(results, quality, "TEST")
        sector_rank = {s.sector: s.rank for s in snaps}
        if "IT" in sector_rank and "PHARMA" in sector_rank:
            assert sector_rank["IT"] < sector_rank["PHARMA"], \
                "IT (bullish) should rank higher (lower rank number) than PHARMA (bearish)"

    def test_unknown_symbols_grouped_as_unknown(self):
        engine  = self._engine()   # only IT/BANKING/PHARMA mapped
        results = self._results({"NEWSTOCK.NS": "TREND_UP", "ANOTHER.NS": "MOMENTUM"})
        quality = self._quality(["NEWSTOCK.NS", "ANOTHER.NS"])
        snaps   = engine.compute(results, quality, "TEST")
        unknown_snaps = [s for s in snaps if s.sector == "UNKNOWN"]
        assert len(unknown_snaps) == 1
        assert unknown_snaps[0].stock_count == 2

    def test_get_sector_returns_correct_sector(self):
        engine = self._engine()
        assert engine.get_sector("INFY.NS")     == "IT"
        assert engine.get_sector("HDFCBANK.NS") == "BANKING"
        assert engine.get_sector("UNKNOWN.NS")  == "UNKNOWN"

    def test_get_symbol_sector_state(self):
        engine  = self._engine()
        results = self._results({
            "INFY.NS": "TREND_UP", "TCS.NS": "TREND_UP", "WIPRO.NS": "TREND_UP",
        })
        quality = self._quality(["INFY.NS","TCS.NS","WIPRO.NS"])
        engine.compute(results, quality, "TEST")
        state = engine.get_symbol_sector_state("INFY.NS")
        # Should be one of the 5 SectorState values
        from stock_regime.sector_engine.models import SectorState
        assert state in [s.value for s in SectorState]

    def test_persist_creates_three_parquets(self, tmp_path):
        engine  = self._engine()
        results = self._results({
            "INFY.NS": "TREND_UP", "TCS.NS": "TREND_UP", "WIPRO.NS": "MOMENTUM",
            "HDFCBANK.NS": "TREND_UP", "ICICIBANK.NS": "RANGE", "AXISBANK.NS": "TREND_DOWN",
        })
        quality = self._quality(list({"INFY.NS","TCS.NS","WIPRO.NS","HDFCBANK.NS","ICICIBANK.NS","AXISBANK.NS"}))
        snaps   = engine.compute(results, quality, "TEST")
        saved   = engine.persist(snaps, tmp_path, append=False)
        assert "metrics"  in saved and saved["metrics"].exists()
        assert "states"   in saved and saved["states"].exists()
        assert "rankings" in saved and saved["rankings"].exists()

    def test_parquet_schema_correct(self, tmp_path):
        engine  = self._engine()
        results = self._results({
            "INFY.NS": "TREND_UP", "TCS.NS": "TREND_UP", "WIPRO.NS": "MOMENTUM",
        })
        quality = self._quality(["INFY.NS","TCS.NS","WIPRO.NS"])
        snaps   = engine.compute(results, quality, "TEST")
        saved   = engine.persist(snaps, tmp_path, append=False)
        # metrics parquet
        df = pd.read_parquet(saved["metrics"])
        for col in ["sector","universe","run_date","pct_bullish","pct_above_ema200",
                    "avg_trend_score","avg_momentum_score","avg_quality_score"]:
            assert col in df.columns, f"Missing column: {col}"
        # states parquet
        df_s = pd.read_parquet(saved["states"])
        for col in ["sector","state","rank","composite_score"]:
            assert col in df_s.columns, f"Missing column in states: {col}"

    def test_persist_append_deduplicates(self, tmp_path):
        engine  = self._engine()
        results = self._results({
            "INFY.NS": "TREND_UP", "TCS.NS": "TREND_UP", "WIPRO.NS": "TREND_UP",
        })
        quality = self._quality(["INFY.NS","TCS.NS","WIPRO.NS"])
        snaps   = engine.compute(results, quality, "TEST")
        # Write twice with same run_date
        engine.persist(snaps, tmp_path, append=False)
        engine.persist(snaps, tmp_path, append=True)
        df = pd.read_parquet(tmp_path / "sectors" / "sector_states.parquet")
        # Should not have duplicate (sector, universe, run_date) rows
        dupes = df.duplicated(subset=["sector","universe","run_date"]).sum()
        assert dupes == 0


# ─────────────────────────────────────────────────────────────────────────────
#  Diagnostics — sector context map
# ─────────────────────────────────────────────────────────────────────────────

class TestSectorDiagnostics:
    def _make_snapshots(self, sector_states: dict):
        from stock_regime.sector_engine.models import (
            SectorMetrics, SectorSnapshot, SectorState
        )
        snaps = []
        for i, (sector, state) in enumerate(sector_states.items()):
            m = SectorMetrics(
                sector=sector, universe="TEST", run_date=date.today(),
                stock_count=5, pct_bullish=60.0, pct_bearish=15.0,
                pct_above_ema200=65.0, momentum_participation=15.0,
                breakout_participation=5.0, advance_decline_ratio=2.5,
                avg_trend_score=0.65, avg_momentum_score=0.55,
                avg_quality_score=0.70, avg_rs_3m=1.04,
                avg_rs_trend=0.001, avg_confidence=0.72, pct_oscillating=5.0,
            )
            snaps.append(SectorSnapshot(
                metrics=m, state=SectorState(state),
                rank=i+1, composite_score=0.70-(i*0.10),
            ))
        return snaps

    def test_context_map_keys_and_fields(self):
        from stock_regime.sector_engine.diagnostics import build_sector_context_map
        snaps = self._make_snapshots({"IT": "LEADING", "BANKING": "STRONG"})
        ctx   = build_sector_context_map(snaps)
        assert "IT"      in ctx
        assert "BANKING" in ctx
        for key in ["state","composite_score","rank","pct_bullish","stock_count"]:
            assert key in ctx["IT"], f"Missing key: {key}"

    def test_context_map_unknown_excluded(self):
        from stock_regime.sector_engine.diagnostics import build_sector_context_map
        from stock_regime.sector_engine.models import SectorMetrics, SectorSnapshot, SectorState
        m = SectorMetrics(
            sector="SMALL", universe="TEST", run_date=date.today(),
            stock_count=1, pct_bullish=50.0, pct_bearish=20.0,
            pct_above_ema200=50.0, momentum_participation=0.0,
            breakout_participation=0.0, advance_decline_ratio=1.5,
            avg_trend_score=0.50, avg_momentum_score=0.40,
            avg_quality_score=0.50, avg_rs_3m=1.0,
            avg_rs_trend=0.0, avg_confidence=0.60, pct_oscillating=10.0,
        )
        snaps = [SectorSnapshot(metrics=m, state=SectorState.UNKNOWN, rank=1, composite_score=0.50)]
        ctx   = build_sector_context_map(snaps)
        assert "SMALL" not in ctx  # UNKNOWN sectors excluded from router context


# ─────────────────────────────────────────────────────────────────────────────
#  Strategy router — 5-level sector state PSM tests
# ─────────────────────────────────────────────────────────────────────────────

class TestRouterFiveLevelSector:
    """Verify the 5-level sector state produces graduated PSM adjustments."""

    def _route(self, sector_state: str, quality: float = 0.72,
               market: str = "BULLISH_TREND", breadth: str = "NEUTRAL",
               b_score: float = 0.55):
        from stock_regime.strategy_router import StrategyRouter
        router = StrategyRouter()
        sr     = _make_stable("SYM", "TREND_UP", "TREND_UP", conf=0.75)
        qs     = _make_quality("SYM", quality)
        return router.route_batch(
            [sr], [qs], market_regime=market,
            breadth_state=breadth, breadth_score=b_score,
            sector_states={"SYM": sector_state},
        )[0]

    def test_leading_sector_highest_psm(self):
        d = self._route("LEADING")
        assert d.allowed
        assert d.position_size_multiplier > 1.0, \
            f"LEADING sector should boost PSM > 1.0, got {d.position_size_multiplier}"

    def test_strong_sector_above_neutral(self):
        d_strong  = self._route("STRONG")
        d_neutral = self._route("NEUTRAL")
        assert d_strong.position_size_multiplier >= d_neutral.position_size_multiplier

    def test_weakening_sector_below_neutral(self):
        d_weak    = self._route("WEAKENING")
        d_neutral = self._route("NEUTRAL")
        assert d_weak.position_size_multiplier < d_neutral.position_size_multiplier

    def test_weak_sector_lowest_psm(self):
        d_weak = self._route("WEAK")
        d_lead = self._route("LEADING")
        assert d_weak.position_size_multiplier < d_lead.position_size_multiplier, \
            f"WEAK ({d_weak.position_size_multiplier}) should be < LEADING ({d_lead.position_size_multiplier})"

    def test_all_five_states_monotone(self):
        """PSM should be monotonically decreasing from LEADING → WEAK."""
        states = ["LEADING", "STRONG", "NEUTRAL", "WEAKENING", "WEAK"]
        psms   = []
        for state in states:
            d = self._route(state)
            if d.allowed:
                psms.append((state, d.position_size_multiplier))
        for i in range(1, len(psms)):
            assert psms[i][1] <= psms[i-1][1], \
                f"PSM not monotone: {psms[i-1]} → {psms[i]}"

    def test_breakout_blocked_in_weak_sector(self):
        """BREAKOUT_SETUP must be blocked when sector is WEAKENING or WEAK."""
        from stock_regime.strategy_router import StrategyRouter
        router = StrategyRouter()
        for bad_sector in ["WEAKENING", "WEAK"]:
            sr = _make_stable("SYM", "BREAKOUT_SETUP", "BREAKOUT_SETUP", conf=0.75)
            qs = _make_quality("SYM", 0.75)
            d  = router.route_batch(
                [sr], [qs], market_regime="BULLISH_TREND",
                breadth_state="EXPANDING", breadth_score=0.70,
                sector_states={"SYM": bad_sector},
            )[0]
            assert d.strategy == "NO_TRADE", \
                f"BREAKOUT_SETUP should be NO_TRADE in {bad_sector} sector"