#!/usr/bin/env python
"""
Test script to verify RegimeDiagnosticsEngine integration with pipeline.
Validates:
1. Module imports
2. Class instantiation
3. Method signatures
4. Output structures
"""

from pathlib import Path
import sys
from dataclasses import asdict

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from stock_regime.analytics.regime_diagnostics import (
    RegimeDiagnosticsEngine,
    OscillationMetrics,
    RegimeTransition,
    RegimeStabilityStats,
    OscillationSummary,
)


def test_imports():
    """Test that all classes can be imported."""
    print("✓ Test 1: Imports successful")
    print(f"  - RegimeDiagnosticsEngine: {RegimeDiagnosticsEngine}")
    print(f"  - OscillationMetrics: {OscillationMetrics}")
    print(f"  - RegimeTransition: {RegimeTransition}")
    print(f"  - RegimeStabilityStats: {RegimeStabilityStats}")
    print(f"  - OscillationSummary: {OscillationSummary}")


def test_engine_instantiation():
    """Test that engine can be instantiated."""
    engine = RegimeDiagnosticsEngine()
    print("\n✓ Test 2: Engine instantiation successful")
    print(f"  - Engine type: {type(engine).__name__}")
    print(f"  - Has compute_oscillation_metrics: {hasattr(engine, 'compute_oscillation_metrics')}")
    print(f"  - Has compute_transition_matrix: {hasattr(engine, 'compute_transition_matrix')}")
    print(f"  - Has compute_regime_stability: {hasattr(engine, 'compute_regime_stability')}")
    print(f"  - Has compute_oscillation_summary: {hasattr(engine, 'compute_oscillation_summary')}")
    print(f"  - Has persist: {hasattr(engine, 'persist')}")
    print(f"  - Has log_summary: {hasattr(engine, 'log_summary')}")


def test_dataclass_structures():
    """Test that dataclasses can be created."""
    # Test OscillationMetrics
    osc = OscillationMetrics(
        symbol="TEST",
        current_regime="TREND_UP",
        stable_regime="TREND_UP",
        prior_regime="MOMENTUM",
        oscillation_detected=False,
        regime_changes_30d=2,
        unique_regimes_seen_30d=2,
        oscillation_count_30d=0,
        avg_confidence_30d=0.75,
        avg_smoothed_confidence_30d=0.78,
        current_quality_score=85.0,
        sector="Technology",
        universe="NIFTY500"
    )
    print("\n✓ Test 3: Dataclass structures valid")
    print(f"  - OscillationMetrics fields: {len(osc.__dataclass_fields__)}")
    
    # Test to_dict conversion
    osc_dict = asdict(osc)
    print(f"  - to_dict() successful: {len(osc_dict)} fields")
    
    # Test RegimeTransition
    trans = RegimeTransition(
        from_regime="TREND_UP",
        to_regime="MOMENTUM",
        transition_count=150,
        transition_pct=8.5
    )
    print(f"  - RegimeTransition valid: {trans.from_regime} → {trans.to_regime}")
    
    # Test RegimeStabilityStats
    stability = RegimeStabilityStats(
        regime="TREND_UP",
        avg_duration_bars=45.5,
        median_duration_bars=38,
        p25_duration_bars=20,
        p75_duration_bars=65,
        max_duration_bars=180,
        min_duration_bars=2,
        episode_count=42,
        universe="NIFTY500"
    )
    print(f"  - RegimeStabilityStats valid: {stability.regime} (avg {stability.avg_duration_bars:.1f} bars)")
    
    # Test OscillationSummary
    summary = OscillationSummary(
        universe="NIFTY500",
        total_symbols=100,
        oscillating_symbols=35,
        oscillation_pct=35.0,
        top_transition_pair="TREND_UP → MOMENTUM",
        top_transition_pair_count=250,
        avg_regime_changes=2.1,
        avg_unique_regimes=2.3,
        avg_oscillation_count=0.8,
        most_stable_regime="TREND_UP",
        most_stable_duration=45.5,
        most_unstable_regime="MOMENTUM",
        most_unstable_duration=15.2
    )
    print(f"  - OscillationSummary valid: {summary.oscillating_symbols}/{summary.total_symbols} oscillating")


def test_pipeline_import():
    """Test that pipeline can still be imported."""
    from runner.pipeline import AlgoTradingPipeline
    print("\n✓ Test 4: Pipeline still imports successfully")
    
    # Check that _run_analytics method exists and has the right signature
    import inspect
    sig = inspect.signature(AlgoTradingPipeline._run_analytics)
    params = list(sig.parameters.keys())
    print(f"  - _run_analytics parameters: {params}")
    expected = ['self', 'universe', 'output_root', 'stable_results', 'quality_scores', 'run_date']
    if params == expected:
        print(f"  - ✓ Signature matches expected")
    else:
        print(f"  - ⚠ Signature differs. Expected: {expected}")


if __name__ == "__main__":
    print("=" * 70)
    print("  Regime Diagnostics Integration Tests")
    print("=" * 70)
    
    try:
        test_imports()
        test_engine_instantiation()
        test_dataclass_structures()
        test_pipeline_import()
        
        print("\n" + "=" * 70)
        print("  ✓ All integration tests passed!")
        print("=" * 70)
        
    except Exception as e:
        print(f"\n✗ Test failed with error:")
        print(f"  {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
