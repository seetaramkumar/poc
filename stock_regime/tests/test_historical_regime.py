from pathlib import Path

import pandas as pd

from stock_regime.analytics.historical_regime import HistoricalRegimeEngine


def _build_history_frame(dates: pd.DatetimeIndex) -> pd.DataFrame:
    rows = []
    for i, d in enumerate(dates):
        close = 100 + i * 0.5
        open_price = close - 0.2
        high = close + 0.8
        low = close - 0.8
        volume = 1000000 + i * 1000
        rows.append({
            "open": open_price,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
        })
    return pd.DataFrame(rows, index=dates)


def test_build_series_from_histories_generates_expected_schema(tmp_path: Path) -> None:
    dates = pd.date_range("2024-01-02", periods=20, freq="B")
    df = _build_history_frame(dates)

    engine = HistoricalRegimeEngine(
        config_path=Path("stock_regime/config/config.yaml"),
        output_dir=str(tmp_path),
        lookback_days=10,
        persist_history=False,
    )

    series = engine.build_series_from_histories(
        symbol_histories={"TEST": df},
        benchmark_history=df,
        symbols=["TEST"],
        market_label="NIFTY500",
    )

    assert not series.empty
    assert list(series.columns) == ["date", "symbol", "market", "regime", "confidence", "stable_regime"]
    assert series["symbol"].eq("TEST").all()
    assert set(series["market"]).issubset({"NIFTY500"})
    assert set(series["regime"]).issubset({
        "TREND_UP",
        "TREND_DOWN",
        "RANGE",
        "MOMENTUM",
        "BREAKOUT_SETUP",
        "VOLATILE",
        "QUIET",
        "UNCERTAIN",
    })


def test_incremental_mode_appends_only_missing_dates(tmp_path: Path) -> None:
    first_dates = pd.date_range("2024-02-01", periods=8, freq="B")
    second_dates = pd.date_range("2024-02-01", periods=10, freq="B")
    first_df = _build_history_frame(first_dates)
    second_df = _build_history_frame(second_dates)

    engine = HistoricalRegimeEngine(
        config_path=Path("stock_regime/config/config.yaml"),
        output_dir=str(tmp_path),
        lookback_days=3,
        warmup_days=2,
        persist_history=True,
        output_file=tmp_path / "historical_regime_series.parquet",
    )

    engine.build_series_from_histories(
        symbol_histories={"TEST": first_df},
        benchmark_history=first_df,
        symbols=["TEST"],
        market_label="NIFTY500",
    )

    persisted_before = pd.read_parquet(engine.output_path)
    initial_dates = set(pd.to_datetime(persisted_before["date"]).dt.date)

    engine.build_series_from_histories(
        symbol_histories={"TEST": second_df},
        benchmark_history=second_df,
        symbols=["TEST"],
        market_label="NIFTY500",
    )

    persisted_after = pd.read_parquet(engine.output_path)
    assert set(pd.to_datetime(persisted_after["date"]).dt.date) >= initial_dates
    assert "stable_regime" in persisted_after.columns
    assert not persisted_after.empty
