"""
stock_regime/analytics/historical_regime.py
==========================================
Historical regime reconstruction engine.

This component replays the existing stock regime classification pipeline over
historical OHLCV data to build a daily regime series without depending on
bot execution history. It is analytics-only and does not modify trading
logic or classification rules.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Optional

import pandas as pd

from market_regime.src.engine import MarketRegimeEngine
from stock_regime.src.engine import StockRegimeEngine
from stock_regime.src.models import MarketRegimeInput

logger = logging.getLogger(__name__)


class HistoricalRegimeEngine:
    """Incrementally reconstruct and maintain a historical regime series."""

    def __init__(
        self,
        config_path: Optional[Path] = None,
        output_dir: str | Path = "output",
        lookback_days: int = 90,
        warmup_days: int = 250,
        persist_history: bool = True,
        output_file: Optional[str | Path] = None,
        rebuild: bool = False,
    ) -> None:
        self.output_dir = Path(output_dir)
        self.lookback_days = int(lookback_days)
        self.warmup_days = int(warmup_days)
        self.persist_history = bool(persist_history)
        self.rebuild = bool(rebuild)

        self._stock_engine = StockRegimeEngine(config_path=config_path, output_dir=self.output_dir)
        self._market_engine = MarketRegimeEngine()

        analytics_cfg = getattr(self._stock_engine.config, "analytics", None)
        historical_cfg = None
        if analytics_cfg is not None:
            historical_cfg = getattr(analytics_cfg, "historical_regime", None)

        if historical_cfg is not None:
            self.lookback_days = int(getattr(historical_cfg, "lookback_days", self.lookback_days) or self.lookback_days)
            self.warmup_days = int(getattr(historical_cfg, "warmup_days", self.warmup_days) or self.warmup_days)
            self.persist_history = bool(getattr(historical_cfg, "persist_history", self.persist_history))
            self.rebuild = bool(getattr(historical_cfg, "rebuild", self.rebuild))
            configured_output = getattr(historical_cfg, "output_file", output_file)
            if configured_output is not None:
                self.output_file = Path(configured_output)
            else:
                self.output_file = Path(output_file) if output_file is not None else Path("output/historical_regimes/historical_regime_series.parquet")
        else:
            self.output_file = Path(output_file) if output_file is not None else Path("output/historical_regimes/historical_regime_series.parquet")

        self.total_window = self.lookback_days + self.warmup_days
        self.output_path = self._resolve_output_path(self.output_file)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def build_series_from_histories(
        self,
        symbol_histories: dict[str, pd.DataFrame],
        benchmark_history: Optional[pd.DataFrame] = None,
        symbols: Optional[list[str]] = None,
        market_label: str = "UNKNOWN",
    ) -> pd.DataFrame:
        """Create or extend a daily regime series from supplied OHLCV histories."""
        if not symbol_histories:
            logger.warning("No symbol histories supplied for historical regime reconstruction")
            return self._empty_series()

        started_at = time.perf_counter()
        target_symbols = symbols or list(symbol_histories.keys())
        benchmark_df = self._normalise_frame(benchmark_history) if benchmark_history is not None else None

        if self.rebuild and self.output_path.exists():
            self.output_path.unlink(missing_ok=True)

        existing_series = self._load_existing_series()
        if self.rebuild or existing_series.empty:
            mode = "BOOTSTRAP"
            series = self._build_bootstrap_series(target_symbols, symbol_histories, benchmark_df, market_label)
            rows_added = len(series)
        else:
            mode = "INCREMENTAL"
            series = self._build_incremental_series(existing_series, target_symbols, symbol_histories, benchmark_df, market_label)
            rows_added = max(len(series) - len(existing_series), 0)

        series = self._finalise_series(series)
        if self.persist_history:
            self.persist(series)

        runtime_seconds = round(time.perf_counter() - started_at, 2)
        if mode == "BOOTSTRAP":
            logger.info(
                "HistoricalRegimeEngine Mode=BOOTSTRAP Stocks=%d BarsLoaded=%d RowsGenerated=%d Runtime=%.2fs",
                len(target_symbols),
                self._estimate_bar_count(symbol_histories),
                len(series),
                runtime_seconds,
            )
        else:
            logger.info(
                "HistoricalRegimeEngine Mode=INCREMENTAL LastStoredDate=%s LatestAvailableDate=%s MissingTradingDays=%d RowsAdded=%d Runtime=%.2fs",
                self._latest_date(series).date().isoformat() if not series.empty else "n/a",
                self._latest_available_date(symbol_histories).date().isoformat() if symbol_histories else "n/a",
                self._estimate_missing_days(existing_series, symbol_histories),
                rows_added,
                runtime_seconds,
            )
        return series

    def persist(self, series: pd.DataFrame, output_path: Optional[str | Path] = None) -> Path:
        """Persist the historical regime series to parquet."""
        if series is None or series.empty:
            logger.warning("No historical regime rows to persist")
            return self.output_path

        path = self._resolve_output_path(output_path or self.output_file)
        path.parent.mkdir(parents=True, exist_ok=True)
        series.to_parquet(path, engine="pyarrow", compression="snappy", index=False)
        self.output_path = path
        logger.info("Persisted historical regime series → %s", path)
        return path

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_bootstrap_series(
        self,
        target_symbols: list[str],
        symbol_histories: dict[str, pd.DataFrame],
        benchmark_df: Optional[pd.DataFrame],
        market_label: str,
    ) -> pd.DataFrame:
        rows: list[dict] = []
        for symbol in target_symbols:
            history_df = self._normalise_frame(symbol_histories.get(symbol))
            if history_df is None or history_df.empty:
                continue
            rows.extend(
                self._reconstruct_symbol_series(
                    symbol=symbol,
                    history_df=history_df,
                    benchmark_df=benchmark_df,
                    market_label=market_label,
                )
            )

        if not rows:
            logger.warning("Historical regime reconstruction generated no rows")
            return self._empty_series()

        return pd.DataFrame(rows, columns=["date", "symbol", "market", "regime", "confidence", "stable_regime"])

    def _build_incremental_series(
        self,
        existing_series: pd.DataFrame,
        target_symbols: list[str],
        symbol_histories: dict[str, pd.DataFrame],
        benchmark_df: Optional[pd.DataFrame],
        market_label: str,
    ) -> pd.DataFrame:
        rows: list[dict] = []
        existing_series = existing_series.copy()
        existing_series["date"] = pd.to_datetime(existing_series["date"])

        for symbol in target_symbols:
            history_df = self._normalise_frame(symbol_histories.get(symbol))
            if history_df is None or history_df.empty:
                continue

            symbol_existing = existing_series.loc[existing_series["symbol"] == symbol].copy()
            symbol_existing_dates = set(pd.to_datetime(symbol_existing["date"]).dt.normalize())
            last_stored_date = None
            if not symbol_existing.empty:
                last_stored_date = pd.to_datetime(symbol_existing["date"]).max().normalize()

            for idx in range(self._min_required_bars(len(history_df)), len(history_df)):
                date_value = history_df.index[idx]
                if last_stored_date is not None and pd.Timestamp(date_value).normalize() <= last_stored_date:
                    continue
                if pd.Timestamp(date_value).normalize() in symbol_existing_dates:
                    continue

                tail = history_df.iloc[max(0, idx - self.total_window + 1): idx + 1].copy()
                if len(tail) < self._min_required_bars(len(history_df)):
                    continue

                benchmark_slice = self._slice_benchmark(benchmark_df, tail.index[-1])
                market_ctx = self._build_market_context(benchmark_slice)
                try:
                    result = self._stock_engine.analyze_single(
                        symbol=symbol,
                        df=tail,
                        market_regime=market_ctx,
                        benchmark_data=benchmark_slice,
                        market_label=market_label,
                    )
                except Exception as exc:
                    logger.debug("Historical classification failed for %s on %s: %s", symbol, tail.index[-1], exc)
                    continue

                regime = getattr(result.stock_regime, "value", str(result.stock_regime))
                rows.append({
                    "date": tail.index[-1],
                    "symbol": symbol,
                    "market": market_label,
                    "regime": regime,
                    "confidence": float(getattr(result, "confidence", 0.0) or 0.0),
                    "stable_regime": regime,
                })

        if not rows:
            return existing_series

        new_rows = pd.DataFrame(rows, columns=["date", "symbol", "market", "regime", "confidence", "stable_regime"])
        combined = pd.concat([existing_series, new_rows], ignore_index=True, sort=False)
        return combined

    def _reconstruct_symbol_series(
        self,
        symbol: str,
        history_df: pd.DataFrame,
        benchmark_df: Optional[pd.DataFrame],
        market_label: str,
    ) -> list[dict]:
        rows: list[dict] = []
        if history_df.empty:
            return rows

        min_required_bars = self._min_required_bars(len(history_df))
        for idx in range(min_required_bars - 1, len(history_df)):
            tail = history_df.iloc[max(0, idx - self.total_window + 1): idx + 1].copy()
            if len(tail) < min_required_bars:
                continue

            benchmark_slice = self._slice_benchmark(benchmark_df, tail.index[-1])
            market_ctx = self._build_market_context(benchmark_slice)
            try:
                result = self._stock_engine.analyze_single(
                    symbol=symbol,
                    df=tail,
                    market_regime=market_ctx,
                    benchmark_data=benchmark_slice,
                    market_label=market_label,
                )
            except Exception as exc:
                logger.debug("Historical classification failed for %s on %s: %s", symbol, tail.index[-1], exc)
                continue

            regime = getattr(result.stock_regime, "value", str(result.stock_regime))
            rows.append({
                "date": tail.index[-1],
                "symbol": symbol,
                "market": market_label,
                "regime": regime,
                "confidence": float(getattr(result, "confidence", 0.0) or 0.0),
                "stable_regime": regime,
            })

        return rows

    def _finalise_series(self, series: pd.DataFrame) -> pd.DataFrame:
        if series is None or series.empty:
            return self._empty_series()

        series = series.copy()
        if "date" in series.columns:
            series["date"] = pd.to_datetime(series["date"])
        if "confidence" in series.columns:
            series["confidence"] = pd.to_numeric(series["confidence"], errors="coerce")
        if "stable_regime" not in series.columns:
            series["stable_regime"] = series["regime"]

        series = series.sort_values(["symbol", "date"]).reset_index(drop=True)
        series = series.groupby("symbol", group_keys=False).tail(self.lookback_days)
        series = series.sort_values(["date", "symbol"]).reset_index(drop=True)
        return series[["date", "symbol", "market", "regime", "confidence", "stable_regime"]]

    def _load_existing_series(self) -> pd.DataFrame:
        if not self.output_path.exists():
            return self._empty_series()
        try:
            series = pd.read_parquet(self.output_path)
        except Exception as exc:
            logger.warning("Unable to read existing historical regime series: %s", exc)
            return self._empty_series()
        if series.empty:
            return self._empty_series()
        series = series.copy()
        if "date" in series.columns:
            series["date"] = pd.to_datetime(series["date"])
        if "stable_regime" not in series.columns and "regime" in series.columns:
            series["stable_regime"] = series["regime"]
        return series

    def _build_market_context(self, benchmark_df: Optional[pd.DataFrame]) -> MarketRegimeInput:
        if benchmark_df is None or benchmark_df.empty:
            return MarketRegimeInput(regime="UNCERTAIN", confidence=0.0)
        try:
            market_result = self._market_engine.analyze(benchmark_df, row_index=-1)
            regime = getattr(market_result.regime, "value", str(market_result.regime))
            return MarketRegimeInput(regime=regime, confidence=float(getattr(market_result, "confidence", 0.0) or 0.0))
        except Exception as exc:
            logger.debug("Market regime context unavailable: %s", exc)
            return MarketRegimeInput(regime="UNCERTAIN", confidence=0.0)

    @staticmethod
    def _normalise_frame(frame: Optional[pd.DataFrame]) -> Optional[pd.DataFrame]:
        if frame is None:
            return None
        if frame.empty:
            return frame.copy()

        df = frame.copy()
        if not isinstance(df.index, pd.DatetimeIndex):
            if "date" in df.columns:
                df = df.set_index(pd.to_datetime(df["date"]))
            else:
                df.index = pd.to_datetime(df.index)
        df.columns = [c.lower().strip() for c in df.columns]
        if "date" not in df.columns and isinstance(df.index, pd.DatetimeIndex):
            df = df.copy()
            df.index.name = "date"
        return df

    @staticmethod
    def _slice_benchmark(benchmark_df: Optional[pd.DataFrame], as_of: pd.Timestamp) -> Optional[pd.DataFrame]:
        if benchmark_df is None or benchmark_df.empty:
            return None
        try:
            slice_df = benchmark_df.loc[benchmark_df.index <= as_of].copy()
            return slice_df if not slice_df.empty else None
        except Exception:
            return None

    @staticmethod
    def _min_required_bars(history_length: int) -> int:
        if history_length <= 0:
            return 1
        return max(5, min(history_length, 20))

    @staticmethod
    def _latest_date(series: pd.DataFrame) -> pd.Timestamp:
        if series is None or series.empty:
            return pd.Timestamp("1970-01-01")
        return pd.to_datetime(series["date"]).max()

    @staticmethod
    def _latest_available_date(symbol_histories: dict[str, pd.DataFrame]) -> pd.Timestamp:
        dates: list[pd.Timestamp] = []
        for frame in symbol_histories.values():
            if frame is None or getattr(frame, "empty", True):
                continue
            normalised = HistoricalRegimeEngine._normalise_frame(frame)
            if normalised is None or normalised.empty:
                continue
            if isinstance(normalised.index, pd.DatetimeIndex):
                dates.append(pd.Timestamp(normalised.index[-1]))
        return max(dates) if dates else pd.Timestamp("1970-01-01")

    def _estimate_bar_count(self, symbol_histories: dict[str, pd.DataFrame]) -> int:
        total = 0
        for frame in symbol_histories.values():
            normalised = self._normalise_frame(frame)
            if normalised is None:
                continue
            total += len(normalised)
        return total

    def _estimate_missing_days(self, existing_series: pd.DataFrame, symbol_histories: dict[str, pd.DataFrame]) -> int:
        if existing_series.empty or not symbol_histories:
            return 0
        return max(0, len(symbol_histories) * max(1, self.lookback_days))

    def _resolve_output_path(self, output_path: str | Path) -> Path:
        path = Path(output_path)
        if not path.is_absolute():
            path = self.output_dir / path
        return path

    @staticmethod
    def _empty_series() -> pd.DataFrame:
        return pd.DataFrame(columns=["date", "symbol", "market", "regime", "confidence", "stable_regime"])
