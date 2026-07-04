from __future__ import annotations

import fnmatch
import sys
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from reporting.report_models import ReportData


class ReportDataLoader:
    """Load the parquet artifacts produced for a selected run date."""

    def __init__(self, output_root: str | Path | None = None) -> None:
        self.output_root = Path(output_root or PROJECT_ROOT / "runner" / "output").resolve()

    def load(self, run_date: str) -> ReportData:
        run_root = self.output_root
        data = ReportData(run_date=run_date, output_root=run_root)
        data.parquet_files = self._inventory_parquet_files(run_date)

        data.classifications = self._load_dataset(run_date, ["classifications"], ["*classification*.parquet", "*classifications*.parquet"])
        data.indicators = self._load_dataset(run_date, ["indicators"], ["*indicator*.parquet", "*indicators*.parquet"])
        data.signals = self._load_dataset(run_date, ["signals"], ["*signal*.parquet", "*signals*.parquet"])
        data.breadth = self._load_dataset(run_date, ["breadth"], ["*breadth*.parquet", "*summary*.parquet"])
        data.quality = self._load_dataset(run_date, ["quality"], ["*quality*.parquet", "*score*.parquet"])
        data.router = self._load_dataset(run_date, ["router", "routing"], ["*router*.parquet", "*routing*.parquet", "*decision*.parquet"])
        data.rankings = self._load_dataset(run_date, ["rankings", "opportunities"], ["*ranking*.parquet", "*opportunity*.parquet", "*rank*.parquet"])
        data.analytics = self._load_dataset(run_date, ["analytics"], ["*analytics*.parquet", "*transition*.parquet", "*current*.parquet", "*duration*.parquet", "*stability*.parquet", "*oscillation*.parquet"])
        data.sector_metrics = self._load_dataset(run_date, ["sectors", "sector"], ["*sector*.parquet", "*metric*.parquet"])
        data.sector_rankings = self._load_dataset(run_date, ["sectors", "sector"], ["*ranking*.parquet", "*rank*.parquet"])
        data.sector_states = self._load_dataset(run_date, ["sectors", "sector"], ["*state*.parquet", "*states*.parquet"])
        data.historical_regime = self._load_dataset(run_date, ["analytics", "historical_regimes"], ["*historical*.parquet", "*regime*.parquet"])
        data.score_distributions = self._load_dataset(run_date, ["scoring"], ["*score*.parquet", "*distribution*.parquet"])
        data.stable_classifications = self._load_dataset(run_date, ["stable_classifications"], ["*stable*.parquet"])
        data.filter_summaries = self._load_dataset(run_date, ["filters"], ["*filter*.parquet", "*summary*.parquet"])
        data.rejected_symbols = self._load_dataset(run_date, ["filters"], ["*rejected*.parquet"])
        return data

    def _inventory_parquet_files(self, run_date: str) -> list[dict[str, Any]]:
        if not self.output_root.exists():
            return []
        rows: list[dict[str, Any]] = []
        for path in self.output_root.rglob("*.parquet"):
            if run_date not in path.parts:
                continue
            rows.append({
                "path": str(path),
                "name": path.name,
                "size_mb": round(path.stat().st_size / (1024 * 1024), 2),
                "relative_path": str(path.relative_to(self.output_root)),
            })
        rows.sort(key=lambda item: item["relative_path"])
        return rows

    def _load_dataset(self, run_date: str, category_candidates: Iterable[str], filename_patterns: Iterable[str]) -> pd.DataFrame:
        root = self.output_root
        category_names = {c.lower() for c in category_candidates}
        filename_globs = [pattern.lower() for pattern in filename_patterns]
        candidate_paths: list[Path] = []

        for base in [root / run_date, root]:
            if not base.exists():
                continue
            for path in base.rglob("*.parquet"):
                if run_date not in path.parts:
                    continue
                parts = [part.lower() for part in path.parts]
                if not any(part in category_names for part in parts):
                    continue
                if any(fnmatch.fnmatch(path.name.lower(), pattern) for pattern in filename_globs):
                    candidate_paths.append(path)

        if not candidate_paths:
            return pd.DataFrame()
        candidate_paths.sort(key=lambda item: (len(item.parts), str(item)))
        try:
            return pd.read_parquet(candidate_paths[0])
        except Exception:
            return pd.DataFrame()
