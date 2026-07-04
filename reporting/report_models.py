from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd


@dataclass
class ReportData:
    run_date: str
    output_root: Path
    classifications: pd.DataFrame = field(default_factory=pd.DataFrame)
    indicators: pd.DataFrame = field(default_factory=pd.DataFrame)
    signals: pd.DataFrame = field(default_factory=pd.DataFrame)
    breadth: pd.DataFrame = field(default_factory=pd.DataFrame)
    quality: pd.DataFrame = field(default_factory=pd.DataFrame)
    router: pd.DataFrame = field(default_factory=pd.DataFrame)
    rankings: pd.DataFrame = field(default_factory=pd.DataFrame)
    analytics: pd.DataFrame = field(default_factory=pd.DataFrame)
    sector_metrics: pd.DataFrame = field(default_factory=pd.DataFrame)
    sector_rankings: pd.DataFrame = field(default_factory=pd.DataFrame)
    sector_states: pd.DataFrame = field(default_factory=pd.DataFrame)
    historical_regime: pd.DataFrame = field(default_factory=pd.DataFrame)
    score_distributions: pd.DataFrame = field(default_factory=pd.DataFrame)
    stable_classifications: pd.DataFrame = field(default_factory=pd.DataFrame)
    filter_summaries: pd.DataFrame = field(default_factory=pd.DataFrame)
    rejected_symbols: pd.DataFrame = field(default_factory=pd.DataFrame)
    parquet_files: list[dict[str, Any]] = field(default_factory=list)

    def is_empty(self) -> bool:
        return all(
            df.empty for df in [
                self.classifications,
                self.indicators,
                self.signals,
                self.breadth,
                self.quality,
                self.router,
                self.rankings,
                self.analytics,
                self.sector_metrics,
                self.sector_rankings,
                self.sector_states,
                self.historical_regime,
                self.score_distributions,
                self.stable_classifications,
                self.filter_summaries,
                self.rejected_symbols,
            ]
        )
