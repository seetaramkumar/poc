from __future__ import annotations

import html
import sys
from pathlib import Path
from typing import Any

import pandas as pd
from jinja2 import Environment, FileSystemLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from reporting.charts import ReportCharts
from reporting.report_models import ReportData


class MarketReportGenerator:
    """Render a polished market intelligence HTML report from ReportData."""

    def __init__(self, output_dir: str | Path | None = None) -> None:
        self.output_dir = Path(output_dir or PROJECT_ROOT / "output" / "reports").resolve()
        self.template_dir = Path(__file__).resolve().parent / "templates"
        self.static_dir = Path(__file__).resolve().parent / "static"
        self.env = Environment(loader=FileSystemLoader(self.template_dir), autoescape=True)

    def generate(self, data: ReportData) -> Path:
        report_dir = self.output_dir / data.run_date
        report_dir.mkdir(parents=True, exist_ok=True)
        html_path = report_dir / "market_report.html"

        context = self._build_context(data)
        template = self.env.get_template("report.html")
        html_content = template.render(**context)
        html_path.write_text(html_content, encoding="utf-8")
        return html_path

    def _build_context(self, data: ReportData) -> dict[str, Any]:
        summary = self._build_summary(data)
        regime_distribution = self._build_regime_distribution(data)
        sector_table = self._build_sector_table(data)
        strategy_breakdown = self._build_strategy_breakdown(data)
        opportunity_table = self._build_opportunity_table(data)
        quality_summary = self._build_quality_summary(data)
        system_health = self._build_system_health(data)
        return {
            "run_date": data.run_date,
            "summary": summary,
            "regime_distribution": regime_distribution,
            "sector_table": sector_table,
            "strategy_breakdown": strategy_breakdown,
            "opportunity_table": opportunity_table,
            "quality_summary": quality_summary,
            "system_health": system_health,
            "breadth_chart": ReportCharts.pie_chart(["Bull", "Bear"], [50, 50], "Breadth Mix"),
            "regime_chart": ReportCharts.pie_chart(list(regime_distribution.keys()), list(regime_distribution.values()), "Regime Distribution"),
            "quality_hist": ReportCharts.histogram(pd.Series(quality_summary.get("scores", [])), "Quality Distribution"),
            "sector_chart": ReportCharts.bar_chart(sector_table.head(10), "sector", "score", "Top Sectors"),
            "opportunity_chart": ReportCharts.bar_chart(opportunity_table.head(10), "ticker", "score", "Top Opportunities"),
            "parquet_files": data.parquet_files,
            "style_css": (self.static_dir / "style.css").read_text(encoding="utf-8"),
        }

    @staticmethod
    def _build_summary(data: ReportData) -> dict[str, Any]:
        stable = data.stable_classifications
        quality = data.quality
        breadth = data.breadth
        router = data.router
        regime_value = stable["stable_regime"].mode().iloc[0] if not stable.empty and "stable_regime" in stable.columns else "N/A"
        confidence = round(float(stable["confidence"].mean()), 3) if not stable.empty and "confidence" in stable.columns else 0.0
        breadth_score = float(breadth.iloc[0]["breadth_score"]) if not breadth.empty and "breadth_score" in breadth.columns else 0.0
        bull_pct = float(breadth.iloc[0]["bull_pct"]) if not breadth.empty and "bull_pct" in breadth.columns else 0.0
        bear_pct = float(breadth.iloc[0]["bear_pct"]) if not breadth.empty and "bear_pct" in breadth.columns else 0.0
        above_ema = float(breadth.iloc[0]["above_ema200"]) if not breadth.empty and "above_ema200" in breadth.columns else 0.0
        adv_decl = float(breadth.iloc[0]["advance_decline"]) if not breadth.empty and "advance_decline" in breadth.columns else 0.0
        quality_score = float(quality["quality_score"].mean()) if not quality.empty and "quality_score" in quality.columns else 0.0
        top_strategy = router["strategy"].mode().iloc[0] if not router.empty and "strategy" in router.columns else "N/A"
        return {
            "run_date": data.run_date,
            "market_regime": regime_value,
            "confidence": confidence,
            "breadth_state": "Bullish" if breadth_score > 0 else "Neutral",
            "bull_pct": bull_pct,
            "bear_pct": bear_pct,
            "above_ema200": above_ema,
            "advance_decline": adv_decl,
            "accepted": len(stable),
            "rejected": 0,
            "tradable": len(router),
            "top_sector": "N/A",
            "top_strategy": top_strategy,
            "market_health": round(quality_score, 2),
            "quality_score": quality_score,
        }

    @staticmethod
    def _build_regime_distribution(data: ReportData) -> dict[str, int]:
        if data.stable_classifications.empty or "stable_regime" not in data.stable_classifications.columns:
            return {"UNCERTAIN": 1}
        counts = data.stable_classifications["stable_regime"].value_counts().to_dict()
        return {k: int(v) for k, v in counts.items()}

    @staticmethod
    def _build_sector_table(data: ReportData) -> pd.DataFrame:
        if not data.sector_metrics.empty:
            return data.sector_metrics.head(20)
        if not data.sector_rankings.empty:
            return data.sector_rankings.head(20)
        return pd.DataFrame(columns=["sector", "score"])

    @staticmethod
    def _build_strategy_breakdown(data: ReportData) -> pd.DataFrame:
        if data.router.empty or "strategy" not in data.router.columns:
            return pd.DataFrame(columns=["strategy", "count"])
        counts = data.router["strategy"].fillna("No Trade").value_counts().reset_index()
        counts.columns = ["strategy", "count"]
        return counts

    @staticmethod
    def _build_opportunity_table(data: ReportData) -> pd.DataFrame:
        if data.rankings.empty:
            return pd.DataFrame(columns=["ticker", "score"])
        df = data.rankings.copy()
        if "ticker" not in df.columns and "symbol" in df.columns:
            df = df.rename(columns={"symbol": "ticker"})
        if "score" not in df.columns:
            score_candidates = [c for c in df.columns if "score" in c.lower()]
            if score_candidates:
                df = df.rename(columns={score_candidates[0]: "score"})
        df = df[[c for c in ["ticker", "score", "strategy", "sector"] if c in df.columns]]
        return df.head(50)

    @staticmethod
    def _build_quality_summary(data: ReportData) -> dict[str, Any]:
        quality = data.quality
        if quality.empty or "quality_score" not in quality.columns:
            return {"scores": [], "average": 0.0, "median": 0.0}
        scores = pd.to_numeric(quality["quality_score"], errors="coerce").dropna()
        return {"scores": scores.tolist(), "average": round(float(scores.mean()), 2), "median": round(float(scores.median()), 2)}

    @staticmethod
    def _build_system_health(data: ReportData) -> dict[str, Any]:
        return {
            "parquet_files": len(data.parquet_files),
            "file_sizes": sum(item["size_mb"] for item in data.parquet_files),
            "warnings": 0,
            "filters": len(data.filter_summaries),
            "rejections": len(data.rejected_symbols),
        }
