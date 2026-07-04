from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dashboard.loader import discover_output_root, discover_run_dates, load_dataset
from dashboard.utils import make_bar_chart, metric_card, render_table


st.set_page_config(layout="wide")

output_root = discover_output_root()
run_dates = discover_run_dates(output_root)
run_date = st.session_state.get("run_date") or (run_dates[0] if run_dates else "latest")

stable_df = load_dataset(output_root, run_date, ["stable_classifications", "classifications"], ["*_stable.parquet", "*stable*.parquet", "*classification*.parquet"])
routing_df = load_dataset(output_root, run_date, ["router", "routing"], ["*routing*.parquet", "*router*.parquet", "*decision*.parquet"])
quality_df = load_dataset(output_root, run_date, ["quality"], ["*quality*.parquet", "*score*.parquet"])
breadth_df = load_dataset(output_root, run_date, ["breadth"], ["*breadth*.parquet", "*summary*.parquet"])
sector_df = load_dataset(output_root, run_date, ["sectors", "sector"], ["*sector*.parquet", "*sector_rank*.parquet"])

st.title("Executive Summary")
st.caption(f"Current run date: {run_date}")

if stable_df.empty and routing_df.empty and quality_df.empty and breadth_df.empty:
    st.info("Run the pipeline to populate the dashboard with parquet outputs.")
    st.stop()

col1, col2, col3, col4 = st.columns(4)
with col1:
    metric_card("Market Regime", stable_df["stable_regime"].mode().iloc[0] if not stable_df.empty and "stable_regime" in stable_df.columns else "n/a")
with col2:
    metric_card("Confidence", round(stable_df["confidence"].mean(), 3) if not stable_df.empty and "confidence" in stable_df.columns else "n/a")
with col3:
    metric_card("Tradable Stocks", int(routing_df.shape[0]) if not routing_df.empty else "n/a")
with col4:
    metric_card("Quality Stocks", int(quality_df.shape[0]) if not quality_df.empty else "n/a")

st.divider()

left, right = st.columns(2)
with left:
    if not breadth_df.empty:
        metrics = [col for col in ["breadth_score", "bull_pct", "bear_pct", "advance_decline"] if col in breadth_df.columns]
        if metrics:
            sample = breadth_df.iloc[0]
            chart_df = pd.DataFrame({"Metric": metrics, "Value": [sample.get(col, 0) for col in metrics]})
            fig = px.bar(chart_df, x="Metric", y="Value", text="Value", title="Breadth Snapshot")
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
with right:
    if not sector_df.empty:
        sector_plot_df = sector_df.head(10).copy()
        if "sector" in sector_plot_df.columns and "score" in sector_plot_df.columns:
            fig = px.bar(sector_plot_df, x="sector", y="score", text="score", title="Top Sectors")
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

st.divider()

if not routing_df.empty:
    strategy_counts = routing_df["strategy"].fillna("No Trade").value_counts().reset_index()
    strategy_counts.columns = ["strategy", "count"]
    make_bar_chart(strategy_counts, x="strategy", y="count", title="Strategy Allocation")

if not quality_df.empty:
    render_table(quality_df.head(50), title="Quality Snapshot", key_prefix="home_quality")
