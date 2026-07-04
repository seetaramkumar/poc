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
from dashboard.utils import render_table

output_root = discover_output_root()
run_dates = discover_run_dates(output_root)
run_date = st.session_state.get("run_date") or (run_dates[0] if run_dates else "latest")

sector_df = load_dataset(output_root, run_date, ["sectors", "sector"], ["*sector*.parquet", "*sector_rank*.parquet"])

st.title("Sectors")
st.caption(f"Run date: {run_date}")

if sector_df.empty:
    st.info("Sector parquet files are not available for this run date.")
    st.stop()

plot_df = sector_df.copy()
if "score" not in plot_df.columns and "sector_score" in plot_df.columns:
    plot_df = plot_df.rename(columns={"sector_score": "score"})

if "sector" in plot_df.columns and "score" in plot_df.columns:
    fig = px.bar(plot_df.head(15), x="sector", y="score", text="score", title="Sector Leaderboard")
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

heatmap_cols = [col for col in ["strength", "momentum", "quality", "score"] if col in plot_df.columns]
if heatmap_cols:
    heatmap_df = plot_df[["sector"] + heatmap_cols].dropna().head(20)
    fig = px.imshow(heatmap_df.set_index("sector")[heatmap_cols].T, title="Sector Heatmap")
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

render_table(plot_df, title="Sector Metrics", key_prefix="sectors")
