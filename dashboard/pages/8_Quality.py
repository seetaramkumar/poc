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

quality_df = load_dataset(output_root, run_date, ["quality"], ["*quality*.parquet", "*score*.parquet"])

st.title("Quality")
st.caption(f"Run date: {run_date}")

if quality_df.empty:
    st.info("Quality parquet output is not available for this run date.")
    st.stop()

score_col = "quality_score" if "quality_score" in quality_df.columns else next((c for c in quality_df.columns if "score" in c.lower()), None)
if score_col is None:
    st.info("Quality score column was not found in the parquet output.")
    st.stop()

quality_df = quality_df.copy()
quality_df[score_col] = pd.to_numeric(quality_df[score_col], errors="coerce")
fig = px.histogram(quality_df, x=score_col, nbins=20, title="Quality Score Distribution")
st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

col1, col2 = st.columns(2)
with col1:
    render_table(quality_df.nlargest(20, score_col), title="Top Quality Stocks", key_prefix="quality_top")
with col2:
    render_table(quality_df.nsmallest(20, score_col), title="Lowest Quality Stocks", key_prefix="quality_bottom")
