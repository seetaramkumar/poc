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

router_df = load_dataset(output_root, run_date, ["router", "routing"], ["*routing*.parquet", "*router*.parquet", "*decision*.parquet"])

st.title("Strategy Router")
st.caption(f"Run date: {run_date}")

if router_df.empty:
    st.info("Router parquet output is not available for this run date.")
    st.stop()

strategy_counts = router_df["strategy"].fillna("No Trade").value_counts().reset_index()
strategy_counts.columns = ["strategy", "count"]
fig = px.pie(strategy_counts, names="strategy", values="count", title="Strategy Allocation")
st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

selected_strategy = st.selectbox("Filter strategy", options=sorted(strategy_counts["strategy"].tolist()))
filtered = router_df[router_df["strategy"].fillna("No Trade") == selected_strategy]
render_table(filtered, title=f"Stocks assigned to {selected_strategy}", key_prefix="router")
