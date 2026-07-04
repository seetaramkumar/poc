from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dashboard.loader import discover_output_root, discover_run_dates, load_dataset
from dashboard.utils import render_table

output_root = discover_output_root()
run_dates = discover_run_dates(output_root)
run_date = st.session_state.get("run_date") or (run_dates[0] if run_dates else "latest")

ranking_df = load_dataset(output_root, run_date, ["rankings", "opportunities"], ["*ranking*.parquet", "*opportunity*.parquet", "*rank*.parquet"])

st.title("Opportunities")
st.caption(f"Run date: {run_date}")

if ranking_df.empty:
    st.info("Opportunity rankings parquet output is not available for this run date.")
    st.stop()

if "symbol" in ranking_df.columns:
    ranking_df = ranking_df.copy()
    ranking_df["symbol"] = ranking_df["symbol"].astype(str)

if "score" not in ranking_df.columns and "rank_score" in ranking_df.columns:
    ranking_df = ranking_df.rename(columns={"rank_score": "score"})

if "strategy" not in ranking_df.columns and "router_decision" in ranking_df.columns:
    ranking_df = ranking_df.rename(columns={"router_decision": "strategy"})

for column in ["score", "quality", "trend", "momentum", "sector", "breadth_alignment"]:
    if column in ranking_df.columns:
        ranking_df[column] = pd.to_numeric(ranking_df[column], errors="coerce")

if "symbol" in ranking_df.columns:
    st.button(
        "Open selected symbol in explorer",
        on_click=lambda: st.session_state.update({"selected_symbol": ranking_df.iloc[0]["symbol"]}) if "symbol" in ranking_df.columns else None,
    )

render_table(ranking_df, title="Top Opportunities", key_prefix="opportunities")
