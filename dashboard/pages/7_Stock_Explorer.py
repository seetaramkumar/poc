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
from dashboard.utils import metric_card

output_root = discover_output_root()
run_dates = discover_run_dates(output_root)
run_date = st.session_state.get("run_date") or (run_dates[0] if run_dates else "latest")

stable_df = load_dataset(output_root, run_date, ["stable_classifications", "classifications"], ["*_stable.parquet", "*stable*.parquet", "*classification*.parquet"])
routing_df = load_dataset(output_root, run_date, ["router", "routing"], ["*routing*.parquet", "*router*.parquet", "*decision*.parquet"])
quality_df = load_dataset(output_root, run_date, ["quality"], ["*quality*.parquet", "*score*.parquet"])
historical_df = load_dataset(output_root, run_date, ["analytics", "historical_regimes"], ["*historical*.parquet", "*regime*.parquet"])

st.title("Stock Explorer")
st.caption(f"Run date: {run_date}")

symbols = sorted({str(s) for s in stable_df["symbol"].dropna().astype(str)}) if not stable_df.empty else []
selected_symbol = st.text_input("Search symbol", value=st.session_state.get("selected_symbol", ""))
if not selected_symbol:
    selected_symbol = st.selectbox("Select symbol", options=symbols, index=0) if symbols else ""

if not selected_symbol:
    st.info("No symbols available yet.")
    st.stop()

row = stable_df[stable_df["symbol"].astype(str) == selected_symbol]
if row.empty:
    st.info(f"Symbol {selected_symbol} was not found for this run date.")
    st.stop()

record = row.iloc[0]

col1, col2, col3, col4 = st.columns(4)
with col1:
    metric_card("Current Regime", record.get("stable_regime", record.get("raw_regime", "n/a")))
with col2:
    metric_card("Confidence", round(float(record.get("confidence", 0.0)), 3))
with col3:
    metric_card("Router Decision", routing_df.loc[routing_df["symbol"].astype(str) == selected_symbol, "strategy"].iloc[0] if not routing_df.empty and "strategy" in routing_df.columns and routing_df["symbol"].astype(str).eq(selected_symbol).any() else "n/a")
with col4:
    metric_card("Quality", round(float(quality_df.loc[quality_df["symbol"].astype(str) == selected_symbol, "quality_score"].iloc[0]), 3) if not quality_df.empty and "quality_score" in quality_df.columns and quality_df["symbol"].astype(str).eq(selected_symbol).any() else "n/a")

st.divider()

with st.expander("Signals and Indicators"):
    detail_cols = [col for col in row.columns if col not in {"symbol", "market", "run_date", "raw_regime", "stable_regime", "prior_stable_regime"}]
    st.dataframe(row[detail_cols], use_container_width=True)

with st.expander("Historical Regime Timeline"):
    if not historical_df.empty and {"date", "symbol", "regime"}.issubset(historical_df.columns):
        hist = historical_df[historical_df["symbol"].astype(str) == selected_symbol].copy()
        if not hist.empty:
            hist["date"] = pd.to_datetime(hist["date"], errors="coerce")
            fig = px.line(hist, x="date", y="confidence", color="regime", title=f"{selected_symbol} regime evolution")
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
        else:
            st.info("Historical regime history is not available for this symbol.")

with st.expander("Why selected / rejected"):
    st.write("Selection rationale is drawn from the persisted ranking and quality parquet output when available.")
    st.write("If the pipeline emits additional notes or flags, this panel can be expanded with those fields.")
