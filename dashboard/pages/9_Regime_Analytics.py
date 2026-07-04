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

oscillations_df = load_dataset(output_root, run_date, ["analytics"], ["*oscillation*.parquet", "*oscillator*.parquet"])
top_osc_df = load_dataset(output_root, run_date, ["analytics"], ["*top*.parquet", "*oscillator*.parquet"])
stability_df = load_dataset(output_root, run_date, ["analytics"], ["*stability*.parquet", "*regime*.parquet"])
transitions_df = load_dataset(output_root, run_date, ["analytics"], ["*transition*.parquet", "*matrix*.parquet"])
historical_df = load_dataset(output_root, run_date, ["analytics", "historical_regimes"], ["*historical*.parquet", "*regime*.parquet"])

st.title("Regime Analytics")
st.caption(f"Run date: {run_date}")

if historical_df.empty and oscillations_df.empty:
    st.info("Regime analytics parquet files are not available for this run date.")
    st.stop()

if not historical_df.empty and {"date", "regime"}.issubset(historical_df.columns):
    hist = historical_df.copy()
    hist["date"] = pd.to_datetime(hist["date"], errors="coerce")
    summary = hist.groupby(["date", "regime"]).size().reset_index(name="count")
    fig = px.line(summary, x="date", y="count", color="regime", title="Historical Regime Evolution")
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

if not oscillations_df.empty:
    render_table(oscillations_df, title="Oscillation Analysis", key_prefix="regime_osc")
if not top_osc_df.empty:
    render_table(top_osc_df, title="Top Oscillators", key_prefix="regime_top_osc")
if not transitions_df.empty:
    render_table(transitions_df, title="Transition Matrix", key_prefix="regime_transitions")
if not stability_df.empty:
    render_table(stability_df, title="Regime Stability", key_prefix="regime_stability")
