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
breadth_df = load_dataset(output_root, run_date, ["breadth"], ["*breadth*.parquet", "*summary*.parquet"])
historical_df = load_dataset(output_root, run_date, ["analytics", "historical_regimes"], ["*historical*.parquet", "*regime*.parquet"])

st.title("Market")
st.caption(f"Run date: {run_date}")

col1, col2, col3, col4 = st.columns(4)
with col1:
    metric_card("Market Regime", stable_df["stable_regime"].mode().iloc[0] if not stable_df.empty and "stable_regime" in stable_df.columns else "n/a")
with col2:
    metric_card("Confidence", round(stable_df["confidence"].mean(), 3) if not stable_df.empty and "confidence" in stable_df.columns else "n/a")
with col3:
    metric_card("Breadth Score", breadth_df.iloc[0].get("breadth_score", "n/a") if not breadth_df.empty else "n/a")
with col4:
    metric_card("Above EMA200", breadth_df.iloc[0].get("above_ema200", "n/a") if not breadth_df.empty else "n/a")

st.divider()

if not historical_df.empty and {"date", "regime"}.issubset(historical_df.columns):
    hist = historical_df.copy()
    hist["date"] = pd.to_datetime(hist["date"], errors="coerce")
    hist = hist.dropna(subset=["date"])
    summary = hist.groupby(["date", "regime"]).size().reset_index(name="count")
    fig = px.line(summary, x="date", y="count", color="regime", title="Historical Regime Timeline")
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

if not breadth_df.empty:
    metrics = [col for col in ["bull_pct", "bear_pct", "advance_decline", "thrust"] if col in breadth_df.columns]
    if metrics:
        sample = breadth_df.iloc[0]
        chart_df = pd.DataFrame({"Metric": metrics, "Value": [sample.get(col, 0) for col in metrics]})
        fig = px.bar(chart_df, x="Metric", y="Value", title="Breadth Components")
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
