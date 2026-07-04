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
from dashboard.utils import format_pct, metric_card

output_root = discover_output_root()
run_dates = discover_run_dates(output_root)
run_date = st.session_state.get("run_date") or (run_dates[0] if run_dates else "latest")

breadth_df = load_dataset(output_root, run_date, ["breadth"], ["*breadth*.parquet", "*summary*.parquet"])

st.title("Breadth")
st.caption(f"Run date: {run_date}")

if breadth_df.empty:
    st.info("Breadth parquet output is not available for this run date.")
    st.stop()

row = breadth_df.iloc[0]
col1, col2, col3, col4 = st.columns(4)
with col1:
    metric_card("Breadth Score", row.get("breadth_score", "n/a"))
with col2:
    metric_card("Bull %", format_pct(row.get("bull_pct", 0)) if "bull_pct" in row else "n/a")
with col3:
    metric_card("Bear %", format_pct(row.get("bear_pct", 0)) if "bear_pct" in row else "n/a")
with col4:
    metric_card("Advance / Decline", row.get("advance_decline", "n/a"))

st.divider()

metrics = [col for col in ["breadth_score", "bull_pct", "bear_pct", "advance_decline", "thrust", "above_ema200"] if col in breadth_df.columns]
if metrics:
    fig = px.bar(
        pd.DataFrame({"Metric": metrics, "Value": [row.get(col, 0) for col in metrics]}),
        x="Metric",
        y="Value",
        text="Value",
        title="Breadth Metrics",
    )
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

st.subheader("What the metrics mean")
st.markdown(
    "- Breadth Score: overall participation strength.\n"
    "- Bull % / Bear %: proportion of advancing versus declining names.\n"
    "- Advance / Decline: net participation over the session.\n"
    "- Thrust: velocity of participation.\n"
    "- Above EMA200: how many names are above the 200-day average."
)
