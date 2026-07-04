from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dashboard.loader import discover_output_root, discover_run_dates

st.set_page_config(page_title="Signal Generator Dashboard", page_icon="📈", layout="wide")


@st.cache_data(show_spinner=False, ttl=60)
def _sidebar_data() -> tuple[object, list[str]]:
    root = discover_output_root()
    dates = discover_run_dates(root)
    return root, dates


root, dates = _sidebar_data()

with st.sidebar:
    st.title("Signal Generator")
    st.caption("Professional EOD signal workspace")
    if dates:
        selected_date = st.selectbox(
            "Run Date",
            options=dates,
            index=0,
            key="run_date",
        )
    else:
        selected_date = st.text_input("Run Date", value="latest", key="run_date")
    st.caption(f"Data root: {root}")
    st.info("Every page reloads to the selected run date.")

if not dates:
    st.warning("No parquet run data was detected yet. Run the pipeline to populate the dashboard.")
