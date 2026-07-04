from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dashboard.loader import discover_output_root, discover_run_dates, load_dataset
from dashboard.utils import metric_card, render_table

output_root = discover_output_root()
run_dates = discover_run_dates(output_root)
run_date = st.session_state.get("run_date") or (run_dates[0] if run_dates else "latest")

filter_df = load_dataset(output_root, run_date, ["filters"], ["*filter*.parquet", "*rejected*.parquet"])
quality_df = load_dataset(output_root, run_date, ["quality"], ["*quality*.parquet", "*score*.parquet"])

st.title("System Health")
st.caption(f"Run date: {run_date}")

root = Path(output_root)
run_dir = root / run_date
parquet_files = list(run_dir.rglob("*.parquet")) if run_dir.exists() else []

col1, col2, col3, col4 = st.columns(4)
with col1:
    metric_card("Parquet Files", len(parquet_files))
with col2:
    metric_card("Accepted Stocks", len(filter_df) if not filter_df.empty else "n/a")
with col3:
    metric_card("Quality Records", len(quality_df) if not quality_df.empty else "n/a")
with col4:
    metric_card("Data Root", root)

if parquet_files:
    file_rows = []
    for path in parquet_files:
        file_rows.append({"file": str(path.relative_to(root)), "size_mb": round(path.stat().st_size / (1024 * 1024), 2)})
    render_table(pd.DataFrame(file_rows), title="File Inventory", key_prefix="health_files")

if not filter_df.empty:
    render_table(filter_df, title="Filter Diagnostics", key_prefix="health_filters")
