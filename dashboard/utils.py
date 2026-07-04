from __future__ import annotations

from typing import Iterable

import pandas as pd
import plotly.express as px
import streamlit as st


def metric_card(label: str, value: object, delta: object | None = None, help_text: str | None = None) -> None:
    st.metric(label=label, value=value, delta=delta, help=help_text)


def format_pct(value: object, precision: int = 2) -> str:
    try:
        return f"{float(value) * 100:.{precision}f}%"
    except Exception:
        return "n/a"


def render_table(df: pd.DataFrame, *, title: str | None = None, key_prefix: str = "table") -> None:
    if df.empty:
        st.info("No data available for this view.")
        return

    if title:
        st.subheader(title)

    search = st.text_input("Search", key=f"{key_prefix}_search")
    cols = [c for c in df.columns if c not in {"date", "run_date"}]
    sort_col = st.selectbox("Sort by", options=cols, key=f"{key_prefix}_sort")
    ascending = st.checkbox("Ascending", value=False, key=f"{key_prefix}_ascending")
    page_size = st.selectbox("Rows per page", [10, 25, 50, 100], key=f"{key_prefix}_page_size")

    filtered = df.copy()
    if search:
        mask = filtered.astype(str).apply(
            lambda col: col.str.contains(search, case=False, na=False)
        ).any(axis=1)
        filtered = filtered.loc[mask]

    if sort_col in filtered.columns:
        try:
            filtered = filtered.sort_values(sort_col, ascending=ascending, na_position="last")
        except Exception:
            filtered = filtered.sort_values(sort_col, ascending=ascending, kind="mergesort")

    total_pages = max(1, (len(filtered) + page_size - 1) // page_size)
    page = st.number_input("Page", min_value=1, max_value=total_pages, value=1, key=f"{key_prefix}_page")
    start = (page - 1) * page_size
    end = start + page_size
    table = filtered.iloc[start:end]

    st.dataframe(table, use_container_width=True)

    csv = table.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Download CSV",
        data=csv,
        file_name=f"{key_prefix}.csv",
        mime="text/csv",
        key=f"{key_prefix}_download",
    )


def ensure_numeric(df: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    result = df.copy()
    for column in columns:
        if column in result.columns:
            result[column] = pd.to_numeric(result[column], errors="coerce")
    return result


def numeric_series(df: pd.DataFrame, column: str) -> pd.Series:
    return pd.to_numeric(df[column], errors="coerce").fillna(0.0)


def make_bar_chart(df: pd.DataFrame, x: str, y: str, title: str) -> None:
    if df.empty or x not in df.columns or y not in df.columns:
        st.info("Chart data is not available yet.")
        return
    fig = px.bar(df, x=x, y=y, text=y, title=title)
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})


def make_line_chart(df: pd.DataFrame, x: str, y: str, title: str) -> None:
    if df.empty or x not in df.columns or y not in df.columns:
        st.info("Chart data is not available yet.")
        return
    fig = px.line(df, x=x, y=y, markers=True, title=title)
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
