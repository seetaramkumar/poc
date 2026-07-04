from __future__ import annotations

from typing import Any

import pandas as pd
import plotly.graph_objects as go
import plotly.offline as po


class ReportCharts:
    """Generate embedded Plotly chart HTML snippets for the report."""

    @staticmethod
    def pie_chart(labels: list[str], values: list[float], title: str) -> str:
        fig = go.Figure(data=[go.Pie(labels=labels, values=values, hole=0.35)])
        fig.update_layout(title=title, template="plotly_dark")
        return po.plot(fig, output_type="div", include_plotlyjs="inline")

    @staticmethod
    def bar_chart(df: pd.DataFrame, x: str, y: str, title: str) -> str:
        fig = go.Figure(data=[go.Bar(x=df[x], y=df[y], text=df[y], marker_color="#4f8cff")])
        fig.update_layout(title=title, template="plotly_dark")
        return po.plot(fig, output_type="div", include_plotlyjs="inline")

    @staticmethod
    def histogram(series: pd.Series, title: str) -> str:
        fig = go.Figure(data=[go.Histogram(x=series.dropna(), marker_color="#4f8cff")])
        fig.update_layout(title=title, template="plotly_dark")
        return po.plot(fig, output_type="div", include_plotlyjs="inline")

    @staticmethod
    def line_chart(df: pd.DataFrame, x: str, y: str, title: str) -> str:
        fig = go.Figure(data=[go.Scatter(x=df[x], y=df[y], mode="lines+markers")])
        fig.update_layout(title=title, template="plotly_dark")
        return po.plot(fig, output_type="div", include_plotlyjs="inline")

    @staticmethod
    def heatmap(df: pd.DataFrame, title: str) -> str:
        values = df.set_index(df.columns[0]).T
        fig = go.Figure(data=[go.Heatmap(z=values.values, x=values.columns, y=values.index, colorscale="Viridis")])
        fig.update_layout(title=title, template="plotly_dark")
        return po.plot(fig, output_type="div", include_plotlyjs="inline")
