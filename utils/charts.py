"""Chart styling and creation utilities."""

import plotly.express as px
from config import BRAND


def style_chart(fig, title: str, show_legend: bool = True, height: int = 400):
    """Apply consistent styling to Plotly charts."""
    fig.update_layout(
        title=dict(text=title, font=dict(size=16, color=BRAND["black"])),
        font=dict(family="Inter", size=12, color=BRAND["black"]),
        plot_bgcolor=BRAND["white"],
        paper_bgcolor=BRAND["white"],
        showlegend=show_legend,
        height=height,
        margin=dict(l=50, r=50, t=60, b=50),
        xaxis=dict(gridcolor="#E2E6EA", linecolor="#E2E6EA"),
        yaxis=dict(gridcolor="#E2E6EA", linecolor="#E2E6EA"),
    )
    return fig


def safe_chart(chart_func, *args, empty_message: str = "No data available for the selected filters.", **kwargs):
    """Safely create a chart with error handling."""
    try:
        fig = chart_func(*args, **kwargs)
        if fig is not None:
            return fig
        else:
            return None
    except Exception as e:
        import streamlit as st
        st.error(f"Error creating chart: {str(e)}")
        return None
