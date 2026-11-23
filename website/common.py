"""
Shared Streamlit helpers. The dashboard reuses the API's own modules
(backend/resources.py, backend/predictor.py, backend/rail_features.py), so it
shows exactly the same data, model and metrics as the React site.
"""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "backend"))

from resources import Resources, load_resources  # noqa: E402

PLOT_LAYOUT = dict(
    paper_bgcolor="#161B22",
    plot_bgcolor="#0D1117",
    font=dict(family="Inter, sans-serif", color="#8B949E", size=12),
    margin=dict(l=40, r=20, t=30, b=40),
    hoverlabel=dict(bgcolor="#161B22", bordercolor="#30363D", font=dict(color="#E6EDF3", size=12)),
    xaxis=dict(gridcolor="#30363D", zerolinecolor="#30363D"),
    yaxis=dict(gridcolor="#30363D", zerolinecolor="#30363D"),
)
COLORS = ["#C8102E", "#1A3A5C", "#2E8B57", "#D29922", "#8957E5", "#2E90FA"]


def layout(**overrides) -> dict:
    """PLOT_LAYOUT with overrides (e.g. margin) replacing the defaults."""
    return {**PLOT_LAYOUT, **overrides}
GITHUB_URL = "https://github.com/pranay-ratan/rail-delay-prediction"


@st.cache_resource(show_spinner="Loading data and model...")
def resources() -> Resources:
    loaded = load_resources()
    if loaded.incidents is None or loaded.model is None:
        st.error("Data or model files are missing from backend/. Run `python scripts/sync_backend.py`.")
        st.stop()
    return loaded


def setup_page(title: str) -> Resources:
    """Page config, theme CSS and sidebar; returns the loaded resources."""
    st.set_page_config(page_title=f"{title} | Rail Risk Intelligence", layout="wide")
    css = Path(__file__).parent / "assets" / "theme.css"
    if css.exists():
        st.markdown(f"<style>{css.read_text()}</style>", unsafe_allow_html=True)
    res = resources()
    with st.sidebar:
        st.markdown("### Rail Risk Intelligence")
        st.caption("Severity of Canadian rail accidents, 2001-2025")
        st.markdown(f"**Model:** {res.summary.get('final_model', '?')}")
        st.markdown(
            "**Data:** [TSB Rail Occurrence Database]"
            "(https://open.canada.ca/data/en/dataset/4f8d9b0d-c4af-4be2-b751-3f5a403993ff), "
            "[Open-Meteo](https://open-meteo.com) weather"
        )
        st.markdown(f"[Source code]({GITHUB_URL}) · Pranay Ratan, SFU")
    return res
