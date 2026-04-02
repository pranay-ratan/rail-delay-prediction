"""
app.py — Streamlit Multi-Page App Entry Point
Rail Risk Intelligence Dashboard
Author: Pranay Ratan | SFU Data Science
"""

import streamlit as st
from pathlib import Path

st.set_page_config(
    page_title="Rail Risk Intelligence",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "Get Help": "https://github.com/pranay-ratan",
        "About": "Canadian Rail Incident & Delay Prediction | Pranay Ratan",
    },
)

css_path = Path(__file__).parent / "assets" / "theme.css"
if css_path.exists():
    with open(css_path) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

with st.sidebar:
    st.markdown("""
    <div style="text-align:center; padding: 1rem 0 0.5rem 0;">
        <div style="
            width:40px; height:40px; border-radius:10px;
            background: linear-gradient(135deg, #C8102E, #8B0000);
            display:inline-flex; align-items:center; justify-content:center;
            font-size:14px; font-weight:800; color:#fff; margin-bottom:8px;
        ">RR</div>
        <div style="font-size:1.05rem; font-weight:700; letter-spacing:-0.3px; color:#E6EDF3;">
            Rail Risk Intelligence
        </div>
        <div style="font-size:0.72rem; color:#6E7681; margin-top:0.2rem;">
            Rail Incident Analytics Platform
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("""
    **Navigation**
    - Overview — Project summary and key statistics
    - Deep Dive Explorer — Interactive data exploration
    - Model Command Center — Performance and interpretation
    - Live Risk Assessor — Real-time risk scoring
    """)

    st.markdown("---")
    st.markdown("""
    **Author**  
    Pranay Ratan  
    BSc Data Science, SFU
    """)

    st.markdown("---")
    st.markdown("""
    **Tech Stack**  
    `Python` `scikit-learn` `XGBoost`  
    `LightGBM` `SHAP` `Streamlit` `Plotly`
    """)

    st.markdown("---")
    st.markdown(
        "<div style='font-size:0.68rem; color:#6E7681; text-align:center'>"
        "Model: Random Forest v1.0<br>"
        "<span style='display:inline-block;width:6px;height:6px;background:#2E8B57;"
        "border-radius:50%;margin-right:4px;vertical-align:middle;'></span>Online"
        "</div>",
        unsafe_allow_html=True,
    )

st.markdown("""
<div class="hero-banner">
    <h1>Canadian Rail Incident Severity Prediction</h1>
    <p>
        An end-to-end machine learning system for predicting incident severity across 
        the Canadian rail network — built with 25 years of Transport Canada 
        railway occurrence data to deliver proactive, data-driven risk intelligence.
    </p>
</div>
""", unsafe_allow_html=True)

st.info("Select a page from the sidebar to explore the project.")

col1, col2, col3, col4 = st.columns(4)
pages = [
    ("01", "Project Overview", "Key stats, motivation, dataset"),
    ("02", "Deep Dive Explorer", "Interactive charts and insights"),
    ("03", "Model Command Center", "Accuracy, ROC-AUC, SHAP values"),
    ("04", "Live Risk Assessor", "Real-time risk scoring"),
]
for col, (num, title, desc) in zip([col1, col2, col3, col4], pages):
    with col:
        st.markdown(f"""
        <div class="stat-card" style="cursor:pointer;">
            <div style="font-size:1.6rem; margin-bottom:0.5rem; font-weight:800;
                        color:#C8102E; font-family:'JetBrains Mono',monospace;">{num}</div>
            <div style="font-weight:700; font-size:0.88rem; color:#E6EDF3;">{title}</div>
            <div style="font-size:0.75rem; color:#6E7681; margin-top:0.3rem;">{desc}</div>
        </div>
        """, unsafe_allow_html=True)
