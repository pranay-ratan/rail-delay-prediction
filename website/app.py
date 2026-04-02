"""
app.py — Streamlit Multi-Page App Entry Point
Canadian Rail Incident & Delay Prediction
Author: Pranay Ratan | SFU Data Science
"""

import streamlit as st
from pathlib import Path

# ---------------------------------------------------------------------------
# Page Config (must be first Streamlit call)
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="CPKC Rail Risk Intelligence",
    page_icon="🚂",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "Get Help": "https://github.com/pranayratan",
        "About": "Canadian Rail Incident & Delay Prediction | Pranay Ratan",
    },
)

# ---------------------------------------------------------------------------
# Load Custom CSS
# ---------------------------------------------------------------------------
css_path = Path(__file__).parent / "assets" / "cpkc_theme.css"
if css_path.exists():
    with open(css_path) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("""
    <div style="text-align:center; padding: 1rem 0 0.5rem 0;">
        <div style="font-size:2.5rem;">🚂</div>
        <div style="font-size:1.1rem; font-weight:800; letter-spacing:-0.3px; margin-top:0.3rem;">
            Rail Risk Intelligence
        </div>
        <div style="font-size:0.78rem; opacity:0.7; margin-top:0.2rem;">
            CPKC · Canadian Pacific Kansas City
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    st.markdown("### 📋 Navigation")
    st.markdown("""
    - 🏠 **Overview** — Project summary & key stats
    - 📊 **EDA** — Interactive data exploration
    - 🤖 **Model Results** — Performance & interpretation
    - 🎯 **Live Predictor** — Real-time risk scoring
    """)

    st.markdown("---")

    st.markdown("### 👤 Author")
    st.markdown("""
    **Pranay Ratan**  
    BSc Data Science, SFU  
    
    [![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-0077B5?logo=linkedin)](https://linkedin.com/in/pranayratan)
    [![GitHub](https://img.shields.io/badge/GitHub-Repo-181717?logo=github)](https://github.com/pranayratan)
    """)

    st.markdown("---")

    st.markdown("### 📦 Tech Stack")
    st.markdown("""
    `Python 3.11` · `scikit-learn` · `XGBoost`  
    `LightGBM` · `SHAP` · `Streamlit`  
    `Plotly` · `pandas` · `Open-Meteo API`
    """)

    st.markdown("---")

    st.markdown(
        "<div style='font-size:0.72rem; opacity:0.6; text-align:center'>"
        "Built with Transport Canada public data.<br>"
        "For demonstration purposes only."
        "</div>",
        unsafe_allow_html=True,
    )

# ---------------------------------------------------------------------------
# Landing redirect notice (default page)
# ---------------------------------------------------------------------------
st.markdown("""
<div class="hero-banner">
    <h1>🚂 Canadian Rail Incident & Delay Prediction</h1>
    <p>
        An end-to-end machine learning system for predicting incident severity across 
        the Canadian rail network — built for <strong>CPKC (Canadian Pacific Kansas City)</strong>, 
        North America's only transnational railroad.
    </p>
</div>
""", unsafe_allow_html=True)

st.info("👈 **Select a page from the sidebar** to explore the project.")

col1, col2, col3, col4 = st.columns(4)
pages = [
    ("🏠", "Project Overview", "Key stats, motivation & dataset"),
    ("📊", "Exploratory Data Analysis", "Interactive charts & insights"),
    ("🤖", "Model Results", "Accuracy, ROC-AUC, SHAP values"),
    ("🎯", "Live Predictor", "Real-time risk scoring"),
]
for col, (icon, title, desc) in zip([col1, col2, col3, col4], pages):
    with col:
        st.markdown(f"""
        <div class="stat-card" style="cursor:pointer;">
            <div style="font-size:2rem; margin-bottom:0.5rem;">{icon}</div>
            <div style="font-weight:700; font-size:0.95rem; color:#1A2B4A;">{title}</div>
            <div style="font-size:0.8rem; color:#6B7280; margin-top:0.3rem;">{desc}</div>
        </div>
        """, unsafe_allow_html=True)
