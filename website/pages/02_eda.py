"""
02_eda.py — Exploratory Data Analysis Page
Canadian Rail Incident & Delay Prediction | Streamlit Dashboard
Author: Pranay Ratan | SFU Data Science
"""

import sys
from pathlib import Path
import streamlit as st
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

st.set_page_config(page_title="EDA | CPKC Rail Risk", page_icon="📊", layout="wide")

css_path = Path(__file__).parent.parent / "assets" / "cpkc_theme.css"
if css_path.exists():
    with open(css_path) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


@st.cache_data(show_spinner="Loading data...")
def load_data() -> pd.DataFrame:
    processed = Path(__file__).resolve().parent.parent.parent / "data" / "processed" / "incidents_featured.parquet"
    if processed.exists():
        return pd.read_parquet(processed)
    from src.data_loader import fetch_transport_canada_dataset, fetch_weather_by_province, get_province_route_density
    from src.preprocessing import run_preprocessing_pipeline
    raw = fetch_transport_canada_dataset(cache=True)
    weather = fetch_weather_by_province(cache=True)
    density = get_province_route_density()
    return run_preprocessing_pipeline(raw, weather_df=weather, density_df=density)


df_full = load_data()

# ---------------------------------------------------------------------------
# PAGE HEADER
# ---------------------------------------------------------------------------
st.markdown("""
<div class="hero-banner" style="padding: 2rem 2.5rem;">
    <h1 style="font-size:1.9rem !important;">📊 Exploratory Data Analysis</h1>
    <p>25 years of Canadian rail occurrence data — filtered, sliced, and visualized interactively.</p>
</div>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# FILTERS
# ---------------------------------------------------------------------------
st.markdown("### 🔧 Filter Data")
col1, col2, col3 = st.columns(3)

with col1:
    all_provinces = sorted(df_full["province"].unique().tolist())
    province_sel = st.multiselect(
        "Province", options=all_provinces,
        default=all_provinces[:6],
        help="Select one or more Canadian provinces"
    )

with col2:
    year_min = int(df_full["year"].min())
    year_max = int(df_full["year"].max())
    year_range = st.slider(
        "Year Range", min_value=year_min, max_value=year_max,
        value=(year_min, year_max)
    )

with col3:
    all_types = sorted(df_full["incident_type"].unique().tolist())
    type_sel = st.multiselect(
        "Incident Type", options=all_types, default=all_types,
    )

# Apply filters
mask = (
    df_full["province"].isin(province_sel if province_sel else all_provinces) &
    df_full["year"].between(*year_range) &
    df_full["incident_type"].isin(type_sel if type_sel else all_types)
)
df = df_full[mask].copy()

st.caption(f"Showing **{len(df):,}** incidents matching your filters.")

if len(df) < 5:
    st.warning("Not enough data for visualizations — please widen your filters.")
    st.stop()

st.markdown("---")

# ---------------------------------------------------------------------------
# CHART 1: Incidents per year
# ---------------------------------------------------------------------------
st.markdown("""<div class="section-header"><h2>📈 Annual Incident Trend</h2></div>""", unsafe_allow_html=True)

import src.visualizations as viz
fig1 = viz.plotly_incidents_per_year(df)
st.plotly_chart(fig1, use_container_width=True)

st.markdown("""
<div class="insight-card">
    <div class="insight-title">💡 Key Insight</div>
    <p>
        Rail incidents in Canada peaked in the mid-2000s before declining — driven by Transport Canada safety regulations, 
        improved track inspection technologies, and stronger dangerous goods protocols post-Lac-Mégantic (2013). 
        The downward trend validates that safety investment yields measurable outcomes on a 10+ year horizon.
    </p>
</div>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# CHART 2: Province × Type Heatmap
# ---------------------------------------------------------------------------
st.markdown("---")
st.markdown("""<div class="section-header"><h2>🗺️ Incident Frequency by Province &amp; Type</h2></div>""", unsafe_allow_html=True)

fig2 = viz.plotly_province_heatmap(df)
st.plotly_chart(fig2, use_container_width=True)

st.markdown("""
<div class="insight-card">
    <div class="insight-title">💡 Key Insight</div>
    <p>
        Ontario and Quebec dominate raw counts due to network density, but Saskatchewan and Alberta 
        show disproportionate <strong>derailment frequency</strong> relative to their population — 
        reflecting the high volume of bulk commodity unit trains (grain, potash, crude oil) operating 
        on mixed terrain. These are prime corridors for CPKC proactive inspection scheduling.
    </p>
</div>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# CHART 3: Top Provinces Bar
# ---------------------------------------------------------------------------
st.markdown("---")
st.markdown("""<div class="section-header"><h2>🏆 Top Provinces by Incident Count</h2></div>""", unsafe_allow_html=True)

import plotly.express as px
top_prov = df["province"].value_counts().head(10).reset_index()
top_prov.columns = ["province", "incidents"]
fig3 = px.bar(
    top_prov, x="incidents", y="province", orientation="h",
    color="incidents", color_continuous_scale="RdYlGn_r",
    text="incidents",
    labels={"incidents": "Total Incidents", "province": ""},
    title="Top 10 Provinces by Total Rail Incidents",
)
fig3.update_traces(texttemplate="%{text:,}", textposition="outside")
fig3.update_layout(template="plotly_white", font=dict(family="Arial", size=13),
                   coloraxis_showscale=False, yaxis=dict(autorange="reversed"))
st.plotly_chart(fig3, use_container_width=True)

# ---------------------------------------------------------------------------
# CHART 4: Season breakdown
# ---------------------------------------------------------------------------
st.markdown("---")
st.markdown("""<div class="section-header"><h2>🌨️ Incidents by Season</h2></div>""", unsafe_allow_html=True)

season_counts = df.groupby("season").size().reset_index(name="count")
order = ["Winter", "Spring", "Summer", "Fall"]
season_counts["season"] = pd.Categorical(season_counts["season"], categories=order, ordered=True)
season_counts = season_counts.sort_values("season")

colors_season = {"Winter": "#1A3A5C", "Spring": "#2E8B57", "Summer": "#D4A017", "Fall": "#C8102E"}
fig4 = px.bar(season_counts, x="season", y="count",
              color="season", color_discrete_map=colors_season,
              text="count",
              labels={"count": "Total Incidents", "season": "Season"},
              title="Rail Incidents by Season (All Years)")
fig4.update_traces(texttemplate="%{text:,}", textposition="outside")
fig4.update_layout(template="plotly_white", showlegend=False,
                   font=dict(family="Arial", size=13))
st.plotly_chart(fig4, use_container_width=True)

st.markdown("""
<div class="insight-card">
    <div class="insight-title">💡 Key Insight</div>
    <p>
        Winter sees the highest incident count and disproportionally more <em>high-severity</em> events. 
        Cold temperatures cause rail contraction and brittle joints, while reduced daylight hours impair 
        crossing visibility. CPKC's winter operations protocols and track warming programs directly address 
        this peak-risk window.
    </p>
</div>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# CHART 5: Incident Type Breakdown
# ---------------------------------------------------------------------------
st.markdown("---")
st.markdown("""<div class="section-header"><h2>🔍 Incident Type Distribution</h2></div>""", unsafe_allow_html=True)

type_counts = df["incident_type"].value_counts().reset_index()
type_counts.columns = ["incident_type", "count"]
fig5 = px.pie(type_counts, names="incident_type", values="count",
              color_discrete_sequence=["#C8102E","#1A3A5C","#2E8B57","#D4A017",
                                       "#6A3D9A","#FF6B35","#00B4D8","#8B0000","#888"],
              title="Rail Incident Type Breakdown",
              hole=0.4)
fig5.update_traces(textinfo="percent+label")
fig5.update_layout(template="plotly_white", font=dict(family="Arial", size=12),
                   legend=dict(orientation="v", x=1.05))
st.plotly_chart(fig5, use_container_width=True)

# ---------------------------------------------------------------------------
# CHART 7: Rolling 12-month by Province
# ---------------------------------------------------------------------------
st.markdown("---")
st.markdown("""<div class="section-header"><h2>📉 Rolling 12-Month Trend by Province</h2></div>""", unsafe_allow_html=True)

top_n_slider = st.slider("Number of provinces to show", min_value=2, max_value=8, value=5)
fig7 = viz.plotly_rolling_by_province(df, top_n=top_n_slider)
st.plotly_chart(fig7, use_container_width=True)

st.markdown("""
<div class="insight-card">
    <div class="insight-title">💡 Key Insight</div>
    <p>
        The rolling 12-month view smooths noise and reveals structural shifts: Ontario experienced 
        a notable surge mid-decade coinciding with increased intermodal traffic, while Saskatchewan 
        tracks closely with grain export cycles — peaking post-harvest in September through November. 
        This temporal pattern is captured by the <strong>rolling_12m_incidents</strong> feature, 
        the top SHAP predictor in our model.
    </p>
</div>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# SEVERITY BREAKDOWN
# ---------------------------------------------------------------------------
st.markdown("---")
st.markdown("""<div class="section-header"><h2>🎯 Severity Class Distribution</h2></div>""", unsafe_allow_html=True)

col_s1, col_s2 = st.columns([2, 1])
with col_s1:
    sev_df = df["incident_severity"].value_counts().reset_index()
    sev_df.columns = ["severity", "count"]
    sev_df["label"] = sev_df["severity"].map({0: "LOW Risk", 1: "HIGH Risk"})
    fig_sev = px.bar(sev_df, x="label", y="count",
                     color="label",
                     color_discrete_map={"LOW Risk": "#2E8B57", "HIGH Risk": "#C8102E"},
                     text="count",
                     labels={"count": "Incidents", "label": "Severity Class"},
                     title="Incident Severity Distribution")
    fig_sev.update_traces(texttemplate="%{text:,}", textposition="outside")
    fig_sev.update_layout(template="plotly_white", showlegend=False,
                          font=dict(family="Arial", size=13))
    st.plotly_chart(fig_sev, use_container_width=True)

with col_s2:
    n_high = df[df["incident_severity"] == 1].shape[0]
    n_low  = df[df["incident_severity"] == 0].shape[0]
    total  = len(df)
    st.metric("HIGH Risk Incidents", f"{n_high:,}", f"{n_high/total*100:.1f}% of total")
    st.metric("LOW Risk Incidents",  f"{n_low:,}",  f"{n_low/total*100:.1f}% of total")
    st.metric("Total Incidents",     f"{total:,}")

st.markdown("""
<div class="disclaimer">
    ⚠️ For demonstration purposes — built using Transport Canada public data.
</div>
""", unsafe_allow_html=True)
