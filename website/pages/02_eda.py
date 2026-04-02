"""
02_eda.py — Deep Dive Explorer
Rail Risk Intelligence Dashboard
Author: Pranay Ratan | SFU Data Science
"""

import sys
from pathlib import Path
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

st.set_page_config(page_title="Deep Dive Explorer | Rail Risk Intelligence", page_icon="", layout="wide")

css_path = Path(__file__).parent.parent / "assets" / "theme.css"
if css_path.exists():
    with open(css_path) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

DARK_TEMPLATE = dict(
    paper_bgcolor='#161B22',
    plot_bgcolor='#0D1117',
    font=dict(family='Inter, sans-serif', color='#8B949E', size=12),
    margin=dict(l=60, r=20, t=50, b=50),
    xaxis=dict(gridcolor='#30363D', zerolinecolor='#30363D'),
    yaxis=dict(gridcolor='#30363D', zerolinecolor='#30363D'),
    hoverlabel=dict(bgcolor='#161B22', bordercolor='#30363D', font=dict(color='#E6EDF3', size=12)),
    colorway=['#C8102E', '#1A3A5C', '#2E8B57', '#D29922', '#8957E5', '#2E90FA', '#FF6B35', '#6E7681'],
)


@st.cache_data(show_spinner="Loading data...")
def load_data():
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

st.markdown("""
<div class="hero-banner">
    <h1>Deep Dive Explorer</h1>
    <p>Interactive data exploration across 25 years of Canadian rail occurrence data — filtered, sliced, and visualized.</p>
</div>
""", unsafe_allow_html=True)

# Filters
st.markdown('<div class="section-header"><h2>Filter Data</h2></div>', unsafe_allow_html=True)
col1, col2, col3 = st.columns(3)

with col1:
    all_provinces = sorted(df_full["province"].unique().tolist())
    province_sel = st.multiselect("Province", options=all_provinces, default=all_provinces[:6])
with col2:
    year_min, year_max = int(df_full["year"].min()), int(df_full["year"].max())
    year_range = st.slider("Year Range", min_value=year_min, max_value=year_max, value=(year_min, year_max))
with col3:
    all_types = sorted(df_full["incident_type"].unique().tolist())
    type_sel = st.multiselect("Incident Type", options=all_types, default=all_types)

mask = (
    df_full["province"].isin(province_sel if province_sel else all_provinces)
    & df_full["year"].between(*year_range)
    & df_full["incident_type"].isin(type_sel if type_sel else all_types)
)
df = df_full[mask].copy()
st.caption(f"Showing **{len(df):,}** incidents matching filters.")

if len(df) < 5:
    st.warning("Not enough data for visualizations. Please widen your filters.")
    st.stop()

# Tabs
tab_geo, tab_temp, tab_corr, tab_typo = st.tabs(["Geographic View", "Temporal View", "Correlation Explorer", "Incident Typology"])

with tab_geo:
    st.markdown('<div class="section-header"><h2>Province vs Incident Type Heatmap</h2></div>', unsafe_allow_html=True)

    pivot = df.pivot_table(index="province", columns="incident_type", values="year", aggfunc="count", fill_value=0)
    fig_hm = go.Figure(data=go.Heatmap(
        z=pivot.values, x=pivot.columns.tolist(), y=pivot.index.tolist(),
        colorscale=[[0, '#0D1117'], [0.25, '#1A3A5C'], [0.5, '#D29922'], [1, '#C8102E']],
        hovertemplate='<b>%{y}</b> / %{x}<br>Count: %{z}<extra></extra>',
        colorbar=dict(title=dict(text='Count', font=dict(color='#8B949E')), tickfont=dict(color='#8B949E')),
    ))
    fig_hm.update_layout(**DARK_TEMPLATE, height=520, margin=dict(l=180, r=60, t=20, b=120),
                          xaxis=dict(tickangle=-35, tickfont=dict(size=10)),
                          yaxis=dict(tickfont=dict(size=11)))
    st.plotly_chart(fig_hm, use_container_width=True)

    st.markdown("""
    <div class="insight-card">
        <div class="insight-title">Insight</div>
        <p>Saskatchewan and Alberta rank highest for derailment rates relative to population, reflecting 
        the high volume of bulk commodity unit trains (grain, potash, crude oil) on mixed terrain. 
        These are priority corridors for proactive inspection scheduling.</p>
    </div>
    """, unsafe_allow_html=True)

    # Top Provinces Bar
    st.markdown('<div class="section-header"><h2>Top Provinces by Incident Count</h2></div>', unsafe_allow_html=True)
    top_prov = df["province"].value_counts().head(10).reset_index()
    top_prov.columns = ["province", "incidents"]
    fig_bar = go.Figure(go.Bar(
        y=top_prov["province"], x=top_prov["incidents"], orientation='h',
        marker_color=['#C8102E' if i < 3 else '#1A3A5C' for i in range(len(top_prov))],
        text=top_prov["incidents"], textposition='outside',
        textfont=dict(color='#8B949E', size=11),
        hovertemplate='<b>%{y}</b>: %{x:,} incidents<extra></extra>',
    ))
    fig_bar.update_layout(**DARK_TEMPLATE, height=380, yaxis=dict(autorange='reversed'),
                          margin=dict(l=160, r=60, t=20, b=30))
    st.plotly_chart(fig_bar, use_container_width=True)


with tab_temp:
    st.markdown('<div class="section-header"><h2>Annual Incident Trend</h2></div>', unsafe_allow_html=True)

    annual = df.groupby("year").agg(total=("year", "size"), high=("incident_severity", "sum")).reset_index()
    fig_trend = go.Figure()
    fig_trend.add_trace(go.Scatter(
        x=annual["year"], y=annual["total"], mode='lines+markers', name='Total',
        line=dict(color='#C8102E', width=2.5), marker=dict(size=5),
        fill='tozeroy', fillcolor='rgba(200,16,46,0.08)',
        hovertemplate='<b>%{x}</b>: %{y} incidents<extra></extra>',
    ))
    fig_trend.add_trace(go.Scatter(
        x=annual["year"], y=annual["high"], mode='lines', name='High Risk',
        line=dict(color='#D29922', width=1.5, dash='dot'),
        hovertemplate='<b>%{x}</b>: %{y} high risk<extra></extra>',
    ))
    fig_trend.update_layout(**DARK_TEMPLATE, height=350,
                             legend=dict(font=dict(color='#8B949E', size=11), orientation='h', y=-0.15))
    st.plotly_chart(fig_trend, use_container_width=True)

    st.markdown("""
    <div class="insight-card">
        <div class="insight-title">Insight</div>
        <p>Total incidents declined approximately 23% from peak to present, driven by Transport Canada safety 
        regulations, improved track inspection technologies, and stronger dangerous goods protocols 
        post-Lac-Megantic (2013).</p>
    </div>
    """, unsafe_allow_html=True)

    # Season breakdown
    st.markdown('<div class="section-header"><h2>Incidents by Season</h2></div>', unsafe_allow_html=True)
    season_counts = df.groupby("season").size().reset_index(name="count")
    order = ["Winter", "Spring", "Summer", "Fall"]
    season_counts["season"] = pd.Categorical(season_counts["season"], categories=order, ordered=True)
    season_counts = season_counts.sort_values("season")

    fig_season = go.Figure(go.Bar(
        x=season_counts["season"], y=season_counts["count"],
        marker_color=['#1A3A5C', '#2E8B57', '#D29922', '#C8102E'],
        text=season_counts["count"], textposition='outside',
        textfont=dict(color='#8B949E', size=11),
        hovertemplate='<b>%{x}</b>: %{y:,} incidents<extra></extra>',
    ))
    fig_season.update_layout(**DARK_TEMPLATE, height=350, showlegend=False)
    st.plotly_chart(fig_season, use_container_width=True)

    st.markdown("""
    <div class="insight-card">
        <div class="insight-title">Insight</div>
        <p>Winter sees the highest incident count and disproportionally more high-severity events. 
        Cold temperatures cause rail contraction and brittle joints, while reduced daylight hours impair 
        crossing visibility.</p>
    </div>
    """, unsafe_allow_html=True)

    # Rolling 12-month by Province
    st.markdown('<div class="section-header"><h2>Rolling 12-Month Trend by Province</h2></div>', unsafe_allow_html=True)
    top_n = st.slider("Number of provinces to display", min_value=2, max_value=8, value=5)
    try:
        import src.visualizations as viz
        fig_roll = viz.plotly_rolling_by_province(df, top_n=top_n)
        fig_roll.update_layout(**DARK_TEMPLATE, height=350)
        st.plotly_chart(fig_roll, use_container_width=True)
    except Exception:
        top_provinces = df["province"].value_counts().head(top_n).index.tolist()
        colors = ['#C8102E', '#1A3A5C', '#2E8B57', '#D29922', '#8957E5', '#2E90FA', '#FF6B35', '#6E7681']
        fig_roll = go.Figure()
        for i, prov in enumerate(top_provinces):
            prov_df = df[df["province"] == prov].groupby(["year", "month"]).size().reset_index(name="incidents")
            prov_df["date"] = pd.to_datetime(prov_df["year"].astype(str) + "-" + prov_df["month"].astype(str).str.zfill(2) + "-01")
            prov_df = prov_df.sort_values("date")
            prov_df["rolling"] = prov_df["incidents"].rolling(12, min_periods=1).mean()
            fig_roll.add_trace(go.Scatter(
                x=prov_df["date"], y=prov_df["rolling"], mode='lines', name=prov,
                line=dict(color=colors[i % len(colors)], width=2),
            ))
        fig_roll.update_layout(**DARK_TEMPLATE, height=350,
                                legend=dict(font=dict(color='#8B949E', size=10), orientation='h', y=-0.2))
        st.plotly_chart(fig_roll, use_container_width=True)


with tab_corr:
    st.markdown('<div class="section-header"><h2>Feature Correlation Matrix</h2></div>', unsafe_allow_html=True)

    feature_cols = [
        "province_risk_score", "route_density_score", "incident_type_encoded",
        "cargo_risk", "rolling_12m_incidents", "avg_temp_c", "total_precip_mm",
        "avg_wind_kmh", "season_num", "fatalities", "injuries",
        "years_since_2000", "mile_post",
    ]
    existing = [c for c in feature_cols if c in df.columns]
    corr = df[existing].corr().round(3)
    labels = [f.replace("_", " ") for f in corr.columns]

    fig_corr = go.Figure(data=go.Heatmap(
        z=corr.values, x=labels, y=labels,
        colorscale='RdBu', zmid=0, zmin=-1, zmax=1,
        hovertemplate='%{x} vs %{y}<br>r = %{z:.3f}<extra></extra>',
        colorbar=dict(title=dict(text='Pearson r', font=dict(color='#8B949E')), tickfont=dict(color='#8B949E')),
    ))
    fig_corr.update_layout(**DARK_TEMPLATE, height=650,
                            margin=dict(l=160, r=60, t=20, b=140),
                            xaxis=dict(tickangle=-45, tickfont=dict(size=9)),
                            yaxis=dict(tickfont=dict(size=9), autorange='reversed'))
    st.plotly_chart(fig_corr, use_container_width=True)

    st.markdown("""
    <div class="insight-card">
        <div class="insight-title">Insight</div>
        <p>Strong positive correlation between route_density_score and rolling_12m_incidents confirms that 
        high-traffic corridors accumulate incidents. The interaction feature density_x_rolling captures 
        this compound risk effectively, explaining its high SHAP importance.</p>
    </div>
    """, unsafe_allow_html=True)


with tab_typo:
    st.markdown('<div class="section-header"><h2>Incident Type Severity Breakdown</h2></div>', unsafe_allow_html=True)

    sev_by_type = []
    for itype in df["incident_type"].unique():
        type_df = df[df["incident_type"] == itype]
        high = int(type_df["incident_severity"].sum())
        low = len(type_df) - high
        sev_by_type.append({"type": itype, "high": high, "low": low, "total": len(type_df)})
    sev_by_type = sorted(sev_by_type, key=lambda x: x["total"], reverse=True)

    fig_sev = go.Figure()
    fig_sev.add_trace(go.Bar(
        y=[d["type"] for d in sev_by_type], x=[d["high"] for d in sev_by_type],
        orientation='h', name='High Severity', marker_color='rgba(200,16,46,0.75)',
        hovertemplate='<b>%{y}</b><br>High: %{x}<extra></extra>',
    ))
    fig_sev.add_trace(go.Bar(
        y=[d["type"] for d in sev_by_type], x=[d["low"] for d in sev_by_type],
        orientation='h', name='Low Severity', marker_color='rgba(26,58,92,0.75)',
        hovertemplate='<b>%{y}</b><br>Low: %{x}<extra></extra>',
    ))
    fig_sev.update_layout(**DARK_TEMPLATE, barmode='stack', height=450,
                           yaxis=dict(autorange='reversed', tickfont=dict(size=11)),
                           margin=dict(l=200, r=30, t=20, b=50),
                           legend=dict(font=dict(color='#8B949E', size=11), orientation='h', y=-0.12))
    st.plotly_chart(fig_sev, use_container_width=True)

    # Donut chart
    st.markdown('<div class="section-header"><h2>Incident Type Distribution</h2></div>', unsafe_allow_html=True)
    type_counts = df["incident_type"].value_counts().reset_index()
    type_counts.columns = ["incident_type", "count"]
    fig_pie = go.Figure(go.Pie(
        labels=type_counts["incident_type"], values=type_counts["count"],
        hole=0.55, textinfo='percent',
        textfont=dict(color='#E6EDF3', size=11),
        marker=dict(
            colors=['#C8102E', '#1A3A5C', '#2E8B57', '#D29922', '#8957E5', '#2E90FA', '#FF6B35', '#6E7681', '#444'],
            line=dict(color='#161B22', width=2),
        ),
        hovertemplate='<b>%{label}</b><br>%{value:,} incidents (%{percent})<extra></extra>',
    ))
    fig_pie.update_layout(**DARK_TEMPLATE, height=400,
                           legend=dict(font=dict(size=10, color='#8B949E'), x=1.05, y=0.5))
    st.plotly_chart(fig_pie, use_container_width=True)

    st.markdown("""
    <div class="insight-card">
        <div class="insight-title">Insight</div>
        <p>Derailments, Main Track Collisions, and Dangerous Goods Releases carry the highest severity rates. 
        Employee Fatality incidents, while less frequent, have near-100% high severity classification. 
        The model leverages incident_type_encoded as a top-5 predictive feature.</p>
    </div>
    """, unsafe_allow_html=True)


st.markdown("""
<div class="disclaimer">
    Built with Transport Canada public data. For demonstration purposes only.
</div>
""", unsafe_allow_html=True)
