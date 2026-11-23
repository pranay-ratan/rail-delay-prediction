"""Data explorer: where, when and what kind of accidents, and how severe."""

from __future__ import annotations

import plotly.graph_objects as go
import streamlit as st

from common import COLORS, PLOT_LAYOUT, setup_page
from insights import SEASON_ORDER, by_type

res = setup_page("Data Explorer")
df = res.incidents
st.title("Data Explorer")
st.caption("TSB rail accidents 2001-2025. Severe = at least one fatality or serious injury.")

provinces = sorted(df["province"].unique())
chosen = st.multiselect("Provinces", provinces, default=provinces)
years = st.slider("Years", int(df["year"].min()), int(df["year"].max()), (int(df["year"].min()), int(df["year"].max())))
view = df[df["province"].isin(chosen) & df["year"].between(*years)]
if view.empty:
    st.warning("No accidents match these filters.")
    st.stop()
st.markdown(f"**{len(view):,} accidents**, {view['incident_severity'].mean():.1%} severe")

left, right = st.columns(2)
with left:
    prov = view.groupby("province")["incident_severity"].agg(total="size", rate="mean").sort_values("total")
    fig = go.Figure(go.Bar(
        x=prov["total"], y=prov.index, orientation="h",
        marker=dict(color=prov["rate"] * 100, colorscale=[[0, "#1A3A5C"], [0.5, "#D29922"], [1, "#C8102E"]],
                    showscale=True, colorbar=dict(title="Severe %")),
        hovertemplate="%{y}: %{x} accidents<extra></extra>"))
    fig.update_layout(**PLOT_LAYOUT, height=420, title="Accidents by province (colour = severe %)")
    st.plotly_chart(fig, use_container_width=True)
with right:
    season = view.groupby("season")["incident_severity"].mean().reindex(SEASON_ORDER) * 100
    fig = go.Figure(go.Bar(x=season.index, y=season.values, marker_color=COLORS[:4],
                           text=[f"{v:.1f}%" for v in season.values], textposition="outside"))
    fig.update_layout(**PLOT_LAYOUT, height=420, title="Severe rate by season", yaxis_ticksuffix="%")
    st.plotly_chart(fig, use_container_width=True)

types = by_type(view).assign(severe_rate=lambda g: (g["high"] / g["total"] * 100).round(1))
fig = go.Figure([
    go.Bar(y=types["type"], x=types["high"], name="Severe", orientation="h", marker_color="rgba(200,16,46,0.75)"),
    go.Bar(y=types["type"], x=types["low"], name="Not severe", orientation="h", marker_color="rgba(26,58,92,0.75)"),
])
fig.update_layout(**PLOT_LAYOUT, barmode="stack", height=480, title="Severity by accident type")
fig.update_yaxes(autorange="reversed")
st.plotly_chart(fig, use_container_width=True)
st.dataframe(
    types.rename(columns={"type": "Accident type", "total": "Accidents", "high": "Severe",
                          "low": "Not severe", "severe_rate": "Severe %"}),
    hide_index=True, use_container_width=True,
)
st.info("Accident type is only known after the fact, so it is not a model input. "
        "The model must infer risk from conditions known beforehand.")

hourly = view.dropna(subset=["hour"]).groupby("hour")["incident_severity"].agg(total="size", rate="mean")
fig = go.Figure([
    go.Bar(x=hourly.index, y=hourly["total"], name="Accidents", marker_color="#1A3A5C"),
    go.Scatter(x=hourly.index, y=hourly["rate"] * 100, name="Severe %", yaxis="y2", line=dict(color="#C8102E", width=3)),
])
fig.update_layout(**PLOT_LAYOUT, height=380, title="By hour of day",
                  yaxis2=dict(overlaying="y", side="right", ticksuffix="%", showgrid=False),
                  legend=dict(orientation="h", y=-0.2))
st.plotly_chart(fig, use_container_width=True)
