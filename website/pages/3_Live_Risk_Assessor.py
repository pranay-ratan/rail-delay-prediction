"""Live risk assessor: the same model and feature code as /api/predict."""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from pydantic import ValidationError

from common import layout, setup_page
from predictor import (
    PredictionRequest, build_features, calibrated_probability, explain, risk_level, similar_incidents,
)

TRACK = {"Main line": "main", "Yard": "yard", "Other / siding": "other"}
ACTIVITY = {"Proceeding": "proceeding", "Switching": "switching", "Standing": "standing", "Other": "other"}
TRAIN = {"Freight": "freight", "Passenger": "passenger", "Cut of cars": "cars_only", "Track unit": "track_unit", "Other": "other"}

res = setup_page("Live Risk Assessor")
df = res.incidents
base_rate = res.summary["positive_rate"]

st.title("Live Risk Assessor")
st.caption("If an accident happens under these conditions, how likely is it to kill or seriously injure someone? "
           "Leave optional fields at 0 / blank for 'unknown'.")


def optional(value: float):
    return None if value is None or value == 0 else value


with st.form("scenario"):
    c1, c2, c3 = st.columns(3)
    province = c1.selectbox("Province", sorted(df["province"].unique()), index=sorted(df["province"].unique()).index("Ontario"))
    track = c2.radio("Track", list(TRACK), horizontal=True)
    activity = c3.radio("Train activity", list(ACTIVITY), horizontal=True)
    train = c1.radio("Train type", list(TRAIN), horizontal=True)
    year = c2.slider("Year", 2001, 2030, 2025)
    month = c3.slider("Month", 1, 12, 1)
    hour = c1.slider("Hour", 0, 23, 12)
    weekend = c2.checkbox("Weekend")
    speed = c3.number_input("Train speed (mph, 0 = unknown)", 0.0, 120.0, 40.0)
    cars = c1.number_input("Cars (0 = unknown)", 0, 400, 0)
    tonnage = c2.number_input("Tonnage (0 = unknown)", 0.0, 40000.0, 0.0)
    temp = c3.text_input("Temperature °C (blank = climatology)", "")
    submitted = st.form_submit_button("Assess severity risk", type="primary")

if submitted:
    try:
        req = PredictionRequest(
            province=province, year=year, month=month, hour=hour, is_weekend=weekend,
            track_type=TRACK[track], activity=ACTIVITY[activity], train_type=TRAIN[train],
            approx_speed_mph=optional(speed), num_cars=optional(cars), train_tonnage=optional(tonnage),
            wx_temp_c=float(temp) if temp.strip() else None,
        )
    except (ValidationError, ValueError) as exc:
        st.error(f"Invalid input: {exc}")
        st.stop()

    X, notes = build_features(req, df, res.feature_names)
    score = float(res.model.predict_proba(X)[0, 1])
    prob = calibrated_probability(score, res.calibration)
    level = risk_level(prob, base_rate)
    factors, _ = explain(res.model, X)

    m1, m2, m3 = st.columns(3)
    m1.metric("Chance the accident is severe", f"{prob:.1%}")
    m2.metric("vs overall rate", f"{prob / base_rate:.2f}×", f"base rate {base_rate:.1%}", delta_color="off")
    m3.metric("Risk level", level, "HIGH ≥ 2×, MEDIUM ≥ 1× base rate", delta_color="off")
    st.caption(f"Calibrated: share of past accidents with a similar model score that were severe. "
               f"Province history as of {notes['history_as_of']}; weather from climatology: "
               f"{', '.join(notes['weather_filled_from_climatology']) or 'none'}; unknown → training median: "
               f"{', '.join(notes['imputed_by_model']) or 'none'}.")

    f = pd.DataFrame(factors).iloc[::-1]
    fig = go.Figure(go.Bar(x=f["value"], y=f["feature"], orientation="h",
                           marker_color=["#C8102E" if v > 0 else "#2E8B57" for v in f["value"]]))
    fig.update_layout(**layout(height=380, title="Why: this prediction's SHAP contributions (log-odds)",
                               margin=dict(l=220, r=20, t=40, b=40)))
    st.plotly_chart(fig, use_container_width=True)

    similar = similar_incidents(req, df)
    if similar:
        st.subheader("Most recent similar real accidents")
        st.dataframe(pd.DataFrame(similar), hide_index=True, use_container_width=True)
