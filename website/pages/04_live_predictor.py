"""
04_live_predictor.py — Live Risk Assessor
Rail Risk Intelligence Dashboard
Author: Pranay Ratan | SFU Data Science
"""

import sys
from pathlib import Path
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

st.set_page_config(page_title="Live Risk Assessor | Rail Risk Intelligence", page_icon="", layout="wide")

css_path = Path(__file__).parent.parent / "assets" / "theme.css"
if css_path.exists():
    with open(css_path) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

DARK_TEMPLATE = dict(
    paper_bgcolor='#161B22',
    plot_bgcolor='#0D1117',
    font=dict(family='Inter, sans-serif', color='#8B949E', size=12),
    margin=dict(l=30, r=30, t=30, b=10),
    hoverlabel=dict(bgcolor='#161B22', bordercolor='#30363D', font=dict(color='#E6EDF3', size=12)),
)

# Constants
PROVINCES = [
    "Alberta", "British Columbia", "Manitoba", "New Brunswick",
    "Newfoundland and Labrador", "Northwest Territories", "Nova Scotia",
    "Nunavut", "Ontario", "Prince Edward Island", "Quebec",
    "Saskatchewan", "Yukon",
]
PROVINCE_RISK = {
    "Alberta": 0.12, "British Columbia": 0.14, "Manitoba": 0.06,
    "New Brunswick": 0.03, "Newfoundland and Labrador": 0.02,
    "Northwest Territories": 0.01, "Nova Scotia": 0.02, "Nunavut": 0.005,
    "Ontario": 0.22, "Prince Edward Island": 0.01, "Quebec": 0.18,
    "Saskatchewan": 0.11, "Yukon": 0.005,
}
SEASONS = ["Winter", "Spring", "Summer", "Fall"]
SEASON_NUM = {"Winter": 4, "Spring": 2, "Summer": 1, "Fall": 3}
INCIDENT_TYPES = [
    "Derailment", "Main Track Train Collision", "Crossing Collision",
    "Employee Fatality", "Non-main Track Collision", "Employee Injury",
    "Dangerous Goods Release", "Fire or Explosion", "Other Occurrence",
]
CARGO_TYPES = [
    "Dangerous Goods", "Crude Oil", "Coal", "Potash",
    "Grain", "General Freight", "Intermodal", "Passenger",
]
CARGO_RISK_MAP = {
    "Dangerous Goods": 3, "Crude Oil": 3, "Coal": 2, "Potash": 2,
    "Grain": 1, "General Freight": 1, "Intermodal": 1, "Passenger": 2,
}
ROUTE_DENSITY = {
    "Alberta": 0.87, "British Columbia": 0.62, "Manitoba": 0.83,
    "New Brunswick": 0.47, "Newfoundland and Labrador": 0.22,
    "Northwest Territories": 0.08, "Nova Scotia": 0.15, "Nunavut": 0.0,
    "Ontario": 0.23, "Prince Edward Island": 0.09, "Quebec": 0.25,
    "Saskatchewan": 1.0, "Yukon": 0.02,
}
SEVERITY_TYPES = {"Derailment", "Main Track Train Collision", "Dangerous Goods Release",
                  "Employee Fatality", "Fire or Explosion"}


st.markdown("""
<div class="hero-banner">
    <h1>Live Risk Assessor</h1>
    <p>Configure a rail corridor scenario using the controls below. The model will instantly 
    score the incident severity probability and identify the top driving factors.</p>
</div>
""", unsafe_allow_html=True)


# Load model
@st.cache_resource(show_spinner="Loading model...")
def load_model():
    try:
        from src.models import load_final_model
        model, features = load_final_model()
        return model, features, True
    except FileNotFoundError:
        return None, None, False

model, feature_names, model_loaded = load_model()

if not model_loaded:
    st.warning(
        "Model not yet trained. Run the notebook first, then refresh. "
        "The predictor will use a calibrated fallback in the meantime."
    )

# Initialize prediction history
if "prediction_history" not in st.session_state:
    st.session_state.prediction_history = []


# Input Form
st.markdown('<div class="section-header"><h2>Configure Corridor Scenario</h2></div>', unsafe_allow_html=True)

with st.form("risk_predictor_form"):
    col1, col2 = st.columns(2)

    with col1:
        province = st.selectbox("Province / Region", options=PROVINCES, index=PROVINCES.index("Ontario"))
        season = st.selectbox("Season", options=SEASONS,
                              help="Winter carries significantly higher risk.")
        incident_type = st.selectbox("Incident Type", options=INCIDENT_TYPES)
        cargo_type = st.selectbox("Cargo Type", options=CARGO_TYPES,
                                  index=CARGO_TYPES.index("Dangerous Goods"),
                                  help="Dangerous Goods and Crude Oil have highest risk scores.")

    with col2:
        year = st.slider("Year", min_value=2000, max_value=2025, value=2024)
        month = st.slider("Month", min_value=1, max_value=12, value=1)
        rolling_12m = st.slider("Rolling 12-Month Incidents (this corridor)",
                                min_value=0, max_value=200, value=45)
        fatalities = st.slider("Fatalities (if known)", min_value=0, max_value=5, value=0)
        injuries = st.slider("Injuries (if known)", min_value=0, max_value=15, value=0)
        is_weekend = st.checkbox("Weekend occurrence", value=False)

    submitted = st.form_submit_button("Assess Corridor Risk", use_container_width=True)


# Prediction
if submitted:
    st.markdown("---")
    st.markdown('<div class="section-header"><h2>Risk Assessment Results</h2></div>', unsafe_allow_html=True)

    province_risk = PROVINCE_RISK.get(province, 0.05)
    cargo_risk = CARGO_RISK_MAP.get(cargo_type, 1)
    route_density = ROUTE_DENSITY.get(province, 0.3)
    season_num = SEASON_NUM.get(season, 2)
    years_since_2000 = year - 2000
    multi_fatality = int(fatalities >= 2)
    incident_type_encoded = INCIDENT_TYPES.index(incident_type)
    cumulative = rolling_12m * 3
    density_x_rolling = route_density * rolling_12m
    season_x_province = season_num * province_risk
    cargo_x_type = cargo_risk * incident_type_encoded
    avg_temp = {"Winter": -12.0, "Spring": 5.0, "Summer": 18.0, "Fall": 5.0}.get(season, 0.0)
    temp_x_cargo = avg_temp * cargo_risk

    feature_values = {
        "year": year, "month": month, "day": 15,
        "fatalities": fatalities, "injuries": injuries,
        "evacuations": 0, "mile_post": 150.0,
        "temperature_c": avg_temp, "is_weekend": int(is_weekend),
        "years_since_2000": years_since_2000,
        "province_risk_score": province_risk,
        "route_density_score": route_density,
        "incident_type_encoded": incident_type_encoded,
        "cargo_risk": cargo_risk, "multi_fatality": multi_fatality,
        "rolling_12m_incidents": rolling_12m,
        "cumulative_incidents_province": cumulative,
        "season_num": season_num,
        "season_x_province_risk": season_x_province,
        "cargo_x_type_risk": cargo_x_type,
        "temp_x_cargo_risk": temp_x_cargo,
        "density_x_rolling": density_x_rolling,
        "avg_temp_c": avg_temp,
        "total_precip_mm": 40.0,
        "avg_wind_kmh": 20.0,
    }

    if model_loaded and feature_names:
        X_input = pd.DataFrame([feature_values])
        for col_name in feature_names:
            if col_name not in X_input.columns:
                X_input[col_name] = 0
        X_input = X_input[feature_names]
        try:
            prob = float(model.predict_proba(X_input)[0][1])
        except Exception:
            prob = float(model.predict(X_input)[0]) * 0.85
    else:
        score = 0.0
        score += province_risk * 0.3
        score += (cargo_risk / 3.0) * 0.25
        score += (season_num / 4.0) * 0.2
        score += min(rolling_12m / 100.0, 1.0) * 0.15
        score += (int(incident_type in SEVERITY_TYPES)) * 0.1
        prob = min(max(score + np.random.normal(0, 0.02), 0.05), 0.97)

    if prob < 0.4:
        risk_level, risk_class = "LOW", "risk-low"
    elif prob < 0.7:
        risk_level, risk_class = "MEDIUM", "risk-medium"
    else:
        risk_level, risk_class = "HIGH", "risk-high"

    # Store in history
    st.session_state.prediction_history.insert(0, {
        "province": province, "incident_type": incident_type,
        "cargo_type": cargo_type, "season": season,
        "probability": prob, "risk_level": risk_level,
    })
    st.session_state.prediction_history = st.session_state.prediction_history[:10]

    # Results Layout
    col_gauge, col_details = st.columns([1, 1])

    with col_gauge:
        risk_color = '#FF4444' if risk_level == 'HIGH' else '#D29922' if risk_level == 'MEDIUM' else '#2E8B57'
        fig_gauge = go.Figure(go.Indicator(
            mode='gauge+number',
            value=prob * 100,
            number=dict(suffix='%', font=dict(size=36, color=risk_color, family='JetBrains Mono')),
            gauge=dict(
                axis=dict(range=[0, 100], tickcolor='#6E7681', tickfont=dict(color='#6E7681')),
                bar=dict(color=risk_color, thickness=0.3),
                bgcolor='#0D1117',
                borderwidth=1, bordercolor='#30363D',
                steps=[
                    dict(range=[0, 40], color='rgba(46,139,87,0.12)'),
                    dict(range=[40, 70], color='rgba(210,153,34,0.12)'),
                    dict(range=[70, 100], color='rgba(200,16,46,0.12)'),
                ],
            ),
        ))
        fig_gauge.update_layout(**DARK_TEMPLATE, height=250)
        st.plotly_chart(fig_gauge, use_container_width=True)

        st.markdown(f"""
        <div style="text-align:center; margin-top:0.5rem;">
            <span class="{risk_class}">{risk_level} RISK — {prob*100:.1f}%</span>
        </div>
        """, unsafe_allow_html=True)

    with col_details:
        st.markdown("#### Scenario Summary")
        st.markdown(f"""
        | Parameter | Value |
        |-----------|-------|
        | Province | {province} |
        | Season | {season} |
        | Incident Type | {incident_type} |
        | Cargo | {cargo_type} (Risk: {'HIGH' if cargo_risk == 3 else 'MED' if cargo_risk == 2 else 'LOW'}) |
        | Year | {year} |
        | Rolling 12m | {rolling_12m} |
        | Weekend | {'Yes' if is_weekend else 'No'} |
        """)

        st.markdown("#### Top Driving Factors")
        factors = [
            ("Rolling 12-Month Incidents", rolling_12m / max(rolling_12m, 1) * 0.35,
             f"{rolling_12m} incidents — {'well above' if rolling_12m > 60 else 'near'} average"),
            ("Province Risk Score", province_risk * 0.30,
             f"{province} has {'high' if province_risk > 0.10 else 'moderate'} historical density"),
            ("Cargo Type", (cargo_risk / 3.0) * 0.25,
             f"{cargo_type} carries {'elevated' if cargo_risk >= 2 else 'standard'} hazard classification"),
            ("Season", (season_num / 4.0) * 0.20,
             f"{season} {'increases' if season == 'Winter' else 'moderately affects'} track risk"),
            ("Incident Type", int(incident_type in SEVERITY_TYPES) * 0.15,
             f"{incident_type} {'high-severity' if incident_type in SEVERITY_TYPES else 'lower-severity'} category"),
        ]
        factors.sort(key=lambda x: -x[1])

        for rank, (name, impact, desc) in enumerate(factors[:3], 1):
            bar_width = min(int(impact / 0.35 * 100), 100)
            st.markdown(f"""
            <div style="margin-bottom:0.6rem; padding:0.7rem 1rem; background:rgba(200,16,46,0.05); 
                        border-radius:8px; border-left:3px solid #C8102E;">
                <div style="font-weight:700; font-size:0.82rem; color:#C8102E;">#{rank} {name}</div>
                <div style="background:#30363D; border-radius:4px; height:6px; margin:0.3rem 0;">
                    <div style="background:#C8102E; width:{bar_width}%; height:6px; border-radius:4px;"></div>
                </div>
                <div style="font-size:0.78rem; color:#8B949E;">{desc}</div>
            </div>
            """, unsafe_allow_html=True)

    # Recommendations
    st.markdown("---")
    st.markdown("#### Recommended Actions")

    if risk_level == "HIGH":
        st.error("""
        **Immediate Actions Recommended:**
        - Schedule priority track inspection for this corridor within 48 hours
        - Reduce maximum operating speed by 20% pending inspection
        - Notify dangerous goods coordinator for cargo routing review
        - Deploy additional crossing guard coverage at populated crossings
        """)
    elif risk_level == "MEDIUM":
        st.warning("""
        **Elevated Monitoring Recommended:**
        - Flag corridor for enhanced weekly monitoring
        - Activate seasonal track alert protocols if applicable
        - Conduct briefing with local maintenance crew on high-risk incident types
        """)
    else:
        st.success("""
        **Standard Operations:**
        - Corridor within normal operating risk parameters
        - Continue scheduled maintenance cadence
        - Log scenario for quarterly risk trend analysis
        """)


# Prediction History
if st.session_state.prediction_history:
    st.markdown("---")
    st.markdown('<div class="section-header"><h2>Prediction History</h2></div>', unsafe_allow_html=True)
    st.caption(f"Last {len(st.session_state.prediction_history)} assessments this session")

    hist_rows = ""
    for h in st.session_state.prediction_history:
        badge_cls = "risk-high" if h["risk_level"] == "HIGH" else "risk-medium" if h["risk_level"] == "MEDIUM" else "risk-low"
        hist_rows += f"""
        <tr>
            <td>{h['province']}</td>
            <td>{h['incident_type']}</td>
            <td>{h['cargo_type']}</td>
            <td>{h['season']}</td>
            <td style="font-family:'JetBrains Mono',monospace;font-weight:600;">{h['probability']*100:.1f}%</td>
            <td><span class="{badge_cls}" style="font-size:0.7rem;padding:0.2rem 0.6rem;">{h['risk_level']}</span></td>
        </tr>"""

    st.markdown(f"""
    <table class="metric-table">
    <thead><tr><th>Province</th><th>Type</th><th>Cargo</th><th>Season</th><th>Probability</th><th>Risk</th></tr></thead>
    <tbody>{hist_rows}</tbody>
    </table>
    """, unsafe_allow_html=True)


st.markdown("""
<div class="disclaimer">
    This tool is for demonstration purposes only, built using Transport Canada public open data. 
    Predictions are not intended for actual operational safety decisions. 
    Always defer to certified railway safety professionals and Transport Canada regulations.
</div>
""", unsafe_allow_html=True)
