"""
Rail Risk Intelligence: Streamlit dashboard (home page).

Run locally from the repo root:  streamlit run website/app.py
"""

from __future__ import annotations

import plotly.graph_objects as go
import streamlit as st

from common import PLOT_LAYOUT, setup_page
from insights import compute_insights

res = setup_page("Overview")
df = res.incidents
summary = res.summary
cv = res.cv_metrics.set_index("model")
holdout = res.holdout_metrics.set_index("model")
final = summary["final_model"]
ins = compute_insights(df, res.shap_importance)

st.title("Canadian Rail Accident Severity Prediction")
st.markdown(
    f"Given that a rail accident happens, how likely is it to **kill or seriously injure** someone? "
    f"Trained on **{len(df):,} accidents** reported to the Transportation Safety Board of Canada "
    f"({summary['year_range'][0]}-{summary['year_range'][1]}), using only information known *before* the accident."
)

c1, c2, c3, c4 = st.columns(4)
c1.metric("Accidents", f"{len(df):,}", f"{summary['year_range'][0]}-{summary['year_range'][1]}", delta_color="off")
c2.metric("Severe", f"{ins['severe_rate_pct']}%", "fatality or serious injury", delta_color="off")
c3.metric(f"ROC-AUC ({final})", f"{cv.loc[final, 'roc_auc']:.3f}", "10-fold stratified CV", delta_color="off")
c4.metric("ROC-AUC, unseen 2021-25", f"{holdout.loc[final, 'roc_auc']:.3f}", "trained on 2001-2020", delta_color="off")

annual = df.groupby("year")["incident_severity"].agg(total="size", severe="sum")
fig = go.Figure([
    go.Bar(x=annual.index, y=annual["severe"], name="Severe", marker_color="rgba(200,16,46,0.75)"),
    go.Bar(x=annual.index, y=annual["total"] - annual["severe"], name="Not severe", marker_color="rgba(26,58,92,0.75)"),
])
fig.update_layout(**PLOT_LAYOUT, barmode="stack", height=360, title="Accidents per year",
                  legend=dict(orientation="h", y=-0.15))
st.plotly_chart(fig, use_container_width=True)

seasons = ins["severe_rate_by_season_pct"]
st.subheader("Key findings (computed from the data)")
st.markdown(
    f"""
- Accidents fell **{abs(ins['change_peak_to_last_pct'])}%** from the {ins['peak_year']} peak ({ins['peak_count']:,}) to {ins['last_year']} ({ins['last_count']:,}).
- Severity is concentrated in a few accident types: {', '.join(f"{t['type']} {t['severe_rate_pct']}%" for t in ins['deadliest_types'])}.
  The most common type, {ins['most_common_type']['type']} ({ins['most_common_type']['share_pct']}%), is almost never severe.
- Summer has the highest severe rate ({seasons['Summer']}%), winter the lowest ({seasons['Winter']}%).
- Strongest model drivers (SHAP): {', '.join(ins['top_shap_features'])}.
"""
)

st.subheader("How it works")
st.markdown(
    f"""
1. **Data**: TSB Rail Occurrence Database (accidents only) + daily Open-Meteo weather for the accident date.
2. **Target**: at least one fatality or serious injury ({ins['severe_rate_pct']}% of accidents).
3. **Features ({summary['n_features']})**: time, province, track type, train activity, train type and size, speed,
   weather and *past-only* provincial accident history. Accident type, injuries and damage are outcomes, so they are excluded.
4. **Models**: Logistic Regression, Random Forest, XGBoost, LightGBM vs a majority-class baseline,
   compared with 10-fold stratified CV; XGBoost tuned with 10-fold CV and checked on a 2021-2025 holdout.
5. **Explainability**: SHAP values for every prediction.

Use the pages in the sidebar to explore the data, the model results and live predictions.
"""
)
