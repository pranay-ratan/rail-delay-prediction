"""Model results: CV leaderboard, time holdout, leakage checks, ROC, SHAP."""

from __future__ import annotations

import plotly.graph_objects as go
import streamlit as st

from common import COLORS, PLOT_LAYOUT, layout, setup_page

METRIC_COLS = ["model", "roc_auc", "pr_auc", "f1", "precision", "recall", "balanced_accuracy", "accuracy"]
LABELS = {"model": "Model", "roc_auc": "ROC-AUC", "pr_auc": "PR-AUC", "f1": "F1 (severe)",
          "precision": "Precision", "recall": "Recall", "balanced_accuracy": "Balanced acc.", "accuracy": "Accuracy"}

res = setup_page("Model Results")
summary = res.summary
final = summary["final_model"]
base_rate = summary["positive_rate"]


def table(df):
    out = df[[c for c in METRIC_COLS if c in df.columns]].sort_values("roc_auc", ascending=False)
    return out.rename(columns=LABELS).round(3)


st.title("Model Results")
st.markdown(
    f"Target: fatality or serious injury, **{base_rate:.1%}** of accidents. Because the classes are imbalanced, "
    f"always predicting \"not severe\" already gets {1 - base_rate:.1%} accuracy, so **ROC-AUC and PR-AUC** "
    f"(random guessing: 0.5 and {base_rate:.3f}) are the metrics to judge by."
)

st.subheader("10-fold stratified cross-validation")
st.caption(summary["cv"])
st.dataframe(table(res.cv_metrics), hide_index=True, use_container_width=True)

st.subheader("Time-based holdout")
st.caption(f"{summary['time_holdout']}; XGBoost was tuned on the training years only")
st.dataframe(table(res.holdout_metrics), hide_index=True, use_container_width=True)

st.subheader("Leakage checks")
st.dataframe(res.leakage_checks[["model", "roc_auc", "pr_auc", "n_rows", "note"]].round(3),
             hide_index=True, use_container_width=True)
st.caption("TSB records more detail for serious accidents. A model that only sees which fields are blank reaches "
           "the AUC above; the tuned model scores almost the same on fully complete records, so it is not relying on blanks.")

left, right = st.columns(2)
with left:
    fig = go.Figure()
    for i, (name, c) in enumerate(res.roc_curves.items()):
        fig.add_scatter(x=c["fpr"], y=c["tpr"], mode="lines", name=f"{name} ({c['auc']:.3f})",
                        line=dict(color=COLORS[i % len(COLORS)], width=2))
    fig.add_scatter(x=[0, 1], y=[0, 1], mode="lines", name="Chance", line=dict(color="#6E7681", dash="dash"))
    fig.update_layout(**PLOT_LAYOUT, height=480, title="ROC curves (10-fold out-of-fold)",
                      legend=dict(orientation="h", y=-0.25))
    st.plotly_chart(fig, use_container_width=True)
with right:
    shap = res.shap_importance.head(15).iloc[::-1]
    fig = go.Figure(go.Bar(x=shap["mean_abs_shap"], y=shap["feature"], orientation="h", marker_color="#C8102E"))
    fig.update_layout(**layout(height=480, title=f"SHAP: mean |value| for {final}",
                               margin=dict(l=180, r=20, t=40, b=40)))
    st.plotly_chart(fig, use_container_width=True)

st.markdown(
    "**Interpretation:** fatal and serious-injury accidents cluster on main-line track with trains proceeding at "
    "speed (crossing and trespasser accidents); the far more common yard and switching accidents are rarely severe."
)
