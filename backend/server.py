"""
Rail Risk Intelligence API (FastAPI).

Every number served here is computed from the real TSB accident data or read
from the training run's outputs (backend/data/, synced by
scripts/sync_backend.py). Nothing is hard-coded or simulated; if the model
is missing, /api/predict returns 503 instead of guessing.
"""

from __future__ import annotations

import logging
import os
import sys
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

sys.path.insert(0, str(Path(__file__).resolve().parent))

from insights import SEASON_ORDER, compute_insights  # noqa: E402
from insights import by_type as accidents_by_type  # noqa: E402
from predictor import (  # noqa: E402
    PredictionRequest, build_features, calibrated_probability, explain, risk_level,
    similar_incidents,
)
from resources import load_resources  # noqa: E402

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

TOP_ROLLING_PROVINCES = 6
ROLLING_MONTHS = 12
PERCENT = 100
CORRELATION_FEATURES = [
    "incident_severity", "year", "month", "hour", "mile_post", "approx_speed_mph",
    "num_cars", "num_locomotives", "train_tonnage", "wx_temp_c", "wx_precip_mm",
    "wx_wind_kmh", "rolling_12m_incidents", "prior_incidents_province",
    "prior_severe_rate_province", "track_main", "activity_proceeding", "train_passenger",
]

R = load_resources()

app = FastAPI(title="Rail Risk Intelligence API")
# Same-origin on Vercel; CORS only matters for local development.
app.add_middleware(
    CORSMiddleware,
    allow_origins=[o for o in os.getenv("CORS_ORIGINS", "http://localhost:3000").split(",") if o],
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)


def _data() -> pd.DataFrame:
    if R.incidents is None:
        raise HTTPException(status_code=503, detail="Incident data not loaded")
    return R.incidents


def _records(df: pd.DataFrame) -> list[dict[str, Any]]:
    return df.astype(object).where(df.notna(), None).to_dict("records")


def _title(s: pd.Series) -> pd.Series:
    return s.astype("string").str.title()


@app.get("/api/health")
def health() -> dict[str, Any]:
    return {
        "status": "healthy" if R.model is not None and R.incidents is not None else "degraded",
        "model_loaded": R.model is not None,
        "data_loaded": R.incidents is not None,
        "model": R.summary.get("final_model"),
        "n_features": len(R.feature_names or []),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


@app.get("/api/stats")
def stats() -> dict[str, Any]:
    df = _data()
    final = R.summary.get("final_model")
    cv = R.cv_metrics.set_index("model") if R.cv_metrics is not None else pd.DataFrame()
    ho = R.holdout_metrics.set_index("model") if R.holdout_metrics is not None else pd.DataFrame()
    return {
        "total_incidents": int(len(df)),
        "provinces_covered": int(df["province"].nunique()),
        "year_range": f"{int(df['year'].min())}-{int(df['year'].max())}",
        "high_risk_pct": round(float(df["incident_severity"].mean()) * PERCENT, 1),
        "feature_count": int(R.summary.get("n_features", len(R.feature_names or []))),
        "best_model": final,
        "best_auc": round(float(cv.loc[final, "roc_auc"]) * PERCENT, 1) if final in cv.index else None,
        "holdout_auc": round(float(ho.loc[final, "roc_auc"]) * PERCENT, 1) if final in ho.index else None,
        "pr_auc": round(float(cv.loc[final, "pr_auc"]), 3) if final in cv.index else None,
        "incident_types": int(df["accident_type"].nunique()),
        "data_source": R.summary.get("data_source"),
        "target": R.summary.get("target"),
    }


@app.get("/api/provinces")
def provinces() -> dict[str, Any]:
    df = _data()
    grouped = df.groupby("province")
    out = pd.DataFrame({
        "incidents": grouped.size(),
        "high_risk": grouped["incident_severity"].sum(),
        "top_incident_type": grouped["accident_type"].agg(lambda s: s.mode().iat[0]).str.title(),
    }).reset_index()
    out["risk_pct"] = (out["high_risk"] / out["incidents"] * PERCENT).round(1)
    return {"provinces": _records(out.sort_values("incidents", ascending=False))}


@app.get("/api/incidents/annual")
def annual() -> dict[str, Any]:
    g = _data().groupby("year")["incident_severity"].agg(total="size", high_risk="sum")
    return {
        "years": g.index.astype(int).tolist(),
        "total": g["total"].astype(int).tolist(),
        "high_risk": g["high_risk"].astype(int).tolist(),
        "low_risk": (g["total"] - g["high_risk"]).astype(int).tolist(),
    }


@app.get("/api/incidents/by-season")
def by_season() -> dict[str, Any]:
    g = _data().groupby("season")["incident_severity"].agg(total="size", high_risk="sum")
    g = g.reindex(SEASON_ORDER)
    return {
        "seasons": SEASON_ORDER,
        "total": g["total"].astype(int).tolist(),
        "high_risk": g["high_risk"].astype(int).tolist(),
        "severe_rate_pct": (g["high_risk"] / g["total"] * PERCENT).round(1).tolist(),
    }


def _by_type() -> pd.DataFrame:
    return accidents_by_type(_data())


@app.get("/api/incidents/by-type")
def by_type() -> dict[str, Any]:
    g = _by_type()
    return {"types": g["type"].tolist(), "total": g["total"].astype(int).tolist(),
            "high_risk": g["high"].astype(int).tolist()}


@app.get("/api/incidents/severity-by-type")
def severity_by_type() -> dict[str, Any]:
    return {"data": _records(_by_type()[["type", "high", "low", "total"]])}


@app.get("/api/incidents/heatmap")
def heatmap() -> dict[str, Any]:
    df = _data().assign(type=lambda d: _title(d["accident_type"]))
    top_types = df["type"].value_counts().index[:12]
    pivot = df[df["type"].isin(top_types)].pivot_table(
        index="province", columns="type", values="year", aggfunc="count", fill_value=0)
    return {"provinces": pivot.index.tolist(), "incident_types": pivot.columns.tolist(),
            "values": pivot.values.astype(int).tolist()}


@app.get("/api/incidents/correlation")
def correlation() -> dict[str, Any]:
    corr = _data()[CORRELATION_FEATURES].corr().round(3)
    return {"features": corr.columns.tolist(),
            "values": corr.astype(object).where(corr.notna(), None).values.tolist()}


@app.get("/api/incidents/rolling")
def rolling() -> dict[str, Any]:
    df = _data()
    result = {}
    for prov in df["province"].value_counts().index[:TOP_ROLLING_PROVINCES]:
        monthly = df[df["province"] == prov].set_index("date").resample("MS").size()
        smooth = monthly.rolling(ROLLING_MONTHS, min_periods=ROLLING_MONTHS).sum().dropna()
        result[prov] = {"dates": smooth.index.strftime("%Y-%m").tolist(),
                        "values": smooth.astype(int).tolist()}
    return {"provinces": result, "description": "Accidents in the preceding 12 months"}


@lru_cache(maxsize=1)
def _insights() -> dict[str, Any]:
    return compute_insights(_data(), R.shap_importance)


@app.get("/api/insights")
def insights() -> dict[str, Any]:
    return _insights()


def _metric_rows(df: pd.DataFrame | None) -> list[dict[str, Any]]:
    if df is None:
        return []
    ordered = df.sort_values("roc_auc", ascending=False).round(4).rename(columns={"model": "name"})
    return _records(ordered)


@app.get("/api/models/metrics")
def model_metrics() -> dict[str, Any]:
    return {
        "models": _metric_rows(R.cv_metrics),
        "holdout": _metric_rows(R.holdout_metrics),
        "leakage_checks": _records(R.leakage_checks.round(4)) if R.leakage_checks is not None else [],
        "final_model": R.summary.get("final_model"),
        "cv": R.summary.get("cv"),
        "time_holdout": R.summary.get("time_holdout"),
        "positive_rate": R.summary.get("positive_rate"),
    }


@app.get("/api/models/roc")
def model_roc() -> dict[str, Any]:
    return {"curves": R.roc_curves, "source": "10-fold out-of-fold predictions"}


@app.get("/api/models/feature-importance")
def feature_importance() -> dict[str, Any]:
    if R.shap_importance is None:
        raise HTTPException(status_code=503, detail="SHAP importance not available")
    top = R.shap_importance.head(15)
    return {
        "features": top["feature"].tolist(),
        "importances": top["mean_abs_shap"].round(4).tolist(),
        "method": f"Mean |SHAP value| (log-odds) for {R.summary.get('final_model')}",
    }


@app.post("/api/predict")
def predict(req: PredictionRequest) -> dict[str, Any]:
    if R.model is None or R.feature_names is None:
        raise HTTPException(status_code=503, detail="Model not loaded; no prediction made")
    if not R.calibration:
        raise HTTPException(status_code=503, detail="Calibration table not loaded")
    incidents = _data()
    X, notes = build_features(req, incidents, R.feature_names)
    score = float(R.model.predict_proba(X)[0, 1])
    probability = calibrated_probability(score, R.calibration)
    base_rate = float(R.summary.get("positive_rate", incidents["incident_severity"].mean()))
    contributions, bias = explain(R.model, X)
    return {
        "source": "model",
        "model": R.summary.get("final_model"),
        "probability": round(probability, 4),
        "model_score": round(score, 4),
        "base_rate": round(base_rate, 4),
        "relative_risk": round(probability / base_rate, 2),
        "risk_level": risk_level(probability, base_rate),
        "contributions": contributions,
        "contribution_units": "log-odds (XGBoost SHAP)",
        "baseline_logodds": round(bias, 4),
        "similar_incidents": similar_incidents(req, incidents),
        "inputs": notes,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
