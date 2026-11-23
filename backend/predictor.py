"""
Builds the model's feature row from a scenario, using the SAME encoder as
training (rail_features.py is a copy of src/features.py), and explains the
prediction with XGBoost's own SHAP contributions.
"""

from __future__ import annotations

import calendar
from datetime import date
from typing import Any, Literal, Optional

import numpy as np
import pandas as pd
import xgboost as xgb
from pydantic import BaseModel, Field, field_validator

from rail_features import (
    ACTIVITY_CHOICES, ACTIVITY_FLAGS, PROVINCE_FEATURES, PROVINCE_ORDER, TRACK_CHOICES,
    TRACK_FLAGS, TRAIN_CHOICES, TRAIN_FLAGS, TRAIN_TYPES, TRACK_TYPES, WEATHER_FEATURES,
    WEEKEND_DAYS, encode_context, feature_matrix, history_as_of,
)

MIN_YEAR, MAX_YEAR = 2001, 2030
MID_MONTH = 15
TOP_CONTRIBUTIONS = 8
SIMILAR_LIMIT = 5
HIGH_RISK_MULTIPLE = 2.0

# Grouped one-hot features are explained as one factor.
FEATURE_GROUPS: dict[str, str] = {
    **{c: "Province" for c in PROVINCE_FEATURES},
    **{f"track_{t}": "Track type" for t in TRACK_FLAGS},
    **{f"activity_{a}": "Train activity" for a in ACTIVITY_FLAGS},
    **{f"train_{t}": "Train type" for t in TRAIN_FLAGS},
}
FEATURE_LABELS: dict[str, str] = {
    "year": "Year", "month": "Month", "hour": "Hour of day", "is_weekend": "Weekend",
    "mile_post": "Mile post", "approx_speed_mph": "Train speed",
    "num_cars": "Number of cars", "num_locomotives": "Locomotives",
    "train_tonnage": "Train tonnage", "wx_temp_c": "Temperature",
    "wx_precip_mm": "Precipitation", "wx_wind_kmh": "Wind",
    "rolling_12m_incidents": "Province accidents, past 12 months",
    "prior_incidents_province": "Province accidents, all prior",
    "prior_severe_rate_province": "Province past severe rate",
}


class PredictionRequest(BaseModel):
    """A scenario described only with information known before an accident."""

    province: str = "Ontario"
    year: int = Field(2025, ge=MIN_YEAR, le=MAX_YEAR)
    month: int = Field(1, ge=1, le=12)
    hour: int = Field(12, ge=0, le=23)
    is_weekend: bool = False
    track_type: Literal["main", "yard", "other"] = "main"
    activity: Literal["proceeding", "switching", "standing", "other"] = "proceeding"
    train_type: Literal["freight", "passenger", "cars_only", "track_unit", "other"] = "freight"
    approx_speed_mph: Optional[float] = Field(None, ge=0, le=120)
    num_cars: Optional[int] = Field(None, ge=0, le=400)
    num_locomotives: Optional[int] = Field(None, ge=0, le=20)
    train_tonnage: Optional[float] = Field(None, ge=0, le=40_000)
    mile_post: Optional[float] = Field(None, ge=0, le=2_000)
    wx_temp_c: Optional[float] = Field(None, ge=-60, le=50)
    wx_precip_mm: Optional[float] = Field(None, ge=0, le=300)
    wx_wind_kmh: Optional[float] = Field(None, ge=0, le=200)

    @field_validator("province")
    @classmethod
    def _known_province(cls, value: str) -> str:
        if value not in PROVINCE_ORDER:
            raise ValueError(f"province must be one of {list(PROVINCE_ORDER)}")
        return value


def scenario_date(year: int, month: int, weekend: bool) -> pd.Timestamp:
    """A date in the month, nearest mid-month, on a weekend day if requested."""
    days = range(1, calendar.monthrange(year, month)[1] + 1)
    matching = [d for d in days if (date(year, month, d).weekday() in WEEKEND_DAYS) == weekend]
    return pd.Timestamp(year, month, min(matching, key=lambda d: abs(d - MID_MONTH)))


def _weather(req: PredictionRequest, incidents: pd.DataFrame) -> tuple[dict[str, float], list[str]]:
    """User weather, else the median Open-Meteo value for that province & month."""
    same = incidents[(incidents["province"] == req.province) & (incidents["month"] == req.month)]
    pool = same if len(same) else incidents[incidents["month"] == req.month]
    values, filled = {}, []
    for col in WEATHER_FEATURES:
        given = getattr(req, col)
        if given is None:
            filled.append(col)
        values[col] = float(given) if given is not None else float(pool[col].median())
    return values, filled


def build_features(req: PredictionRequest, incidents: pd.DataFrame,
                   feature_names: list[str]) -> tuple[pd.DataFrame, dict[str, Any]]:
    when = scenario_date(req.year, req.month, req.is_weekend)
    raw = pd.DataFrame([{
        "date": when, "hour": req.hour, "province": req.province, "mile_post": req.mile_post,
        "track_type": TRACK_CHOICES[req.track_type],
        "activity": ACTIVITY_CHOICES[req.activity],
        "train_type": TRAIN_CHOICES[req.train_type],
        "approx_speed_mph": req.approx_speed_mph, "num_cars": req.num_cars,
        "num_locomotives": req.num_locomotives, "train_tonnage": req.train_tonnage,
    }])
    # History is "as of" the scenario date, but never later than the data:
    # beyond it the counts would just drop to zero.
    as_of = min(when, incidents["date"].max() + pd.Timedelta(days=1))
    prov = incidents[incidents["province"] == req.province]
    history = history_as_of(prov["date"], prov["incident_severity"], as_of)
    weather, weather_filled = _weather(req, incidents)
    X = feature_matrix(encode_context(raw).assign(**history, **weather))
    if list(X.columns) != feature_names:
        raise ValueError("Feature order does not match the trained model")
    notes = {
        "history_as_of": as_of.date().isoformat(),
        "weather_filled_from_climatology": weather_filled,
        "imputed_by_model": [c for c in X.columns if pd.isna(X.at[0, c])],
        **history,
    }
    return X, notes


def calibrated_probability(score: float, calibration: list[dict[str, float]]) -> float:
    """Observed severe rate among historical accidents with a similar score."""
    for band in calibration:
        if score <= band["score_max"]:
            return band["observed_rate"]
    return calibration[-1]["observed_rate"]


def risk_level(probability: float, base_rate: float) -> str:
    if probability >= HIGH_RISK_MULTIPLE * base_rate:
        return "HIGH"
    return "MEDIUM" if probability >= base_rate else "LOW"


def explain(model: Any, X: pd.DataFrame) -> tuple[list[dict[str, Any]], float]:
    """Per-factor SHAP contributions (log-odds) via XGBoost pred_contribs."""
    imputed = model[:-1].transform(X)
    contribs = model[-1].get_booster().predict(xgb.DMatrix(np.asarray(imputed)), pred_contribs=True)[0]
    per_feature = pd.Series(contribs[:-1], index=X.columns)
    grouped = per_feature.groupby(lambda c: FEATURE_GROUPS.get(c, FEATURE_LABELS.get(c, c))).sum()
    top = grouped.reindex(grouped.abs().sort_values(ascending=False).index).head(TOP_CONTRIBUTIONS)
    factors = [
        {"feature": name, "value": round(float(v), 4), "direction": "increase" if v > 0 else "decrease"}
        for name, v in top.items()
    ]
    return factors, float(contribs[-1])


def similar_incidents(req: PredictionRequest, incidents: pd.DataFrame) -> list[dict[str, Any]]:
    """Most recent real accidents with the same province, track and train category."""
    track = incidents["track_type"].astype("string").str.upper().map(TRACK_TYPES).fillna("other")
    train = incidents["train_type"].astype("string").str.upper().map(TRAIN_TYPES).fillna("other")
    match = incidents[(incidents["province"] == req.province)
                      & (track == req.track_type) & (train == req.train_type)]
    rows = match.sort_values("date", ascending=False).head(SIMILAR_LIMIT)
    return [{
        "date": r.date.date().isoformat(),
        "province": r.province,
        "type": str(r.accident_type).title(),
        "train": str(r.train_type).title() if pd.notna(r.train_type) else "Unknown",
        "fatalities": int(r.fatalities),
        "serious_injuries": int(r.serious_injuries),
        "severity": "SEVERE" if r.incident_severity == 1 else "NOT SEVERE",
    } for r in rows.itertuples()]
