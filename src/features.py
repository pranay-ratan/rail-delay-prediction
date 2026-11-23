"""
features.py
===========
Single source of truth for the model's input features. Used by training
(src/preprocessing.py) and, as an identical copy, by the API
(backend/rail_features.py, kept in sync by scripts/sync_backend.py).

Only information available BEFORE an accident is allowed in: when, where,
what the train was doing, the train itself, weather that day, and history
computed from earlier dates. Accident type, injuries, damage, etc. are
outcomes and are excluded.

Depends only on numpy and pandas.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TARGET_COLUMN = "incident_severity"

PROVINCE_ORDER: tuple[str, ...] = (
    "Alberta", "British Columbia", "Manitoba", "New Brunswick",
    "Newfoundland and Labrador", "Northwest Territories", "Nova Scotia",
    "Nunavut", "Ontario", "Prince Edward Island", "Quebec",
    "Saskatchewan", "Yukon",
)

# Coarse categories. Missing / rare values fall into "other" (all flags 0),
# so a blank field is never its own category: TSB fills these fields more
# often for serious accidents, which would otherwise leak the target.
TRACK_TYPES: dict[str, str] = {"MAIN": "main", "YARD": "yard"}
ACTIVITY_TYPES: dict[str, str] = {
    "PROCEEDING": "proceeding",
    "SWITCHING": "switching",
    "SWITCHING SHOVING": "switching",
    "SWITCHING PULLING": "switching",
    "SWITCHING HUMPING": "switching",
    "STANDING": "standing",
}
TRAIN_TYPES: dict[str, str] = {
    "FREIGHT": "freight",
    "PASS. CONVENTIONAL": "passenger",
    "PASS. LRC": "passenger",
    "CUT OF CARS": "cars_only",
    "SINGLE CAR": "cars_only",
    "TRACK UNIT/HI-RAIL": "track_unit",
    "TRACK UNIT/OTHER": "track_unit",
}
TRACK_FLAGS: tuple[str, ...] = ("main", "yard")
ACTIVITY_FLAGS: tuple[str, ...] = ("proceeding", "switching", "standing")
TRAIN_FLAGS: tuple[str, ...] = ("freight", "passenger", "cars_only", "track_unit")

# Values TSB uses as "unknown" placeholders, and plausible upper bounds.
NUMERIC_LIMITS: dict[str, float] = {
    "approx_speed_mph": 120,
    "num_cars": 400,
    "num_locomotives": 20,
    "train_tonnage": 40_000,
}

WEEKEND_DAYS: frozenset[int] = frozenset({5, 6})
SEASON_BY_MONTH: dict[int, str] = {
    12: "Winter", 1: "Winter", 2: "Winter", 3: "Spring", 4: "Spring", 5: "Spring",
    6: "Summer", 7: "Summer", 8: "Summer", 9: "Fall", 10: "Fall", 11: "Fall",
}

# Past-only province severity rate: a province with no history starts at
# PRIOR_SEVERE_RATE and moves towards its own record as accidents accumulate.
PRIOR_SEVERE_RATE: float = 0.1
PRIOR_STRENGTH: float = 20.0
ROLLING_WINDOW_MONTHS: int = 12

HISTORY_FEATURES: tuple[str, ...] = (
    "rolling_12m_incidents", "prior_incidents_province", "prior_severe_rate_province",
)
WEATHER_FEATURES: tuple[str, ...] = ("wx_temp_c", "wx_precip_mm", "wx_wind_kmh")
NUMERIC_TRAIN_FEATURES: tuple[str, ...] = tuple(NUMERIC_LIMITS)


def _slug(name: str) -> str:
    return name.lower().replace(" ", "_")


PROVINCE_FEATURES: tuple[str, ...] = tuple(f"province_{_slug(p)}" for p in PROVINCE_ORDER)

PRE_INCIDENT_FEATURES: tuple[str, ...] = (
    ("year", "month", "hour", "is_weekend", "mile_post")
    + tuple(f"track_{t}" for t in TRACK_FLAGS)
    + tuple(f"activity_{a}" for a in ACTIVITY_FLAGS)
    + tuple(f"train_{t}" for t in TRAIN_FLAGS)
    + NUMERIC_TRAIN_FEATURES
    + WEATHER_FEATURES
    + HISTORY_FEATURES
    + PROVINCE_FEATURES
)

# Outcomes / after-the-fact fields. Kept in the processed data for charts,
# never used as model inputs.
OUTCOME_COLUMNS: frozenset[str] = frozenset({
    "accident_type", "fatalities", "serious_injuries", "minor_injuries", TARGET_COLUMN,
})


def flag_columns(values: pd.Series, mapping: dict[str, str],
                 flags: tuple[str, ...], prefix: str) -> pd.DataFrame:
    """Map raw TSB labels to coarse categories and one-hot them (other = all 0)."""
    category = values.astype("string").str.strip().str.upper().map(mapping)
    return pd.DataFrame(
        {f"{prefix}_{f}": (category == f).fillna(False).astype(int) for f in flags},
        index=values.index,
    )


def province_one_hot(province: pd.Series) -> pd.DataFrame:
    """One-hot encode province with a fixed column set (see PROVINCE_ORDER)."""
    return pd.DataFrame(
        {f"province_{_slug(p)}": (province == p).astype(int) for p in PROVINCE_ORDER},
        index=province.index,
    )


def clean_numeric(df: pd.DataFrame) -> pd.DataFrame:
    """Coerce train numerics; out-of-range placeholders (999, 99999...) -> NaN."""
    cleaned = {
        col: pd.to_numeric(df[col], errors="coerce").where(lambda s, hi=hi: s.between(0, hi))
        for col, hi in NUMERIC_LIMITS.items()
    }
    return df.assign(**cleaned)


def encode_context(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add every non-history, non-weather feature from raw context columns:
    ``date``, ``hour``, ``province``, ``mile_post``, ``track_type``,
    ``activity``, ``train_type`` and the numeric train columns.
    """
    dates = pd.to_datetime(df["date"])
    encoded = clean_numeric(df).assign(
        year=dates.dt.year,
        month=dates.dt.month,
        is_weekend=dates.dt.day_of_week.isin(WEEKEND_DAYS).astype(int),
        hour=pd.to_numeric(df["hour"], errors="coerce").where(lambda h: h.between(0, 23)),
        mile_post=pd.to_numeric(df["mile_post"], errors="coerce"),
    )
    return pd.concat([
        encoded,
        flag_columns(df["track_type"], TRACK_TYPES, TRACK_FLAGS, "track"),
        flag_columns(df["activity"], ACTIVITY_TYPES, ACTIVITY_FLAGS, "activity"),
        flag_columns(df["train_type"], TRAIN_TYPES, TRAIN_FLAGS, "train"),
        province_one_hot(df["province"]),
    ], axis=1)


def api_choices(mapping: dict[str, str]) -> dict[str, str | None]:
    """API choice -> a representative TSB label (first one listed), plus 'other'."""
    choices: dict[str, str | None] = {}
    for label, category in mapping.items():
        choices.setdefault(category, label)
    return {**choices, "other": None}


TRACK_CHOICES = api_choices(TRACK_TYPES)
ACTIVITY_CHOICES = api_choices(ACTIVITY_TYPES)
TRAIN_CHOICES = api_choices(TRAIN_TYPES)


def history_as_of(dates: pd.Series, severe: pd.Series, as_of: pd.Timestamp) -> dict[str, float]:
    """
    History features for ONE province at ``as_of``, from that province's
    accident dates and outcomes. Same definition as training: only accidents
    strictly before ``as_of`` count.
    """
    before = dates < as_of
    window = before & (dates >= as_of - pd.DateOffset(months=ROLLING_WINDOW_MONTHS))
    n_before = int(before.sum())
    return {
        "rolling_12m_incidents": int(window.sum()),
        "prior_incidents_province": n_before,
        "prior_severe_rate_province": float(
            (severe[before].sum() + PRIOR_STRENGTH * PRIOR_SEVERE_RATE) / (n_before + PRIOR_STRENGTH)
        ),
    }


def feature_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Select model inputs in the canonical order (float, NaN = unknown)."""
    missing = [c for c in PRE_INCIDENT_FEATURES if c not in df.columns]
    if missing:
        raise ValueError(f"Missing feature columns: {missing}")
    return df[list(PRE_INCIDENT_FEATURES)].astype(float).replace([np.inf, -np.inf], np.nan)
