"""
preprocessing.py
================
Turns raw TSB accidents into a model-ready, leakage-free dataset.

Target (``incident_severity``): the accident killed or seriously injured at
least one person. Model inputs are the pre-incident features defined in
src/features.py; outcome columns are kept only for charts.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from src.features import (
    OUTCOME_COLUMNS,
    PRE_INCIDENT_FEATURES,
    PRIOR_SEVERE_RATE,
    PRIOR_STRENGTH,
    ROLLING_WINDOW_MONTHS,
    SEASON_BY_MONTH,
    TARGET_COLUMN,
    encode_context,
    feature_matrix,
)

logger = logging.getLogger(__name__)

PROCESSED_PATH: Path = (
    Path(__file__).resolve().parent.parent / "data" / "processed" / "incidents_featured.parquet"
)
HHMM_DIVISOR: int = 100


def to_canonical(accidents: pd.DataFrame) -> pd.DataFrame:
    """Rename raw TSB fields to the project's column names and build the target."""
    fatalities = pd.to_numeric(accidents["TotalFatalInjuries"], errors="coerce").fillna(0).astype(int)
    serious = pd.to_numeric(accidents["TotalSeriousInjuries"], errors="coerce").fillna(0).astype(int)
    occ_time = pd.to_numeric(accidents["OccTime"], errors="coerce")
    return pd.DataFrame({
        "occ_no": accidents["OccNo"],
        "date": accidents["date"],
        "hour": occ_time // HHMM_DIVISOR,
        "province": accidents["province"],
        "mile_post": accidents["SubdMileage"],
        "track_type": accidents["TrackTypeID_DisplayEng"],
        "activity": accidents["ActivityTypeID_DisplayEng"],
        "train_type": accidents["TrainTypeID_DisplayEng"],
        "approx_speed_mph": accidents["ApproxTrainSpeed_MPH"],
        "num_cars": accidents["NumCars"],
        "num_locomotives": accidents["NumLocomotives"],
        "train_tonnage": accidents["TrainTonnage"],
        "accident_type": accidents["AccIncTypeID_DisplayEng"],
        "fatalities": fatalities,
        "serious_injuries": serious,
        "minor_injuries": pd.to_numeric(accidents["TotalMinorInjuries"], errors="coerce").fillna(0).astype(int),
        TARGET_COLUMN: ((fatalities >= 1) | (serious >= 1)).astype(int),
        "_source_file": accidents["_source_file"],
    }).sort_values(["date", "occ_no"], kind="stable").reset_index(drop=True)


def compute_past_province_history(df: pd.DataFrame) -> pd.DataFrame:
    """
    Per-province history using only accidents on STRICTLY EARLIER dates:

    - rolling_12m_incidents: accidents in the 12 months before this date
    - prior_incidents_province: all accidents before this date
    - prior_severe_rate_province: smoothed share of those that were severe

    No row sees itself, same-day accidents, or the future.
    """
    out = pd.DataFrame(index=df.index, columns=[
        "rolling_12m_incidents", "prior_incidents_province", "prior_severe_rate_province",
    ], dtype=float)
    for _, grp in df.groupby("province", sort=False):
        grp = grp.sort_values("date", kind="stable")
        dates = grp["date"].to_numpy()
        severe_prefix = np.concatenate([[0], np.cumsum(grp[TARGET_COLUMN].to_numpy())])
        n_before = np.searchsorted(dates, dates, side="left")
        window_start = (grp["date"] - pd.DateOffset(months=ROLLING_WINDOW_MONTHS)).to_numpy()
        n_before_window = np.searchsorted(dates, window_start, side="left")
        out.loc[grp.index, "prior_incidents_province"] = n_before
        out.loc[grp.index, "rolling_12m_incidents"] = n_before - n_before_window
        out.loc[grp.index, "prior_severe_rate_province"] = (
            (severe_prefix[n_before] + PRIOR_STRENGTH * PRIOR_SEVERE_RATE)
            / (n_before + PRIOR_STRENGTH)
        )
    return out.astype({"rolling_12m_incidents": int, "prior_incidents_province": int})


def add_weather(df: pd.DataFrame, weather: pd.DataFrame) -> pd.DataFrame:
    """Join daily Open-Meteo weather for the accident's province and date."""
    cols = ["province", "date", "wx_temp_c", "wx_precip_mm", "wx_wind_kmh"]
    return df.merge(weather[cols], on=["province", "date"], how="left", validate="many_to_one")


def run_preprocessing_pipeline(accidents: pd.DataFrame, weather: pd.DataFrame,
                               save: bool = True) -> pd.DataFrame:
    """Raw TSB accidents + daily weather -> processed, feature-enriched frame."""
    df = to_canonical(accidents)
    df = encode_context(df)
    df = df.join(compute_past_province_history(df))
    df = add_weather(df, weather)
    df["season"] = df["month"].map(SEASON_BY_MONTH)
    if save:
        PROCESSED_PATH.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(PROCESSED_PATH, index=False)
        logger.info("Saved processed data -> %s", PROCESSED_PATH)
    logger.info("Processed %d accidents; severe share %.4f", len(df), df[TARGET_COLUMN].mean())
    return df


def get_feature_columns(df: pd.DataFrame) -> list[str]:
    """
    The model's inputs: the explicit pre-incident allow-list.

    Raises
    ------
    ValueError
        If an outcome column is in the list, or a feature is missing.
    """
    leaked = OUTCOME_COLUMNS.intersection(PRE_INCIDENT_FEATURES)
    if leaked:
        raise ValueError(f"Outcome columns in feature list: {sorted(leaked)}")
    return list(feature_matrix(df).columns)


def build_model_dataset(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """(X, y). NaNs are left in X: models impute inside each CV fold."""
    get_feature_columns(df)
    return feature_matrix(df), df[TARGET_COLUMN].astype(int)
