"""
preprocessing.py
================
Feature engineering and data cleaning pipeline for the Canadian Rail
Incident & Delay Prediction project.

15+ engineered features including temporal, geographic, behavioral,
interaction, and rolling statistics.

Author : Pranay Ratan | SFU Data Science
Project: Canadian Rail Incident & Delay Prediction
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# CONSTANTS
# ---------------------------------------------------------------------------
SEASONS: dict[int, str] = {
    12: "Winter", 1: "Winter", 2: "Winter",
    3: "Spring", 4: "Spring", 5: "Spring",
    6: "Summer", 7: "Summer", 8: "Summer",
    9: "Fall", 10: "Fall", 11: "Fall",
}

SEVERITY_INCIDENT_TYPES: set[str] = {
    "Derailment", "Main Track Train Collision", "Dangerous Goods Release",
    "Employee Fatality", "Fire or Explosion"
}

# Cargo risk encoding: high = 3, medium = 2, low = 1
CARGO_RISK_MAP: dict[str, int] = {
    "Dangerous Goods": 3,
    "Crude Oil": 3,
    "Coal": 2,
    "Potash": 2,
    "Grain": 1,
    "General Freight": 1,
    "Intermodal": 1,
    "Passenger": 2,
}

HIGH_RISK_PROVINCES: set[str] = {
    "Ontario", "Quebec", "Alberta", "British Columbia", "Saskatchewan"
}

REFERENCE_YEAR: int = 2000


# ---------------------------------------------------------------------------
# STEP 1 — CLEAN & STANDARDIZE
# ---------------------------------------------------------------------------

def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Standardize a raw DataFrame from the Transport Canada loader.

    Actions taken:
    - Rename columns to a canonical set
    - Parse date columns
    - Standardize province values (expand abbreviations)
    - Strip whitespace from string columns

    Parameters
    ----------
    df : pd.DataFrame
        Raw DataFrame with snake_case column names.

    Returns
    -------
    pd.DataFrame
        Standardized DataFrame.
    """
    from src.data_loader import PROVINCE_CODE_MAP  # local import to avoid circulars

    df = df.copy()

    # ---- Date parsing -------------------------------------------------------
    date_col_candidates = [c for c in df.columns if "date" in c or "year" in c]

    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
    elif "year" in df.columns and "month" in df.columns:
        day_col = "day" if "day" in df.columns else None
        if day_col:
            df["date"] = pd.to_datetime(
                df[["year", "month", "day"]].assign(day=df["day"].clip(1, 28)),
                errors="coerce"
            )
        else:
            df["date"] = pd.to_datetime(
                df["year"].astype(str) + "-" + df["month"].astype(str).str.zfill(2) + "-01",
                errors="coerce"
            )
    else:
        # Attempt to parse any column that looks like a date
        for col in date_col_candidates:
            try:
                df["date"] = pd.to_datetime(df[col], errors="coerce")
                break
            except Exception:  # pylint: disable=broad-except
                continue
        if "date" not in df.columns:
            df["date"] = pd.NaT
            logger.warning("Could not parse any date column; setting date=NaT.")

    # Extract year/month from parsed date
    if "year" not in df.columns:
        df["year"] = df["date"].dt.year
    if "month" not in df.columns:
        df["month"] = df["date"].dt.month
    if "day" not in df.columns:
        df["day"] = df["date"].dt.day

    # Coerce year, month to int
    df["year"] = pd.to_numeric(df["year"], errors="coerce").astype("Int64")
    df["month"] = pd.to_numeric(df["month"], errors="coerce").astype("Int64")

    # ---- Province normalization ----------------------------------------------
    if "province" in df.columns:
        df["province"] = df["province"].astype(str).str.strip()
        df["province"] = df["province"].replace(PROVINCE_CODE_MAP)
    
    # ---- Numeric coercion ---------------------------------------------------
    for col in ["fatalities", "injuries", "evacuations"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)

    # ---- Strip strings ------------------------------------------------------
    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].str.strip()

    return df


def handle_missing_values(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """
    Handle missing values with documented fill strategies.

    Fill strategies:
    - ``date``          → rows dropped (cannot engineer temporal features)
    - ``province``      → filled with 'Unknown'
    - ``incident_type`` → filled with 'Other Occurrence'
    - ``cargo_type``    → filled with 'General Freight'
    - ``fatalities``    → filled with 0 (no record = no fatality)
    - ``injuries``      → filled with 0
    - ``evacuations``   → filled with 0
    - Remaining numeric → filled with column median
    - Remaining object  → filled with 'Unknown'

    Parameters
    ----------
    df : pd.DataFrame
        Standardized DataFrame.

    Returns
    -------
    tuple[pd.DataFrame, dict]
        Cleaned DataFrame and a dictionary documenting fill decisions.
    """
    df = df.copy()
    decisions: dict[str, str] = {}

    initial_rows = len(df)
    df = df.dropna(subset=["date"])
    dropped = initial_rows - len(df)
    if dropped > 0:
        decisions["date"] = f"Dropped {dropped} rows with missing date (required for temporal features)"
        logger.info("Dropped %d rows with missing date.", dropped)

    fill_map_str = {
        "province": "Unknown",
        "incident_type": "Other Occurrence",
        "cargo_type": "General Freight",
        "railway_name": "Unknown",
        "subdivision": "Unknown",
        "track_class": "Unknown",
    }
    for col, fill_val in fill_map_str.items():
        if col in df.columns:
            n = df[col].isna().sum()
            if n > 0:
                df[col] = df[col].fillna(fill_val)
                decisions[col] = f"Filled {n} nulls with '{fill_val}'"
                logger.info("  %s: filled %d nulls → '%s'", col, n, fill_val)

    fill_map_zero = {"fatalities": 0, "injuries": 0, "evacuations": 0}
    for col, fill_val in fill_map_zero.items():
        if col in df.columns:
            n = df[col].isna().sum()
            if n > 0:
                df[col] = df[col].fillna(fill_val)
                decisions[col] = f"Filled {n} nulls with {fill_val} (assumed no occurrence)"

    for col in df.select_dtypes(include="number").columns:
        n = df[col].isna().sum()
        if n > 0:
            median_val = df[col].median()
            df[col] = df[col].fillna(median_val)
            decisions[col] = f"Filled {n} nulls with column median ({median_val:.2f})"

    for col in df.select_dtypes(include="object").columns:
        n = df[col].isna().sum()
        if n > 0:
            df[col] = df[col].fillna("Unknown")
            decisions[col] = f"Filled {n} nulls with 'Unknown'"

    return df, decisions


def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove exact duplicate rows from the DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.

    Returns
    -------
    pd.DataFrame
        De-duplicated DataFrame.
    """
    before = len(df)
    df = df.drop_duplicates().reset_index(drop=True)
    after = len(df)
    logger.info("Removed %d duplicate rows (%d → %d).", before - after, before, after)
    return df


# ---------------------------------------------------------------------------
# STEP 2 — FEATURE ENGINEERING
# ---------------------------------------------------------------------------

def engineer_features(
    df: pd.DataFrame,
    weather_df: Optional[pd.DataFrame] = None,
    density_df: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """
    Engineer 15+ predictive features for rail incident severity classification.

    Features:
        Temporal:
            - season (Winter/Spring/Summer/Fall)
            - decade (2000s/2010s/2020s)
            - is_weekend (binary)
            - years_since_2000 (continuous)

        Geographic/Route:
            - province_risk_score (frequency-based encoding)
            - route_density (track-km per capita, joined from density_df)

        Incident Properties:
            - incident_type_encoded (label encoded)
            - cargo_risk (ordinal 1–3)
            - multi_fatality (binary: fatalities >= 2)

        Target:
            - incident_severity (binary: HIGH if severe type + fatalities >= 1)

        Rolling/Cumulative:
            - rolling_12m_incidents (12-month rolling count per province)
            - cumulative_incidents_province (cumulative count per province over time)

        Weather (if weather_df provided):
            - avg_temp_c, total_precip_mm, avg_wind_kmh

        Interaction features:
            - season_x_province_risk
            - cargo_x_type_risk
            - temp_x_cargo_risk (if weather joined)
            - density_x_rolling (route density × rolling incidents)

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned, standardized incident DataFrame.
    weather_df : pd.DataFrame, optional
        Monthly weather by province from Open-Meteo.
    density_df : pd.DataFrame, optional
        Province route density from Statistics Canada.

    Returns
    -------
    pd.DataFrame
        Feature-enriched DataFrame.
    """
    df = df.copy()
    df = df.sort_values("date").reset_index(drop=True)

    # ---- 1. Season ----------------------------------------------------------
    df["season"] = df["month"].map(SEASONS).fillna("Unknown")

    # ---- 2. Decade ----------------------------------------------------------
    df["decade"] = (df["year"] // 10 * 10).astype(str) + "s"

    # ---- 3. Is weekend ------------------------------------------------------
    df["is_weekend"] = df["date"].dt.dayofweek.isin([5, 6]).astype(int)

    # ---- 4. Years since 2000 ------------------------------------------------
    df["years_since_2000"] = (df["year"] - REFERENCE_YEAR).clip(lower=0)

    # ---- 5. Province risk score (frequency-based) ---------------------------
    province_counts = df["province"].value_counts(normalize=True)
    df["province_risk_score"] = df["province"].map(province_counts).fillna(0).round(4)

    # ---- 6. Route density (join from Statistics Canada data) ----------------
    if density_df is not None and "province" in density_df.columns:
        df = df.merge(
            density_df[["province", "route_density_score"]],
            on="province", how="left"
        )
    else:
        # Fallback: uniform density proxy
        df["route_density_score"] = 0.5

    # ---- 7. Incident type encoded -------------------------------------------
    le_type = LabelEncoder()
    df["incident_type_encoded"] = le_type.fit_transform(
        df["incident_type"].fillna("Other Occurrence")
    )

    # ---- 8. Cargo risk ------------------------------------------------------
    df["cargo_risk"] = df["cargo_type"].map(CARGO_RISK_MAP).fillna(1).astype(int)

    # ---- 9. Multi-fatality --------------------------------------------------
    if "fatalities" in df.columns:
        df["multi_fatality"] = (df["fatalities"] >= 2).astype(int)
    else:
        df["multi_fatality"] = 0

    # ---- 10. Incident severity TARGET (binary) ------------------------------
    is_severe_type = df["incident_type"].isin(SEVERITY_INCIDENT_TYPES)
    has_fatality = df.get("fatalities", pd.Series(0, index=df.index)) >= 1
    has_injuries = df.get("injuries", pd.Series(0, index=df.index)) >= 2
    df["incident_severity"] = (is_severe_type | has_fatality | has_injuries).astype(int)

    # ---- 11. Rolling 12-month incidents per province ------------------------
    df_sorted = df.copy()
    rolling_counts: list[int] = []
    for idx, row in df_sorted.iterrows():
        prov = row["province"]
        cutoff = row["date"] - pd.DateOffset(months=12)
        count = ((df_sorted["province"] == prov) &
                 (df_sorted["date"] >= cutoff) &
                 (df_sorted["date"] < row["date"])).sum()
        rolling_counts.append(int(count))
    df["rolling_12m_incidents"] = rolling_counts

    # ---- 12. Cumulative incidents per province ------------------------------
    df["cumulative_incidents_province"] = (
        df.groupby("province").cumcount() + 1
    )

    # ---- 13. Weather features (if available) --------------------------------
    if weather_df is not None and not weather_df.empty:
        year_col = "year" if "year" in weather_df.columns else None
        month_col = "month" if "month" in weather_df.columns else None
        if year_col and month_col:
            df = df.merge(
                weather_df[["province", "year", "month",
                            "avg_temp_c", "total_precip_mm", "avg_wind_kmh"]],
                on=["province", "year", "month"],
                how="left",
            )
            df["avg_temp_c"] = df["avg_temp_c"].fillna(df["avg_temp_c"].median())
            df["total_precip_mm"] = df["total_precip_mm"].fillna(0)
            df["avg_wind_kmh"] = df["avg_wind_kmh"].fillna(df["avg_wind_kmh"].median())

    # ---- Interaction features -----------------------------------------------
    season_map = {"Winter": 4, "Spring": 2, "Summer": 1, "Fall": 3}
    df["season_num"] = df["season"].map(season_map).fillna(2)

    # 14. Season × province risk
    df["season_x_province_risk"] = df["season_num"] * df["province_risk_score"]

    # 15. Cargo × incident type risk
    df["cargo_x_type_risk"] = df["cargo_risk"] * df["incident_type_encoded"]

    # 16. Temperature × cargo risk (if weather joined)
    if "avg_temp_c" in df.columns:
        df["temp_x_cargo_risk"] = df["avg_temp_c"] * df["cargo_risk"]

    # 17. Route density × rolling incidents
    df["density_x_rolling"] = df["route_density_score"] * df["rolling_12m_incidents"]

    logger.info(
        "Feature engineering complete. Shape: %s | Severity distribution:\n%s",
        df.shape,
        df["incident_severity"].value_counts().to_string()
    )
    return df


def get_feature_columns(df: pd.DataFrame) -> list[str]:
    """
    Return the list of engineered feature columns suitable for model training.

    Parameters
    ----------
    df : pd.DataFrame
        Feature-enriched DataFrame.

    Returns
    -------
    list of str
        List of numeric feature column names.
    """
    target = "incident_severity"
    exclude = {
        "date", "_source_file", "province", "incident_type", "cargo_type",
        "railway_name", "subdivision", "track_class", "decade", "season",
        target,
    }
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    return [c for c in numeric_cols if c not in exclude]


def build_model_dataset(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """
    Extract feature matrix X and target vector y for model training.

    Parameters
    ----------
    df : pd.DataFrame
        Feature-enriched DataFrame with 'incident_severity' column.

    Returns
    -------
    tuple[pd.DataFrame, pd.Series]
        (X, y) where y = incident_severity binary target.
    """
    feature_cols = get_feature_columns(df)
    X = df[feature_cols].copy()
    y = df["incident_severity"].copy()
    # Final fillna safety net
    X = X.fillna(X.median(numeric_only=True))
    logger.info("Model dataset: X=%s, y distribution=%s",
                X.shape, y.value_counts().to_dict())
    return X, y


# ---------------------------------------------------------------------------
# FULL PIPELINE
# ---------------------------------------------------------------------------

def run_preprocessing_pipeline(
    raw_df: pd.DataFrame,
    weather_df: Optional[pd.DataFrame] = None,
    density_df: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """
    Execute the complete preprocessing pipeline end-to-end.

    Steps:
    1. Standardize columns and parse dates
    2. Handle missing values (with documentation)
    3. Remove duplicates
    4. Engineer all features

    Parameters
    ----------
    raw_df : pd.DataFrame
        Raw Transport Canada DataFrame.
    weather_df : pd.DataFrame, optional
        Monthly weather by province.
    density_df : pd.DataFrame, optional
        Province route density.

    Returns
    -------
    pd.DataFrame
        Fully processed, feature-enriched DataFrame.
    """
    logger.info("=== PREPROCESSING PIPELINE START ===")
    df = standardize_columns(raw_df)
    df, decisions = handle_missing_values(df)

    print("\n--- Missing Value Fill Decisions ---")
    for col, decision in decisions.items():
        print(f"  {col:30s} → {decision}")

    df = remove_duplicates(df)
    df = engineer_features(df, weather_df=weather_df, density_df=density_df)

    save_path = (
        __file__.replace("src/preprocessing.py", "data/processed/incidents_featured.parquet")
        .replace("src\\preprocessing.py", "data\\processed\\incidents_featured.parquet")
    )
    try:
        import os
        from pathlib import Path
        out = Path(__file__).parent.parent / "data" / "processed" / "incidents_featured.parquet"
        out.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(out, index=False)
        logger.info("Saved processed data → %s", out)
    except Exception as exc:  # pylint: disable=broad-except
        logger.warning("Could not save processed parquet: %s", exc)

    logger.info("=== PREPROCESSING PIPELINE COMPLETE === Shape: %s", df.shape)
    return df
