"""
Copy everything the API needs into backend/, which is the only folder the
Vercel backend service bundles. Run after `python -m src.train`:

    python scripts/sync_backend.py
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
BACKEND = ROOT / "backend"
RESULTS = ROOT / "outputs" / "results"

RESULT_FILES = (
    "model_metrics.csv", "model_metrics_time_holdout.csv", "leakage_checks.csv",
    "shap_importance.csv", "roc_curves.json", "calibration.json", "dataset_summary.json",
)
# Columns the API serves or needs (charts, history features, weather defaults).
DATA_COLUMNS = (
    "occ_no", "date", "year", "month", "hour", "season", "province", "accident_type",
    "train_type", "track_type", "activity", "fatalities", "serious_injuries",
    "incident_severity", "mile_post", "approx_speed_mph", "num_cars", "num_locomotives",
    "train_tonnage", "wx_temp_c", "wx_precip_mm", "wx_wind_kmh", "rolling_12m_incidents",
    "prior_incidents_province", "prior_severe_rate_province", "track_main",
    "activity_proceeding", "train_passenger",
)


def main() -> None:
    (BACKEND / "data").mkdir(parents=True, exist_ok=True)
    (BACKEND / "models").mkdir(parents=True, exist_ok=True)

    shutil.copyfile(ROOT / "src" / "features.py", BACKEND / "rail_features.py")
    shutil.copyfile(ROOT / "models" / "final_model.joblib", BACKEND / "models" / "final_model.joblib")
    for name in RESULT_FILES:
        shutil.copyfile(RESULTS / name, BACKEND / "data" / name)

    # gzip CSV, not parquet: reading parquet needs pyarrow (~165 MB unpacked),
    # which would push the function bundle close to Vercel's 500 MB limit.
    processed = pd.read_parquet(ROOT / "data" / "processed" / "incidents_featured.parquet")
    processed[list(DATA_COLUMNS)].to_csv(BACKEND / "data" / "incidents.csv.gz", index=False)
    print(f"Synced {len(processed):,} accidents, model and {len(RESULT_FILES)} result files -> {BACKEND}")


if __name__ == "__main__":
    main()
