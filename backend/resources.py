"""
Loads the files the API serves. Everything lives inside backend/ so the
Vercel service bundle is self-contained (see scripts/sync_backend.py).
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import joblib
import pandas as pd

logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
MODEL_PATH = BASE_DIR / "models" / "final_model.joblib"


@dataclass(frozen=True)
class Resources:
    incidents: Optional[pd.DataFrame]
    model: Any
    feature_names: Optional[list[str]]
    summary: dict[str, Any]
    cv_metrics: Optional[pd.DataFrame]
    holdout_metrics: Optional[pd.DataFrame]
    leakage_checks: Optional[pd.DataFrame]
    shap_importance: Optional[pd.DataFrame]
    roc_curves: dict[str, Any]
    calibration: list[dict[str, float]]


def _read_csv(name: str) -> Optional[pd.DataFrame]:
    path = DATA_DIR / name
    if not path.exists():
        logger.error("Missing data file: %s", path)
        return None
    return pd.read_csv(path)


def _read_json(name: str, default: Any) -> Any:
    path = DATA_DIR / name
    if not path.exists():
        logger.error("Missing data file: %s", path)
        return default
    return json.loads(path.read_text())


def _load_incidents() -> Optional[pd.DataFrame]:
    df = _read_csv("incidents.csv.gz")
    if df is None:
        return None
    return df.assign(date=pd.to_datetime(df["date"]))


def _load_model() -> tuple[Any, Optional[list[str]]]:
    if not MODEL_PATH.exists():
        logger.error("Model file not found: %s", MODEL_PATH)
        return None, None
    payload = joblib.load(MODEL_PATH)
    return payload["model"], list(payload["feature_names"])


def load_resources() -> Resources:
    model, feature_names = _load_model()
    return Resources(
        incidents=_load_incidents(),
        model=model,
        feature_names=feature_names,
        summary=_read_json("dataset_summary.json", {}),
        cv_metrics=_read_csv("model_metrics.csv"),
        holdout_metrics=_read_csv("model_metrics_time_holdout.csv"),
        leakage_checks=_read_csv("leakage_checks.csv"),
        shap_importance=_read_csv("shap_importance.csv"),
        roc_curves=_read_json("roc_curves.json", {}),
        calibration=_read_json("calibration.json", []),
    )
