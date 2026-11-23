"""
train.py
========
End-to-end, leakage-free training on real TSB data. From the repo root:

    python -m src.train

Needs the TSB CSVs in data/raw/tsb/ (see README). Writes:
    data/processed/incidents_featured.parquet
    outputs/results/model_metrics.csv               10-fold stratified CV
    outputs/results/model_metrics_time_holdout.csv  train 2001-2020, test 2021-2025
    outputs/results/leakage_checks.csv
    outputs/results/shap_importance.csv
    outputs/results/dataset_summary.json
    outputs/figures/shap_summary.png, shap_bar.png
    models/final_model.joblib                       tuned XGBoost pipeline
"""

from __future__ import annotations

import json
import logging
import platform
from pathlib import Path
from typing import Any

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import shap  # noqa: E402
import sklearn  # noqa: E402
import xgboost as xgb  # noqa: E402
from sklearn.base import clone  # noqa: E402
from sklearn.metrics import roc_auc_score, roc_curve  # noqa: E402
from sklearn.model_selection import StratifiedKFold, cross_val_predict  # noqa: E402

from src.data_loader import fetch_daily_weather, load_tsb_accidents, province_weather_points  # noqa: E402
from src.features import NUMERIC_TRAIN_FEATURES, TARGET_COLUMN  # noqa: E402
from src.models import (  # noqa: E402
    BASELINE_NAME, N_SPLITS, RANDOM_STATE, RESULTS_DIR, build_models, compute_shap_values,
    evaluate_model_cv, evaluate_time_holdout, positive_weight, save_final_model,
    train_all_models, tune_xgboost,
)
from src.preprocessing import build_model_dataset, run_preprocessing_pipeline  # noqa: E402

logger = logging.getLogger(__name__)

START_YEAR: int = 2001
END_YEAR: int = 2025
FIRST_TEST_YEAR: int = 2021
TUNED_NAME: str = "XGBoost (Tuned)"
FIGURES_DIR: Path = RESULTS_DIR.parent / "figures"
METRIC_COLUMNS: list[str] = [
    "model", "accuracy", "balanced_accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc",
]
SHAP_SAMPLES: int = 3000
SHAP_TOP_N: int = 15
ROC_POINTS: int = 120
CALIBRATION_BINS: int = 20


def leakage_checks(X: pd.DataFrame, y: pd.Series, tuned: Any) -> pd.DataFrame:
    """
    Quantify how much "which fields are blank" alone predicts the target, and
    how the tuned model does on records with no missing train numerics.
    """
    missing_only = X.isna().astype(int).loc[:, lambda d: d.any()]
    weight = positive_weight(y)
    complete = X[list(NUMERIC_TRAIN_FEATURES)].notna().all(axis=1)
    rows = [
        {**evaluate_model_cv(build_models(weight)["XGBoost"], missing_only, y,
                             "Missingness indicators only (XGBoost)"),
         "n_rows": len(X), "note": "how much blank fields alone predict severity"},
        {**evaluate_model_cv(clone(tuned), X[complete], y[complete],
                             "XGBoost (Tuned), complete records only"),
         "n_rows": int(complete.sum()), "note": "rows with every train numeric present"},
    ]
    return pd.DataFrame(rows)[METRIC_COLUMNS + ["n_rows", "note"]]


def out_of_fold_scores(models: dict[str, Any], X: pd.DataFrame, y: pd.Series) -> dict[str, np.ndarray]:
    """10-fold out-of-fold P(severe) for each model (same folds as the CV table)."""
    cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    return {
        name: cross_val_predict(clone(m), X, y, cv=cv, method="predict_proba", n_jobs=-1)[:, 1]
        for name, m in models.items()
    }


def roc_curves(scores: dict[str, np.ndarray], y: pd.Series) -> dict[str, dict[str, list[float]]]:
    """Real ROC curves from out-of-fold scores, thinned to ROC_POINTS points."""
    curves = {}
    for name, s in scores.items():
        fpr, tpr, _ = roc_curve(y, s)
        keep = np.unique(np.linspace(0, len(fpr) - 1, ROC_POINTS).astype(int))
        curves[name] = {"fpr": fpr[keep].round(4).tolist(), "tpr": tpr[keep].round(4).tolist(),
                        "auc": round(float(roc_auc_score(y, s)), 4)}
    return curves


def calibration_table(scores: np.ndarray, y: pd.Series) -> list[dict[str, float]]:
    """
    Map the model's (class-weighted, so uncalibrated) score to the severe rate
    actually observed among out-of-fold accidents in the same score band.
    """
    edges = np.unique(np.quantile(scores, np.linspace(0, 1, CALIBRATION_BINS + 1)))
    bands = np.clip(np.searchsorted(edges, scores, side="right") - 1, 0, len(edges) - 2)
    rows = []
    for b in range(len(edges) - 1):
        in_band = bands == b
        rows.append({"score_min": float(edges[b]), "score_max": float(edges[b + 1]),
                     "n": int(in_band.sum()), "observed_rate": float(y[in_band].mean())})
    return rows


def save_shap(model: Any, X: pd.DataFrame) -> pd.DataFrame:
    """SHAP summary plots + mean |SHAP| table for the final model."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    values, sample = compute_shap_values(model, X, max_samples=SHAP_SAMPLES)
    for kind, name in (("dot", "shap_summary.png"), ("bar", "shap_bar.png")):
        plt.figure()
        shap.summary_plot(values, sample, plot_type=kind, show=False, max_display=SHAP_TOP_N)
        plt.title("SHAP: what pushes an accident towards death / serious injury")
        plt.tight_layout()
        plt.savefig(FIGURES_DIR / name, dpi=150, bbox_inches="tight")
        plt.close("all")
    importance = (
        pd.DataFrame({"feature": sample.columns, "mean_abs_shap": np.abs(values).mean(axis=0)})
        .sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)
    )
    importance.to_csv(RESULTS_DIR / "shap_importance.csv", index=False)
    return importance


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    accidents = load_tsb_accidents(START_YEAR, END_YEAR)
    weather = fetch_daily_weather(province_weather_points(accidents), START_YEAR, END_YEAR)
    processed = run_preprocessing_pipeline(accidents, weather)
    X, y = build_model_dataset(processed)
    years = processed["year"].astype(int)

    cv_metrics, fitted = train_all_models(X, y)

    # Tune on the training years only, so the time-holdout score is untouched.
    train_years = (years < FIRST_TEST_YEAR).to_numpy()
    tuned, best_params, inner_auc = tune_xgboost(X[train_years], y[train_years])
    cv_metrics[TUNED_NAME] = evaluate_model_cv(tuned, X, y, TUNED_NAME)

    holdout = [
        evaluate_time_holdout(m, X, y, years, FIRST_TEST_YEAR, name)
        for name, m in {**fitted, TUNED_NAME: tuned}.items()
    ]
    final_model = clone(tuned).fit(X, y)
    checks = leakage_checks(X, y, tuned)

    oof = out_of_fold_scores({**build_models(positive_weight(y)), TUNED_NAME: tuned}, X, y)
    (RESULTS_DIR / "roc_curves.json").write_text(json.dumps(roc_curves(oof, y)))
    (RESULTS_DIR / "calibration.json").write_text(
        json.dumps(calibration_table(oof[TUNED_NAME], y), indent=1))

    cv_df = pd.DataFrame(cv_metrics.values())[METRIC_COLUMNS + ["roc_auc_std"]]
    holdout_df = pd.DataFrame(holdout)[METRIC_COLUMNS + ["n_train", "n_test"]]
    cv_df.to_csv(RESULTS_DIR / "model_metrics.csv", index=False)
    holdout_df.to_csv(RESULTS_DIR / "model_metrics_time_holdout.csv", index=False)
    checks.to_csv(RESULTS_DIR / "leakage_checks.csv", index=False)
    importance = save_shap(final_model, X)

    summary = {
        "data_source": "TSB Rail Occurrence Database (RODS), accidents only",
        "weather_source": sorted(weather["_weather_source"].unique().tolist()),
        "n_rows": int(len(X)),
        "year_range": [START_YEAR, END_YEAR],
        "n_features": int(X.shape[1]),
        "positive_rate": float(y.mean()),
        "target": "fatality or serious injury (>=1)",
        "cv": f"StratifiedKFold(n_splits={N_SPLITS}, shuffle=True, random_state={RANDOM_STATE})",
        "time_holdout": f"train {START_YEAR}-{FIRST_TEST_YEAR - 1}, test {FIRST_TEST_YEAR}-{END_YEAR}",
        "tuning": {"best_params": best_params, "inner_cv_auc_train_years": inner_auc},
        "final_model": TUNED_NAME,
        "baseline": BASELINE_NAME,
        "versions": {
            "python": platform.python_version(), "scikit-learn": sklearn.__version__,
            "xgboost": xgb.__version__, "pandas": pd.__version__,
            "numpy": np.__version__, "joblib": joblib.__version__,
        },
    }
    (RESULTS_DIR / "dataset_summary.json").write_text(json.dumps(summary, indent=2, default=str))
    save_final_model(final_model, list(X.columns), summary)

    pd.set_option("display.width", 220)
    logger.info("Rows %d | features %d | severe share %.4f", len(X), X.shape[1], y.mean())
    print(f"\n=== {N_SPLITS}-fold stratified CV ===\n{cv_df.round(4).to_string(index=False)}")
    print(f"\n=== Time holdout ===\n{holdout_df.round(4).to_string(index=False)}")
    print(f"\n=== Leakage checks ===\n{checks.round(4).to_string(index=False)}")
    print(f"\n=== Top SHAP ===\n{importance.head(SHAP_TOP_N).round(4).to_string(index=False)}")
    print(f"\nTarget: {TARGET_COLUMN}; tuned params: {best_params}")


if __name__ == "__main__":
    # Threads, not processes: works where multiprocessing semaphores are
    # blocked, and the tree libraries release the GIL anyway.
    with joblib.parallel_backend("threading"):
        main()
