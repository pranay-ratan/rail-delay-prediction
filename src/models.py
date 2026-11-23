"""
models.py
=========
Model definitions, cross-validation, time-holdout evaluation, XGBoost tuning
and SHAP for the rail accident severity classifier.

Every model is a Pipeline that imputes missing values (median) inside each
training fold, so no statistic is computed from held-out rows.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Optional

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.base import clone
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, average_precision_score, balanced_accuracy_score,
    f1_score, precision_score, recall_score, roc_auc_score,
)
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

ROOT: Path = Path(__file__).resolve().parent.parent
MODELS_DIR: Path = ROOT / "models"
RESULTS_DIR: Path = ROOT / "outputs" / "results"
RANDOM_STATE: int = 42
N_SPLITS: int = 10
N_ITER_SEARCH: int = 30
N_TREES: int = 300

BASELINE_NAME: str = "Majority-class baseline"

# Precision / recall / F1 are for the positive ("severe") class. With ~11%
# positives, accuracy alone is misleading, so balanced accuracy and PR-AUC
# are reported too.
SCORING: dict[str, str] = {
    "accuracy": "accuracy",
    "balanced_accuracy": "balanced_accuracy",
    "precision": "precision",
    "recall": "recall",
    "f1": "f1",
    "roc_auc": "roc_auc",
    "pr_auc": "average_precision",
}

XGB_PARAM_GRID: dict[str, list[Any]] = {
    "clf__n_estimators": [200, 300, 500],
    "clf__max_depth": [3, 4, 6, 8],
    "clf__learning_rate": [0.03, 0.05, 0.1],
    "clf__subsample": [0.7, 0.85, 1.0],
    "clf__colsample_bytree": [0.6, 0.8, 1.0],
    "clf__min_child_weight": [1, 5, 10],
    "clf__reg_lambda": [1.0, 2.0, 5.0],
}


def _imputed(clf: Any, scale: bool = False) -> Pipeline:
    steps = [("impute", SimpleImputer(strategy="median"))]
    if scale:
        steps.append(("scale", StandardScaler()))
    return Pipeline([*steps, ("clf", clf)])


def build_models(pos_weight: float) -> dict[str, Any]:
    """The majority baseline plus the four compared classifiers (untrained)."""
    return {
        BASELINE_NAME: DummyClassifier(strategy="most_frequent"),
        "Logistic Regression": _imputed(
            LogisticRegression(max_iter=2000, class_weight="balanced", random_state=RANDOM_STATE),
            scale=True,
        ),
        "Random Forest": _imputed(RandomForestClassifier(
            n_estimators=N_TREES, min_samples_leaf=5, class_weight="balanced",
            n_jobs=-1, random_state=RANDOM_STATE,
        )),
        "XGBoost": _imputed(xgb.XGBClassifier(
            n_estimators=N_TREES, max_depth=6, learning_rate=0.05, subsample=0.8,
            colsample_bytree=0.8, scale_pos_weight=pos_weight, eval_metric="logloss",
            random_state=RANDOM_STATE, n_jobs=-1,
        )),
        "LightGBM": _imputed(lgb.LGBMClassifier(
            n_estimators=N_TREES, max_depth=6, learning_rate=0.05, subsample=0.8,
            subsample_freq=1, colsample_bytree=0.8, class_weight="balanced",
            random_state=RANDOM_STATE, verbose=-1,
        )),
    }


def positive_weight(y: pd.Series) -> float:
    """negatives / positives, used to balance XGBoost like the other models."""
    return float((y == 0).sum() / max((y == 1).sum(), 1))


def evaluate_model_cv(model: Any, X: pd.DataFrame, y: pd.Series, model_name: str,
                      n_splits: int = N_SPLITS) -> dict[str, Any]:
    """Mean of each metric over stratified k-fold cross-validation."""
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    logger.info("Running %d-fold CV for %s ...", n_splits, model_name)
    scores = cross_validate(model, X, y, cv=cv, scoring=SCORING, n_jobs=-1)
    metrics: dict[str, Any] = {"model": model_name}
    metrics.update({name: float(np.mean(scores[f"test_{name}"])) for name in SCORING})
    metrics.update({"roc_auc_std": float(np.std(scores["test_roc_auc"]))})
    logger.info("  %s | AUC=%.4f | PR-AUC=%.4f | F1=%.4f",
                model_name, metrics["roc_auc"], metrics["pr_auc"], metrics["f1"])
    return metrics


def train_all_models(X: pd.DataFrame, y: pd.Series) -> tuple[dict[str, dict], dict[str, Any]]:
    """CV-evaluate every model, then fit each on all rows."""
    all_metrics: dict[str, dict] = {}
    fitted: dict[str, Any] = {}
    for name, model in build_models(positive_weight(y)).items():
        all_metrics[name] = evaluate_model_cv(model, X, y, name)
        fitted[name] = model.fit(X, y)
    return all_metrics, fitted


def evaluate_time_holdout(model: Any, X: pd.DataFrame, y: pd.Series, years: pd.Series,
                          first_test_year: int, model_name: str) -> dict[str, Any]:
    """Fit on years < first_test_year, score on the later years (no shuffling)."""
    train = (years < first_test_year).to_numpy()
    fitted = clone(model).fit(X[train], y[train])
    proba = fitted.predict_proba(X[~train])[:, 1]
    preds = fitted.predict(X[~train])
    y_test = y[~train]
    return {
        "model": model_name,
        "accuracy": accuracy_score(y_test, preds),
        "balanced_accuracy": balanced_accuracy_score(y_test, preds),
        "precision": precision_score(y_test, preds, zero_division=0),
        "recall": recall_score(y_test, preds, zero_division=0),
        "f1": f1_score(y_test, preds, zero_division=0),
        "roc_auc": roc_auc_score(y_test, proba),
        "pr_auc": average_precision_score(y_test, proba),
        "n_train": int(train.sum()),
        "n_test": int((~train).sum()),
    }


def tune_xgboost(X: pd.DataFrame, y: pd.Series) -> tuple[Pipeline, dict[str, Any], float]:
    """RandomizedSearchCV (10-fold stratified, ROC-AUC) over XGB_PARAM_GRID."""
    base = build_models(positive_weight(y))["XGBoost"]
    search = RandomizedSearchCV(
        base, XGB_PARAM_GRID, n_iter=N_ITER_SEARCH, scoring="roc_auc", n_jobs=-1,
        cv=StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE),
        random_state=RANDOM_STATE,
    )
    logger.info("Tuning XGBoost: %d candidates x %d folds ...", N_ITER_SEARCH, N_SPLITS)
    search.fit(X, y)
    logger.info("Best params %s | inner CV AUC %.4f", search.best_params_, search.best_score_)
    return search.best_estimator_, search.best_params_, float(search.best_score_)


def compute_shap_values(pipeline: Pipeline, X: pd.DataFrame,
                        max_samples: int = 2000) -> tuple[np.ndarray, pd.DataFrame]:
    """TreeExplainer SHAP values (log-odds) for a fitted impute->tree pipeline."""
    import shap

    sample = X.sample(min(max_samples, len(X)), random_state=RANDOM_STATE)
    imputed = pd.DataFrame(pipeline[:-1].transform(sample), columns=X.columns, index=sample.index)
    values = shap.TreeExplainer(pipeline[-1]).shap_values(imputed)
    if isinstance(values, list):
        values = values[1]
    return np.asarray(values), imputed


def save_final_model(model: Any, feature_names: list[str],
                     metadata: Optional[dict[str, Any]] = None,
                     path: Optional[Path] = None) -> Path:
    """Persist {'model', 'feature_names', 'metadata'} with joblib."""
    out = path or MODELS_DIR / "final_model.joblib"
    out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "feature_names": feature_names, "metadata": metadata or {}}, out)
    logger.info("Saved final model -> %s", out)
    return out


def load_final_model(path: Optional[Path] = None) -> tuple[Any, list[str]]:
    """Load (model, feature_names); raises FileNotFoundError if not trained yet."""
    src = path or MODELS_DIR / "final_model.joblib"
    if not src.exists():
        raise FileNotFoundError(f"Model not found at {src}. Run `python -m src.train` first.")
    payload = joblib.load(src)
    return payload["model"], payload["feature_names"]
