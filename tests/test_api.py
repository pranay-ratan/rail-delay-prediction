"""Backend API: every endpoint serves real data, and serving matches training."""

import sys
from pathlib import Path

import pandas as pd
import pytest
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "backend"))

import server  # noqa: E402
from rail_features import history_as_of  # noqa: E402

client = TestClient(server.app)

GET_ENDPOINTS = [
    "/api/health", "/api/stats", "/api/provinces", "/api/incidents/annual",
    "/api/incidents/by-season", "/api/incidents/by-type", "/api/incidents/severity-by-type",
    "/api/incidents/heatmap", "/api/incidents/correlation", "/api/incidents/rolling",
    "/api/insights", "/api/models/metrics", "/api/models/roc", "/api/models/feature-importance",
]


def test_backend_feature_code_is_identical_to_training():
    assert (ROOT / "backend" / "rail_features.py").read_text() == (ROOT / "src" / "features.py").read_text()


def test_health_reports_model_and_data_loaded():
    body = client.get("/api/health").json()

    assert body["model_loaded"] is True
    assert body["data_loaded"] is True


@pytest.mark.parametrize("path", GET_ENDPOINTS)
def test_get_endpoints_respond(path):
    assert client.get(path).status_code == 200


def test_stats_come_from_the_data_and_training_outputs():
    body = client.get("/api/stats").json()
    cv = pd.read_csv(ROOT / "backend" / "data" / "model_metrics.csv").set_index("model")

    assert body["total_incidents"] == len(server.R.incidents)
    assert body["best_auc"] == round(cv.loc[body["best_model"], "roc_auc"] * 100, 1)


def test_feature_importance_is_shap_not_hardcoded():
    body = client.get("/api/models/feature-importance").json()
    shap = pd.read_csv(ROOT / "backend" / "data" / "shap_importance.csv")

    assert "SHAP" in body["method"]
    assert body["features"] == shap["feature"].head(15).tolist()


def test_predict_uses_the_model_and_explains_itself():
    res = client.post("/api/predict", json={
        "province": "Ontario", "year": 2024, "month": 1, "hour": 8, "track_type": "main",
        "activity": "proceeding", "train_type": "passenger", "approx_speed_mph": 70,
    })
    body = res.json()

    assert res.status_code == 200
    assert body["source"] == "model"
    assert 0 <= body["probability"] <= 1
    assert body["contributions"]
    assert set(body["inputs"]["weather_filled_from_climatology"]) == {"wx_temp_c", "wx_precip_mm", "wx_wind_kmh"}


def test_main_line_passenger_scores_above_yard_switching():
    common = {"province": "Ontario", "year": 2024, "month": 6}
    risky = client.post("/api/predict", json={**common, "track_type": "main", "activity": "proceeding",
                                              "train_type": "passenger", "approx_speed_mph": 70}).json()
    calm = client.post("/api/predict", json={**common, "track_type": "yard", "activity": "switching",
                                             "train_type": "cars_only", "approx_speed_mph": 5}).json()

    assert risky["model_score"] > calm["model_score"]


@pytest.mark.parametrize("payload", [
    {"province": "Atlantis"},
    {"track_type": "elevated"},
    {"approx_speed_mph": 500},
    {"month": 13},
])
def test_predict_rejects_invalid_input(payload):
    assert client.post("/api/predict", json=payload).status_code == 422


def test_api_history_matches_training_history():
    # The API recomputes history per request; it must equal what training saw.
    df = server.R.incidents
    sample = df.sample(40, random_state=0)
    for row in sample.itertuples():
        prov = df[df["province"] == row.province]
        hist = history_as_of(prov["date"], prov["incident_severity"], row.date)
        assert hist["prior_incidents_province"] == row.prior_incidents_province
        assert hist["rolling_12m_incidents"] == row.rolling_12m_incidents
        assert hist["prior_severe_rate_province"] == pytest.approx(row.prior_severe_rate_province)
