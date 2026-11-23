"""Every Streamlit page renders without exceptions, and the assessor predicts."""

import sys
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

WEBSITE = Path(__file__).resolve().parent.parent / "website"
sys.path.insert(0, str(WEBSITE))

PAGES = ["app.py", "pages/1_Data_Explorer.py", "pages/2_Model_Results.py", "pages/3_Live_Risk_Assessor.py"]
TIMEOUT_S = 120


@pytest.mark.parametrize("page", PAGES)
def test_page_renders_without_exceptions(page):
    at = AppTest.from_file(str(WEBSITE / page), default_timeout=TIMEOUT_S).run()

    assert not at.exception, [e.message for e in at.exception]


def test_overview_shows_real_metrics():
    at = AppTest.from_file(str(WEBSITE / "app.py"), default_timeout=TIMEOUT_S).run()
    metrics = {m.label: m.value for m in at.metric}

    from common import resources  # noqa: PLC0415  (needs website/ on sys.path)

    assert metrics["Accidents"] == f"{len(resources().incidents):,}"
    assert any(label.startswith("ROC-AUC") for label in metrics)


def test_live_assessor_returns_a_prediction():
    at = AppTest.from_file(str(WEBSITE / "pages/3_Live_Risk_Assessor.py"), default_timeout=TIMEOUT_S).run()
    train_type = next(r for r in at.radio if "Passenger" in r.options)

    train_type.set_value("Passenger")
    at.button[0].click().run()

    assert not at.exception, [e.message for e in at.exception]
    labels = [m.label for m in at.metric]
    assert "Chance the accident is severe" in labels
