"""Guards against target leakage in the feature pipeline."""

import pandas as pd
import pytest

from src.preprocessing import (
    OUTCOME_COLUMNS,
    PRE_INCIDENT_FEATURES,
    TARGET_COLUMN,
    compute_past_province_history,
    get_feature_columns,
)


def _incidents(rows):
    return pd.DataFrame(rows, columns=["province", "date", TARGET_COLUMN]).assign(
        date=lambda d: pd.to_datetime(d["date"])
    )


def test_feature_list_contains_no_outcome_columns():
    assert OUTCOME_COLUMNS.isdisjoint(PRE_INCIDENT_FEATURES)


def test_get_feature_columns_rejects_missing_columns():
    with pytest.raises(ValueError, match="Missing feature columns"):
        get_feature_columns(pd.DataFrame({"year": [2000]}))


def test_history_uses_only_strictly_earlier_dates():
    # Arrange: two same-day incidents, then one later; plus another province
    df = _incidents([
        ("Ontario", "2010-01-01", 1),
        ("Ontario", "2010-01-01", 1),
        ("Ontario", "2010-06-01", 0),
        ("Quebec", "2010-03-01", 1),
    ])

    # Act
    hist = compute_past_province_history(df)

    # Assert: same-day rows see no history (not each other, not themselves)
    assert hist.loc[[0, 1], "prior_incidents_province"].tolist() == [0, 0]
    assert hist.loc[2, "prior_incidents_province"] == 2
    assert hist.loc[2, "rolling_12m_incidents"] == 2
    # other province's incidents never count
    assert hist.loc[3, "prior_incidents_province"] == 0


def test_history_ignores_the_current_rows_outcome():
    # Arrange: identical history, different outcome for the current row
    base = [("Ontario", "2010-01-01", 1), ("Ontario", "2010-02-01", 0)]
    severe_now = _incidents(base + [("Ontario", "2010-03-01", 1)])
    mild_now = _incidents(base + [("Ontario", "2010-03-01", 0)])

    # Act
    rate_if_severe = compute_past_province_history(severe_now).loc[2, "prior_severe_rate_province"]
    rate_if_mild = compute_past_province_history(mild_now).loc[2, "prior_severe_rate_province"]

    # Assert
    assert rate_if_severe == rate_if_mild


def test_rolling_window_drops_incidents_older_than_12_months():
    df = _incidents([("Ontario", "2010-01-01", 1), ("Ontario", "2011-06-01", 0)])

    hist = compute_past_province_history(df)

    assert hist.loc[1, "prior_incidents_province"] == 1
    assert hist.loc[1, "rolling_12m_incidents"] == 0
