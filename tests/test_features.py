"""Feature encoding rules shared by training and the API."""

import numpy as np
import pandas as pd

from src.data_loader import _ddmmss_to_degrees
from src.features import (
    ACTIVITY_FLAGS, ACTIVITY_TYPES, NUMERIC_LIMITS, PRE_INCIDENT_FEATURES,
    clean_numeric, flag_columns, province_one_hot,
)
from src.preprocessing import to_canonical


def test_missing_category_is_indistinguishable_from_other():
    # A blank activity must encode exactly like a real-but-uncommon one,
    # otherwise "field left blank" (common for minor accidents) leaks.
    values = pd.Series([None, "INSPECTION", "PROCEEDING"])

    flags = flag_columns(values, ACTIVITY_TYPES, ACTIVITY_FLAGS, "activity")

    assert flags.iloc[0].tolist() == flags.iloc[1].tolist() == [0, 0, 0]
    assert flags.iloc[2]["activity_proceeding"] == 1


def test_placeholder_numbers_become_missing():
    df = pd.DataFrame({c: [10, 99999, -1] for c in NUMERIC_LIMITS})

    cleaned = clean_numeric(df)

    assert cleaned["num_cars"].tolist()[0] == 10
    assert cleaned[list(NUMERIC_LIMITS)].iloc[1:].isna().all().all()


def test_province_one_hot_has_fixed_columns():
    encoded = province_one_hot(pd.Series(["Quebec", "Atlantis"]))

    assert encoded.sum(axis=1).tolist() == [1, 0]
    assert all(c in PRE_INCIDENT_FEATURES for c in encoded.columns)


def test_ddmmss_coordinates_convert_to_degrees():
    degrees = _ddmmss_to_degrees(pd.Series([494700, 1112230]))

    assert np.allclose(degrees, [49 + 47 / 60, 111 + 22 / 60 + 30 / 3600])


def test_target_is_fatality_or_serious_injury_only():
    raw = pd.DataFrame({
        "OccNo": ["a", "b", "c", "d"],
        "date": pd.to_datetime(["2010-01-01"] * 4),
        "OccTime": [1200] * 4,
        "province": ["Ontario"] * 4,
        "SubdMileage": [1.0] * 4,
        "TrackTypeID_DisplayEng": ["MAIN"] * 4,
        "ActivityTypeID_DisplayEng": ["PROCEEDING"] * 4,
        "TrainTypeID_DisplayEng": ["FREIGHT"] * 4,
        "ApproxTrainSpeed_MPH": [30] * 4,
        "NumCars": [50] * 4,
        "NumLocomotives": [2] * 4,
        "TrainTonnage": [5000] * 4,
        "AccIncTypeID_DisplayEng": ["CROSSING"] * 4,
        "TotalFatalInjuries": [1, 0, 0, None],
        "TotalSeriousInjuries": [0, 2, 0, 0],
        "TotalMinorInjuries": [0, 0, 5, 0],
        "_source_file": ["tsb"] * 4,
    })

    canonical = to_canonical(raw)

    assert canonical.set_index("occ_no")["incident_severity"].to_dict() == {
        "a": 1, "b": 1, "c": 0, "d": 0,
    }
