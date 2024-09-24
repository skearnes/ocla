"""Tests for model_calibration.calibration."""

import numpy as np
import pandas as pd
import pytest

from model_calibration.calibration import get_best_confidence_levels, get_global_calibration_data
from model_calibration.confidence import ModelConfidence
from model_calibration.utilities import Transform


@pytest.mark.parametrize(
    "locations,labels,transform,expected",
    (
        (
            [6.1, 6.3, 6.5, 7.1],
            [6.0, 6.0, 6.0, 6.0],
            Transform.LOG,
            [ModelConfidence.A, ModelConfidence.B, ModelConfidence.C, ModelConfidence.D],
        ),
        (
            [12.0, 4.0, 90.0, 0.01],
            [10.0, 10.0, 10.0, 10.0],
            Transform.NONE,
            [ModelConfidence.A, ModelConfidence.B, ModelConfidence.C, ModelConfidence.D],
        ),
    ),
)
def test_get_best_confidence_levels(locations, labels, transform, expected):
    np.testing.assert_equal(get_best_confidence_levels(np.asarray(locations), np.asarray(labels), transform), expected)


def test_get_global_calibration_data():
    data = pd.DataFrame(
        {
            "mol_id": ["a", "a", "b", "b"],
            "assay_date": [
                pd.Timestamp("2023-01-01"),
                pd.Timestamp("2023-01-01"),
                pd.Timestamp("2023-06-01"),
                pd.Timestamp("2023-06-01"),
            ],
            "model_date": [
                pd.Timestamp("2022-01-01"),
                pd.Timestamp("2022-06-01"),
                pd.Timestamp("2022-06-01"),
                pd.Timestamp("2022-01-01"),
            ],
        }
    )
    expected = pd.DataFrame(
        {
            "mol_id": ["a", "b"],
            "assay_date": [pd.Timestamp("2023-01-01"), pd.Timestamp("2023-06-01")],
            "model_date": [pd.Timestamp("2022-06-01"), pd.Timestamp("2022-06-01")],
            "max_prediction_date": [pd.Timestamp("2023-01-01"), pd.Timestamp("2023-06-01")],
        },
        index=[1, 2],
    )
    pd.testing.assert_frame_equal(get_global_calibration_data(data), expected)
