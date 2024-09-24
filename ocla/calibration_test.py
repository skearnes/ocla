# Copyright 2024 Relay Therapeutics, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for ocla.calibration."""

import numpy as np
import pandas as pd
import pytest

from ocla.calibration import get_best_confidence_levels, get_global_calibration_data
from ocla.confidence import ModelConfidence
from ocla.utilities import Transform


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
