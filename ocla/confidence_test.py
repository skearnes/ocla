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

"""Tests for ocla.confidence."""

import numpy as np
import pytest

from ocla.confidence import ModelConfidence, assign_confidence_level, get_bounds
from ocla.utilities import Transform


@pytest.mark.parametrize(
    "labels,threshold,transform,expected_min_values,expected_max_values",
    (
        ([6.0, 7.0], 10.0, Transform.LOG, [5.0, 6.0], [7.0, 8.0]),
        ([10.0, 100.0], 10.0, Transform.NONE, [1.0, 10.0], [100.0, 1000.0]),
    ),
)
def test_get_bounds(labels, threshold, transform, expected_min_values, expected_max_values):
    min_values, max_values = get_bounds(np.asarray(labels), threshold=threshold, transform=transform)
    np.testing.assert_allclose(min_values, expected_min_values)
    np.testing.assert_allclose(max_values, expected_max_values)


@pytest.mark.parametrize(
    "row,expected",
    (
        ([0.9, 0.9, 0.9], ModelConfidence.A),
        ([0.9, 0.1, 0.9], ModelConfidence.A),
        ([0.9, 0.9, 0.1], ModelConfidence.A),
        ([0.9, 0.1, 0.1], ModelConfidence.A),
        ([0.9, np.nan, np.nan], ModelConfidence.A),
        ([0.1, 0.9, 0.9], ModelConfidence.B),
        ([0.1, 0.9, 0.1], ModelConfidence.B),
        ([np.nan, 0.9, np.nan], ModelConfidence.B),
        ([0.1, 0.1, 0.9], ModelConfidence.C),
        ([np.nan, np.nan, 0.9], ModelConfidence.C),
        ([0.1, 0.1, 0.1], ModelConfidence.D),
        ([np.nan, np.nan, 0.1], ModelConfidence.D),
        ([np.nan, 0.1, np.nan], ModelConfidence.D),
        ([0.1, np.nan, np.nan], ModelConfidence.D),
    ),
)
def test_assign_confidence_level(row, expected):
    assert assign_confidence_level(np.asarray(row)) == expected
