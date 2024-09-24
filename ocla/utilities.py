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

"""Utility functions."""

import datetime
from enum import Enum, auto
from typing import Sequence

import numpy as np


class Transform(Enum):
    """Data transforms."""

    NONE = auto()
    LOG = auto()

    def apply(self, values: np.ndarray) -> np.ndarray:
        """Applies the transform to an array."""
        match self:
            case Transform.NONE:
                return np.asarray(values)
            case Transform.LOG:
                return np.log10(values)
            case _:
                raise NotImplementedError(self)

    def reverse(self, values: np.ndarray) -> np.ndarray:
        """Inverts the transform for an array."""
        match self:
            case Transform.NONE:
                return np.asarray(values)
            case Transform.LOG:
                return np.power(10, values)
            case _:
                raise NotImplementedError(self)


def get_training_dates(assay_dates: Sequence[datetime.date]) -> list[datetime.date]:
    """Returns a list of dates for model training with at least one week separation.

    Args:
        assay_dates: List of assay dates.

    Returns:
        List of dates for training new model versions.
    """
    model_dates = []
    for year, week in {date.isocalendar()[:2] for date in assay_dates}:
        model_dates.append(datetime.date.fromisocalendar(year, week, 7))
    return sorted(model_dates)
