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

"""Model calibration."""

import datetime
import logging
from enum import Enum, auto

import numpy as np
import pandas as pd
from scipy.stats import norm

from ocla.confidence import CONFIDENCE_THRESHOLDS, ModelConfidence, assign_confidence_level, get_bounds
from ocla.utilities import Transform

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_best_confidence_levels(locations: np.ndarray, labels: np.ndarray, transform: Transform) -> np.ndarray:
    """Finds the best possible confidence levels that could have been assigned (after we know the experimental values).

    Args:
        locations: Float array with shape [num_values] containing predicted values.
        labels: Float array with shape [num_values] containing experimental values.
        transform: Transform used in the model.

    Returns:
        Array with shape [num_values] containing the best possible confidence level for each prediction.
    """
    confidence_levels = [ModelConfidence.A, ModelConfidence.B, ModelConfidence.C]
    assert len(confidence_levels) == len(CONFIDENCE_THRESHOLDS)
    best_confidence_levels = np.asarray([ModelConfidence.D] * len(locations), dtype=object)
    for confidence_level, threshold in zip(confidence_levels, CONFIDENCE_THRESHOLDS):
        min_values, max_values = get_bounds(labels, threshold, transform)
        mask = np.logical_and(
            best_confidence_levels == ModelConfidence.D,
            np.logical_and(locations >= min_values, locations <= max_values),
        )
        best_confidence_levels[mask] = confidence_level
    return best_confidence_levels


class Calibration(Enum):
    """Calibration categories for individual predictions."""

    # The assigned confidence level matches the "best case" confidence level.
    CORRECT = auto()
    # The assigned confidence level is more conservative than the "best case" confidence level. For instance, if the
    # assigned confidence level is "B" and the "best case" confidence level is "A".
    UNDERCONFIDENT = auto()
    # The assigned confidence level is less conservative than the "best case" confidence level. For instance, if the
    # assigned confidence level is "A" and the "best case" confidence level is "B".
    OVERCONFIDENT = auto()


def assign_calibration_levels(data: pd.DataFrame, transform: Transform) -> pd.DataFrame:
    """Assigns calibration levels to predictions."""
    best_confidence_levels = get_best_confidence_levels(data["location"], data["label"], transform)
    data["transform"] = transform.name
    data["confidence_level_best_case"] = [value.name for value in best_confidence_levels]
    key = "confidence_level_calibration"
    data.loc[data["confidence_level_best_case"] == data["confidence_level"], key] = Calibration.CORRECT.name
    data.loc[data["confidence_level_best_case"] < data["confidence_level"], key] = Calibration.UNDERCONFIDENT.name
    data.loc[data["confidence_level_best_case"] > data["confidence_level"], key] = Calibration.OVERCONFIDENT.name
    return data


def get_baseline_calibration_data(calibration_data: pd.DataFrame, transform: Transform) -> pd.DataFrame:
    """Calculates calibration data for a simple baseline model."""
    columns = [
        "mol_id",
        "label",
        "assay_date",
        "operator",
        "transform",
        "model_id",
        "location",
        "scale",
        "model_date",
        "model_name",
        "model_type",
    ]
    baseline_data = calibration_data.copy()
    for column in baseline_data.columns:
        if column not in columns:
            del baseline_data[column]
    confidence = np.zeros((len(baseline_data), 3))
    for i, threshold in enumerate(CONFIDENCE_THRESHOLDS):
        min_values, max_values = get_bounds(baseline_data["location"], threshold, transform)
        lower = norm.cdf(min_values, loc=baseline_data["location"], scale=baseline_data["scale"])
        upper = norm.cdf(max_values, loc=baseline_data["location"], scale=baseline_data["scale"])
        delta = upper - lower
        confidence[:, i] = delta
        baseline_data[f"p(within {threshold:g}x)"] = delta
    baseline_data["confidence_level"] = [assign_confidence_level(row).name for row in confidence]
    assign_calibration_levels(baseline_data, transform=Transform.LOG)
    assert all(np.equal(baseline_data["confidence_level_best_case"], calibration_data["confidence_level_best_case"]))
    return baseline_data


def get_global_calibration_data(data: pd.DataFrame, delta: datetime.timedelta | None = None) -> pd.DataFrame:
    """Collapses calibration data into a single prediction for each molecule.

    The incoming DataFrame may have multiple predictions for each molecule scattered across multiple model versions.
    This function creates a "global" view of the calibration data by only keeping a single prediction for each
    molecule. By default, this is the last prediction before the MIN_DATE of the corresponding assay data, but `delta`
    can be used to increase the delay between prediction and assay (e.g. to match an expected synthesis/testing delay).

    Args:
        data: DataFrame containing calibration data (from get_calibration_data).
        delta: Time delta between prediction and assay.

    Returns:
        DataFrame containing a single row for each compound.
    """
    data = data.copy()
    if delta is None:
        data["max_prediction_date"] = data["assay_date"]
    else:
        data["max_prediction_date"] = data["assay_date"] - delta
    mask = data["model_date"].dt.date < data["max_prediction_date"].dt.date
    data = data[mask]
    data = data.sort_values("model_date")
    return data.drop_duplicates("mol_id", keep="last")
