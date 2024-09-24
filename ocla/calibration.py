"""Model calibration."""

import datetime
import logging
from enum import Enum, auto

import numpy as np
import pandas as pd

from model_calibration.confidence import CONFIDENCE_LEVELS, ModelConfidence, get_bounds
from model_calibration.utilities import Transform

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
    assert len(confidence_levels) == len(CONFIDENCE_LEVELS)
    best_confidence_levels = np.asarray([ModelConfidence.D] * len(locations), dtype=object)
    for confidence_level, threshold in zip(confidence_levels, CONFIDENCE_LEVELS):
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
    ACCURATE = auto()
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
    data.loc[data["confidence_level_best_case"] == data["confidence_level"], key] = Calibration.ACCURATE.name
    data.loc[data["confidence_level_best_case"] < data["confidence_level"], key] = Calibration.UNDERCONFIDENT.name
    data.loc[data["confidence_level_best_case"] > data["confidence_level"], key] = Calibration.OVERCONFIDENT.name
    return data


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
    if delta is None:
        data["max_prediction_date"] = data["assay_date"]
    else:
        data["max_prediction_date"] = data["assay_date"] - delta
    mask = data["model_date"].dt.date < data["max_prediction_date"].dt.date
    data = data[mask]
    data = data.sort_values("model_date")
    return data.drop_duplicates("mol_id", keep="last")
