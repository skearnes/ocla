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

"""Model confidence."""

import logging
from enum import Enum, auto

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from model_calibration.utilities import Transform

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelConfidence(Enum):
    """Model confidence levels."""

    A = auto()
    B = auto()
    C = auto()
    D = auto()


CONFIDENCE_THRESHOLDS: tuple[float, ...] = (1.5, 3.0, 10.0)
ConfidenceModel = tuple[StandardScaler, LogisticRegression]


def get_bounds(labels: np.ndarray, threshold: float, transform: Transform) -> tuple[np.ndarray, np.ndarray]:
    """Returns (min, max) bounds for the given confidence threshold.

    Args:
        labels: Float array with shope [num_values] containing experimental values. (Note that confidence levels are
            based on the likelihood of being within a threshold of the *true* value, not the predicted value.)
        threshold: Float confidence threshold.
        transform: Transform indicating the shape of the label space.

    Returns:
        min_values: Float array with shape [num_values] containing lower confidence bounds.
        max_values: Float array with shape [num_values] containing upper confidence bounds.
    """
    match transform:
        case Transform.LOG:
            delta = np.log10(threshold)
            min_values = labels - delta
            max_values = labels + delta
        case Transform.NONE:
            min_values = labels / threshold
            max_values = labels * threshold
        case _:
            raise NotImplementedError(transform)
    return min_values, max_values


def build_confidence_models(
    cv_results: pd.DataFrame, transform: Transform, random_state: int
) -> dict[float, ConfidenceModel | None]:
    """Builds models used to predict confidence."""
    features = np.stack([cv_results["location"], cv_results["scale"]]).T
    assert features.shape == (len(cv_results), 2)
    confidence_models = {}
    for threshold in CONFIDENCE_THRESHOLDS:
        locations = cv_results["location"].values
        labels = cv_results["label"].values
        min_values, max_values = get_bounds(labels, threshold, transform)
        confidence_labels = np.logical_and(locations >= min_values, locations <= max_values)
        scaler = StandardScaler()
        # NOTE(skearnes): Use penalty=None since regularization requires feature scaling.
        model = LogisticRegression(penalty=None, random_state=random_state)
        try:
            model.fit(scaler.fit_transform(features), confidence_labels)
            confidence_models[threshold] = (scaler, model)
        except ValueError as error:
            logger.debug(error)
            confidence_models[threshold] = None
    return confidence_models


def predict_confidence(predictions: pd.DataFrame, confidence_models: dict[float, ConfidenceModel]) -> np.ndarray:
    """Predicts confidence level probabilities.

    Args:
        predictions: DataFrame containing predictions.
        confidence_models: Dict mapping confidence thresholds to models.

    Returns:
        Numpy array with shape [num_molecules, num_confidence_levels] containing the probability for each
        confidence model.
    """
    features = np.stack([predictions["location"], predictions["scale"]]).T
    assert features.shape == (len(predictions), 2)
    confidence = np.zeros((len(predictions), len(confidence_models)), dtype=float)
    for i, threshold in enumerate(CONFIDENCE_THRESHOLDS):
        if confidence_models[threshold] is None:
            confidence[:, i] = np.nan
            continue
        scaler, model = confidence_models[threshold]
        confidence[:, i] = model.predict_proba(scaler.transform(features))[:, 1]
    return confidence


def assign_confidence(
    predictions: pd.DataFrame, confidence_models: dict[float, ConfidenceModel | None]
) -> pd.DataFrame:
    """Assigns confidence levels to predictions.

    Args:
        predictions: DataFrame containing predictions.
        confidence_models: Dict mapping confidence thresholds to models.

    Returns:
        DataFrame containing confidence probabilities and assignments.
    """
    confidence_predictions = predict_confidence(predictions, confidence_models=confidence_models)
    rows = []
    for confidence_row in confidence_predictions:
        row = {f"p(within {threshold:g}x)": confidence_row[j] for j, threshold in enumerate(CONFIDENCE_THRESHOLDS)}
        row["confidence_level"] = assign_confidence_level(confidence_row)
        rows.append(row)
    return pd.DataFrame(rows)


def assign_confidence_level(row: np.ndarray, threshold: float = 0.8) -> ModelConfidence:
    """Assigns a confidence level to predictions.

    Args:
        row: Numpy array with shape [num_confidence_levels] containing the probability for each confidence level.
        threshold: Probability threshold for assigning confidence levels.

    Returns:
        ModelConfidence assignment.
    """
    expected_shape = (len(CONFIDENCE_THRESHOLDS),)
    if row.shape != expected_shape:
        raise ValueError(f"Expected shape {expected_shape}; got {row.shape}")
    if row[0] >= threshold:
        return ModelConfidence.A
    if row[1] >= threshold:
        return ModelConfidence.B
    if row[2] >= threshold:
        return ModelConfidence.C
    return ModelConfidence.D
