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

"""Models."""

import logging
from collections.abc import Sequence
from enum import Enum, auto

import numpy as np
import pandas as pd
from rdkit.Chem import MolFromSmiles, MolToSmiles
from rdkit.Chem.MolStandardize.rdMolStandardize import Normalize
from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator
from rdkit.RDLogger import DisableLog
from sklearn.ensemble import RandomForestRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Kernel
from sklearn.model_selection import KFold

from ocla.calibration import assign_calibration_levels
from ocla.confidence import ConfidenceModel, assign_confidence, build_confidence_models
from ocla.utilities import Transform

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
DisableLog("rdApp.*")

RANDOM_SEED = 301


class ModelType(Enum):
    """Model types."""

    GAUSSIAN_PROCESS = auto()
    RANDOM_FOREST = auto()


class TanimotoKernel(Kernel):
    """Tanimoto kernel."""

    def __init__(self):  # pylint: disable=useless-parent-delegation
        """Initializes the kernel."""
        # NOTE(skearnes): This empty __init__ is required to avoid an sklearn RuntimeError about kernel varargs.
        super().__init__()

    def __call__(
        self, features_a: np.ndarray, features_b: np.ndarray | None = None, eval_gradient: bool = False
    ) -> np.ndarray:
        """Evaluates the kernel.

        Args:
            features_a: Array with shape [num_examples_a, num_features].
            features_a: Array with shape [num_examples_b, num_features]. If None, will be assigned to `features_a`.
            eval_gradient: Unused.

        Returns:
            Array with shape [num_examples_a, num_examples_b].
        """
        del eval_gradient  # Unused.
        if features_b is None:
            features_b = features_a
        aa = np.sum(np.square(features_a), axis=1, keepdims=True)
        bb = np.sum(np.square(features_b), axis=1, keepdims=True)
        ab = features_a @ features_b.T
        return ab / (aa + bb.T - ab)

    def diag(self, features: np.ndarray) -> np.ndarray:  # pylint: disable=arguments-renamed
        return np.ones(features.shape[0])

    def is_stationary(self) -> bool:
        return True


class Model:
    """A simple model."""

    def __init__(self, model_type: ModelType, random_state: int = RANDOM_SEED) -> None:
        """Initializes the model.

        Args:
            model_type: Model type.
            random_state: Random seed.
        """
        self._model_type = model_type
        self._random_state = random_state
        self._model = None
        self._training_data = None
        self._confidence_transform = None
        self._cv_data = None
        self._confidence_models = None

    def reset(self) -> None:
        """Creates a new underlying model object."""
        match self.model_type:
            case ModelType.GAUSSIAN_PROCESS:
                model = GaussianProcessRegressor(
                    kernel=TanimotoKernel(), normalize_y=True, random_state=self._random_state
                )
            case ModelType.RANDOM_FOREST:
                model = RandomForestRegressor(random_state=self._random_state)
            case _:
                raise NotImplementedError(self.model_type)
        self._model = model

    @property
    def model(self) -> GaussianProcessRegressor | RandomForestRegressor:
        return self._model

    @property
    def model_type(self) -> ModelType:
        return self._model_type

    @property
    def training_data(self) -> pd.DataFrame:
        """Training data."""
        if self._training_data is None:
            raise ValueError
        return self._training_data

    @property
    def confidence_transform(self) -> Transform:
        """Transform used when calculating confidence levels."""
        if self._confidence_transform is None:
            raise ValueError
        return self._confidence_transform

    @property
    def cv_data(self) -> pd.DataFrame:
        """Cross-validation data."""
        if self._cv_data is None:
            raise ValueError
        return self._cv_data

    @property
    def confidence_models(self) -> dict[float, ConfidenceModel]:
        """Confidence models."""
        if self._confidence_models is None:
            raise ValueError
        return self._confidence_models

    @property
    def confidence_weights(self) -> pd.DataFrame:
        """Confidence model parameter weights and scaling."""
        rows = []
        for threshold, scaler_and_model in self.confidence_models.items():
            if scaler_and_model is None:
                continue
            scaler, model = scaler_and_model
            u_location, u_scale = scaler.mean_.squeeze()
            s_location, s_scale = scaler.scale_.squeeze()
            w_location, w_scale = model.coef_.squeeze()
            intercept = model.intercept_.squeeze().item()
            rows.append(
                {
                    "threshold": threshold,
                    "u_location": u_location,
                    "u_scale": u_scale,
                    "s_location": s_location,
                    "s_scale": s_scale,
                    "w_location": w_location,
                    "w_scale": w_scale,
                    "intercept": intercept,
                }
            )
        return pd.DataFrame(rows)

    @staticmethod
    def normalize(smiles: Sequence[str]) -> tuple[np.ndarray, np.ndarray]:
        """Normalizes SMILES strings.

        Args:
            smiles: List of SMILES strings.

        Returns:
            normalized_smiles: Array of normalized SMILES strings.
            mask: Boolean mask indicating which input SMILES were successfully normalized.
        """
        normalized, mask = [], []
        for value in smiles:
            mol = MolFromSmiles(value)
            if mol is None:
                logger.debug(f"Could not parse SMILES: {value}")
                mask.append(False)
                continue
            normalized_mol = Normalize(mol)
            if normalized_mol is None:
                logger.debug(f"Failed to normalize SMILES: {value}")
                mask.append(False)
                continue
            normalized.append(MolToSmiles(normalized_mol))
            mask.append(True)
        assert len(mask) == len(smiles)
        assert len(normalized) <= len(smiles)
        return np.asarray(normalized), np.asarray(mask)

    @staticmethod
    def featurize(smiles: Sequence[str], radius: int = 2, size: int = 2048) -> np.ndarray:
        """Featurizes a list of SMILES strings.

        Args:
            smiles: List of SMILES strings.
            radius: Fingerprint radius.
            size: Fingerprint size.

        Returns:
            Array with shape [num_examples, num_features].
        """
        generator = GetMorganGenerator(radius=radius, fpSize=size)
        features = np.zeros([len(smiles), size], dtype=int)
        for i, value in enumerate(smiles):
            mol = MolFromSmiles(value)
            if mol is None:
                raise ValueError(f"Could not parse SMILES: {value}")
            features[i, :] = generator.GetFingerprint(mol)
        return features

    def get_features(self, smiles: Sequence[str]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Computes features after normalizing SMILES.

        Args:
            smiles: List of SMILES strings.

        Returns:
            normalized_smiles: Array of normalized SMILES strings.
            mask: Boolean mask indicating which input SMILES were successfully normalized.
            features: Array with shape [num_examples, num_features].
        """
        normalized_smiles, mask = self.normalize(smiles)
        features = self.featurize(normalized_smiles)
        logger.debug(f"Featurized {mask.sum()}/{len(smiles)} SMILES")
        return normalized_smiles, mask, features

    def train(self, data: pd.DataFrame, confidence_transform: Transform, num_folds: int = 10) -> None:
        """Runs cross-validation and returns a model trained on all the data.

        Args:
            data: DataFrame containing `mol_id`, `smiles`, and `label` columns.
            confidence_transform: Transform to use for building confidence models.
            num_folds: Number of cross-validation folds.
        """
        self._confidence_transform = confidence_transform
        if "operator" in data:
            # NOTE(skearnes): Only train on unqualified data, but calibrate on everything.
            mask = data["operator"] == "="
            logger.debug(f"Dropping {len(data) - mask.sum()}/{len(data)} qualified training labels")
            data = data[mask]
        normalized_smiles, mask, features = self.get_features(data["smiles"])
        columns = ["mol_id", "smiles", "label"]
        self._training_data = data.loc[mask, columns].copy()
        self.training_data["normalized_smiles"] = normalized_smiles
        self.training_data["original_label"] = self.training_data["label"]
        logger.debug("Running cross-validation")
        engine = KFold(n_splits=num_folds, random_state=self._random_state, shuffle=True)
        cv_data = []
        for i, (train, test) in enumerate(engine.split(features)):
            self.reset()
            self.model.fit(features[train], self.training_data.iloc[train]["label"].values)
            location, scale = self._predict(features[test])
            test_data = self.training_data.iloc[test]
            cv_data.append(
                pd.DataFrame(
                    {
                        "fold": i,
                        "mol_index": test,
                        "mol_id": test_data["mol_id"],
                        "smiles": test_data["normalized_smiles"],
                        "label": test_data["label"],
                        "location": location,
                        "scale": scale,
                    }
                )
            )
        self._cv_data = pd.concat(cv_data, ignore_index=True)
        assert len(self.cv_data) == len(self.training_data)
        logger.debug("Training full model")
        self.reset()
        self.model.fit(features, self.training_data["label"])
        logger.debug("Training confidence models")
        self._confidence_models = build_confidence_models(
            self.cv_data, transform=confidence_transform, random_state=self._random_state
        )

    def _predict(self, features: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Returns model predictions (location and scale) in the transformed label space.

        Args:
            features: Array with shape [num_examples, num_features].

        Returns:
            location: Array with shape [num_examples] containing predicted means.
            scale: Array with shape [num_examples] containing predicted standard deviations.
        """
        match self.model_type:
            case ModelType.GAUSSIAN_PROCESS:
                return self.model.predict(features, return_std=True)
            case ModelType.RANDOM_FOREST:
                predictions = [tree.predict(features) for tree in self.model.estimators_]
                return np.mean(predictions, axis=0), np.std(predictions, axis=0)
            case _:
                raise NotImplementedError(self.model_type)

    def predict(self, data: pd.DataFrame) -> pd.DataFrame:
        """Returns model predictions (location and scale) in the transformed label space.

        Args:
            data: DataFrame with a `smiles` column.

        Returns:
            Dataframe containing location and scale.
        """
        normalized_smiles, mask, features = self.get_features(data["smiles"])
        location, scale = self._predict(features)
        data = data.reset_index(drop=True)  # Create a copy with a new index.
        data.loc[mask, "normalized_smiles"] = normalized_smiles
        data.loc[mask, "location"] = location
        data.loc[mask, "scale"] = scale
        return data

    def predict_with_confidence(self, data: pd.DataFrame) -> pd.DataFrame:
        """Returns model predictions and confidence levels.

        Args:
            data: DataFrame with a `smiles` column.

        Returns:
            Updated dataframe containing location, scale, and confidence levels.
        """
        predictions = self.predict(data)
        confidence = assign_confidence(predictions, self.confidence_models)
        confidence["confidence_level"] = (
            confidence["confidence_level"].apply(lambda level: level.name if level is not None else None).astype(str)
        )
        combined = pd.concat([predictions, confidence], axis=1)
        assert len(combined) == len(predictions) == len(confidence)
        return combined

    def calibrate(self, test_data: pd.DataFrame) -> pd.DataFrame:
        """Returns calibration data for a test set."""
        predictions = self.predict_with_confidence(test_data)
        return assign_calibration_levels(predictions, transform=self.confidence_transform)
