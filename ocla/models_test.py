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

"""Tests for ocla.models."""

import numpy as np
import pandas as pd
import pytest
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem

from ocla.confidence import Transform
from ocla.models import Model, ModelType, TanimotoKernel

SMILES = ["CCC", "CCO", "CC(=O)N"]


def test_featurize():
    features = Model.featurize(SMILES)
    assert features.shape == (3, 2048)
    assert all(features.sum(axis=1) > 0)


def test_tanimoto_kernel():
    gram_matrix = TanimotoKernel()(Model.featurize(SMILES))
    expected = np.zeros([3, 3], dtype=float)
    generator = AllChem.GetMorganGenerator(radius=2, fpSize=2048)
    for i, smiles_a in enumerate(SMILES):
        mol_a = Chem.MolFromSmiles(smiles_a)
        fingerprint_a = generator.GetFingerprint(mol_a)
        for j, smiles_b in enumerate(SMILES):
            mol_b = Chem.MolFromSmiles(smiles_b)
            fingerprint_b = generator.GetFingerprint(mol_b)
            expected[i, j] = DataStructs.TanimotoSimilarity(fingerprint_a, fingerprint_b)
    np.testing.assert_allclose(gram_matrix, expected)


@pytest.mark.parametrize("model_type", list(ModelType))
def test_train_and_predict(model_type: ModelType):
    labels = [1.1, 2.2, 3.3]
    data = pd.DataFrame({"mol_id": ["a", "b", "c"] * 10, "smiles": SMILES * 10, "label": labels * 10})
    model = Model(model_type)
    model.train(data, confidence_transform=Transform.LOG)
    predictions = model.predict(pd.DataFrame({"smiles": SMILES}))
    np.testing.assert_allclose(predictions["location"], labels)
    np.testing.assert_allclose(predictions["scale"], np.zeros_like(labels), atol=1e-5)
    assert model.cv_data.shape == (30, 7)
    assert model.cv_data["mol_index"].unique().size == len(model.cv_data)
