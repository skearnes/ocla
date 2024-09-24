"""Tests for model_calibration.utilities."""

import datetime

import numpy as np
import pytest

from model_calibration.utilities import Transform, get_training_dates


@pytest.mark.parametrize(
    "transform,inputs,expected",
    (
        (Transform.NONE, [1.1, 2.2, 3.3], [1.1, 2.2, 3.3]),
        (Transform.LOG, [10.0, 100.0, 1000.0], [1.0, 2.0, 3.0]),
    ),
)
def test_transform(transform, inputs, expected):
    transformed = transform.apply(np.asarray(inputs))
    np.testing.assert_allclose(transformed, expected)
    np.testing.assert_allclose(transform.reverse(transformed), inputs)


def test_get_dates():
    model_dates = get_training_dates(
        assay_dates=[
            # Week 1.
            datetime.date(2024, 1, 1),
            datetime.date(2024, 1, 2),
            # Week 2.
            datetime.date(2024, 1, 11),
            datetime.date(2024, 1, 12),
            # Week 3.
            datetime.date(2024, 1, 15),
        ],
    )
    assert model_dates == [datetime.date(2024, 1, 7), datetime.date(2024, 1, 14), datetime.date(2024, 1, 21)]
