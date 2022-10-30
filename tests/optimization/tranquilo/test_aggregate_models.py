import numpy as np
import pytest
from estimagic.optimization.tranquilo.aggregate_models import aggregator_identity
from estimagic.optimization.tranquilo.aggregate_models import (
    aggregator_information_equality_linear,
)
from estimagic.optimization.tranquilo.aggregate_models import (
    aggregator_least_squares_linear,
)
from estimagic.optimization.tranquilo.aggregate_models import aggregator_sum
from estimagic.optimization.tranquilo.models import ModelInfo
from estimagic.optimization.tranquilo.models import VectorModel
from numpy.testing import assert_array_equal


@pytest.mark.parametrize(
    "model, expected_square_terms",
    [
        ("quadratic", np.arange(9).reshape(3, 3)),
        ("linear", np.zeros(9).reshape(3, 3)),
    ],
)
def test_aggregator_identity(model, expected_square_terms):
    if model == "linear":
        model_info = ModelInfo(
            has_intercepts=True, has_squares=False, has_interactions=False
        )
    else:
        model_info = ModelInfo()

    fvec_center = np.array([2.0])

    vector_model = VectorModel(
        intercepts=None,
        linear_terms=np.arange(3).reshape(1, 3),
        square_terms=np.arange(9).reshape(1, 3, 3),
    )

    got = tuple(aggregator_identity(vector_model, fvec_center, model_info))

    assert got[0] == 2.0
    assert_array_equal(got[1], np.arange(3))
    assert_array_equal(got[2], expected_square_terms)


def test_aggregator_sum():
    model_info = ModelInfo(has_intercepts=False)
    fvec_center = np.array([1.0, 2.0])

    vector_model = VectorModel(
        intercepts=None,
        linear_terms=np.arange(6).reshape(2, 3),
        square_terms=np.arange(18).reshape(2, 3, 3),
    )

    got = tuple(aggregator_sum(vector_model, fvec_center, model_info))

    assert got[0] == 3.0
    assert_array_equal(got[1], np.array([3, 5, 7]))
    assert_array_equal(got[2], np.array([[9, 11, 13], [15, 17, 19], [21, 23, 25]]))


def test_aggregator_least_squares_linear():
    vector_model = VectorModel(
        intercepts=np.array([0, 2]),
        linear_terms=np.arange(6).reshape(2, 3),
        square_terms=np.arange(18).reshape(2, 3, 3),  # should not be used by aggregator
    )

    got = tuple(aggregator_least_squares_linear(vector_model, None, None))

    assert got[0] == 4.0
    assert_array_equal(got[1], np.array([12, 16, 20]))
    assert_array_equal(got[2], np.array([[18, 24, 30], [24, 34, 44], [30, 44, 58]]))


def test_aggregator_information_equality_linear():
    model_info = ModelInfo(has_intercepts=False)
    fvec_center = np.array([1.0, 2.0])

    vector_model = VectorModel(
        intercepts=None,
        linear_terms=np.arange(6).reshape(2, 3),
        square_terms=np.arange(18).reshape(2, 3, 3),  # should not be used by aggregator
    )

    got = tuple(
        aggregator_information_equality_linear(vector_model, fvec_center, model_info)
    )

    assert got[0] == 3.0
    assert_array_equal(got[1], np.array([3, 5, 7]))
    assert_array_equal(
        got[2],
        np.array([[-4.5, -6.0, -7.5], [-6.0, -8.5, -11.0], [-7.5, -11.0, -14.5]]),
    )
