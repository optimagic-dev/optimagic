import numpy as np
import pytest
from estimagic.optimization.tranquilo.aggregate_models import (
    aggregator_identity,
    aggregator_information_equality_linear,
    aggregator_least_squares_linear,
    aggregator_sum,
)
from estimagic.optimization.tranquilo.models import ScalarModel, VectorModel
from numpy.testing import assert_array_equal


@pytest.mark.parametrize("square_terms", [np.arange(9).reshape(1, 3, 3), None])
def test_aggregator_identity(square_terms):
    vector_model = VectorModel(
        intercepts=np.array([2.0]),
        linear_terms=np.arange(3).reshape(1, 3),
        square_terms=square_terms,
    )

    if square_terms is None:
        expected_square_terms = np.zeros((3, 3))
    else:
        expected_square_terms = np.arange(9).reshape(3, 3)

    got = ScalarModel(*aggregator_identity(vector_model))

    assert_array_equal(got.intercept, 2.0)
    assert_array_equal(got.linear_terms, np.arange(3))
    assert_array_equal(got.square_terms, expected_square_terms)


def test_aggregator_sum():
    vector_model = VectorModel(
        intercepts=np.array([1.0, 2.0]),
        linear_terms=np.arange(6).reshape(2, 3),
        square_terms=np.arange(18).reshape(2, 3, 3),
    )

    got = ScalarModel(*aggregator_sum(vector_model))

    assert_array_equal(got.intercept, 3.0)
    assert_array_equal(got.linear_terms, np.array([3, 5, 7]))
    assert_array_equal(
        got.square_terms, np.array([[9, 11, 13], [15, 17, 19], [21, 23, 25]])
    )


def test_aggregator_least_squares_linear():
    vector_model = VectorModel(
        intercepts=np.array([0, 2]),
        linear_terms=np.arange(6).reshape(2, 3),
        square_terms=np.arange(18).reshape(2, 3, 3),  # should not be used by aggregator
    )

    got = ScalarModel(*aggregator_least_squares_linear(vector_model))

    assert_array_equal(got.intercept, 4.0)
    assert_array_equal(got.linear_terms, np.array([12, 16, 20]))
    assert_array_equal(
        got.square_terms, np.array([[18, 24, 30], [24, 34, 44], [30, 44, 58]])
    )


def test_aggregator_information_equality_linear():
    vector_model = VectorModel(
        intercepts=np.array([1.0, 2.0]),
        linear_terms=np.arange(6).reshape(2, 3),
        square_terms=np.arange(18).reshape(2, 3, 3),  # should not be used by aggregator
    )

    got = ScalarModel(*aggregator_information_equality_linear(vector_model))

    assert_array_equal(got.intercept, 3.0)
    assert_array_equal(got.linear_terms, np.array([3, 5, 7]))
    assert_array_equal(
        got.square_terms,
        np.array([[-4.5, -6.0, -7.5], [-6.0, -8.5, -11.0], [-7.5, -11.0, -14.5]]),
    )
