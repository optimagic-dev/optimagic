import numpy as np
import pandas as pd
import pytest
from estimagic.parameters.parameter_bounds import get_bounds
from numpy.testing import assert_array_equal


@pytest.fixture
def params():
    params = {
        "delta": 0.95,
        "utility": pd.DataFrame(
            [[0.5, 0]] * 3, index=["a", "b", "c"], columns=["value", "lower_bound"]
        ),
        "probs": np.array([[0.8, 0.2], [0.3, 0.7]]),
    }
    return params


def test_get_bounds_no_arguments(params):
    got_lower, got_upper = get_bounds(params)

    expected_lower = np.array([np.nan] + 3 * [0] + 4 * [np.nan])
    expected_upper = np.full(8, np.nan)

    assert_array_equal(got_lower, expected_lower)
    assert_array_equal(got_upper, expected_upper)


def test_get_bounds_with_lower_bounds(params):
    lower_bounds = {"delta": 0.1}

    got_lower, got_upper = get_bounds(params, lower_bounds=lower_bounds)

    expected_lower = np.array([0.1] + 3 * [0] + 4 * [np.nan])
    expected_upper = np.full(8, np.nan)

    assert_array_equal(got_lower, expected_lower)
    assert_array_equal(got_upper, expected_upper)


def test_get_bounds_with_upper_bounds(params):
    upper_bounds = {
        "utility": pd.DataFrame(
            [[1]] * 3, index=["a", "b", "c"], columns=["upper_bound"]
        ),
    }
    got_lower, got_upper = get_bounds(params, upper_bounds=upper_bounds)

    expected_lower = np.array([np.nan] + 3 * [0] + 4 * [np.nan])
    expected_upper = np.array([np.nan] + 3 * [1] + 4 * [np.nan])

    assert_array_equal(got_lower, expected_lower)
    assert_array_equal(got_upper, expected_upper)


def test_get_bounds_valueerror(params):
    upper_bounds = {
        "utility": pd.DataFrame([[1]] * 2, index=["a", "b"], columns=["upper_bound"]),
    }
    with pytest.raises(ValueError):
        get_bounds(params, upper_bounds=upper_bounds)


def test_get_bounds_numpy():
    params = np.array([1, 2])
    got_lower, got_upper = get_bounds(params)

    expected = np.array([np.nan, np.nan])

    assert_array_equal(got_lower, expected)
    assert_array_equal(got_upper, expected)
