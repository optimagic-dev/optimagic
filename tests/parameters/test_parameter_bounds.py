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

    expected_lower = np.array([-np.inf] + 3 * [0] + 4 * [-np.inf])
    expected_upper = np.full(8, np.inf)

    assert_array_equal(got_lower, expected_lower)
    assert_array_equal(got_upper, expected_upper)


def test_get_bounds_with_lower_bounds(params):
    lower_bounds = {"delta": 0.1}

    got_lower, got_upper = get_bounds(params, lower_bounds=lower_bounds)

    expected_lower = np.array([0.1] + 3 * [0] + 4 * [-np.inf])
    expected_upper = np.full(8, np.inf)

    assert_array_equal(got_lower, expected_lower)
    assert_array_equal(got_upper, expected_upper)


def test_get_bounds_with_upper_bounds(params):
    upper_bounds = {
        "utility": pd.DataFrame(
            [[1]] * 3, index=["a", "b", "c"], columns=["upper_bound"]
        ),
    }
    got_lower, got_upper = get_bounds(params, upper_bounds=upper_bounds)

    expected_lower = np.array([-np.inf] + 3 * [0] + 4 * [-np.inf])
    expected_upper = np.array([np.inf] + 3 * [1] + 4 * [np.inf])

    assert_array_equal(got_lower, expected_lower)
    assert_array_equal(got_upper, expected_upper)


def test_get_bounds_valueerror(params):
    # we let the specified bounds have one less item than the params data frame
    upper_bounds = {
        "utility": pd.DataFrame([[1]] * 2, index=["a", "b"], columns=["upper_bound"]),
    }
    with pytest.raises(ValueError):
        get_bounds(params, upper_bounds=upper_bounds)

    lower_bounds = {
        "utility": pd.DataFrame([[1]] * 2, index=["a", "b"], columns=["lower_bound"]),
    }
    with pytest.raises(ValueError):
        get_bounds(params, lower_bounds=lower_bounds)


def test_get_bounds_numpy():
    params = np.array([1, 2])
    got_lower, got_upper = get_bounds(params)

    expected = np.array([np.inf, np.inf])

    assert_array_equal(got_lower, -expected)
    assert_array_equal(got_upper, expected)
