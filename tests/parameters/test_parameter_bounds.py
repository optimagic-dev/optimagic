import numpy as np
import pandas as pd
import pytest
from estimagic.exceptions import InvalidBoundsError
from estimagic.parameters.parameter_bounds import get_bounds
from numpy.testing import assert_array_equal


@pytest.fixture()
def pytree_params():
    pytree_params = {
        "delta": 0.95,
        "utility": pd.DataFrame(
            [[0.5, 0]] * 3, index=["a", "b", "c"], columns=["value", "lower_bound"],
        ),
        "probs": np.array([[0.8, 0.2], [0.3, 0.7]]),
    }
    return pytree_params


@pytest.fixture()
def array_params():
    return np.arange(2)


def test_get_bounds_subdataframe(pytree_params):
    upper_bounds = {
        "utility": pd.DataFrame([[2]] * 2, index=["b", "c"], columns=["value"]),
    }
    lower_bounds = {
        "delta": 0,
        "utility": pd.DataFrame([[1]] * 2, index=["a", "b"], columns=["value"]),
    }
    lb, ub = get_bounds(
        pytree_params, lower_bounds=lower_bounds, upper_bounds=upper_bounds,
    )

    assert np.all(lb[1:3] == np.ones(2))
    assert np.all(ub[2:4] == 2 * np.ones(2))


TEST_CASES = [
    ({"selector": lambda p: p["delta"], "lower_bounds": 0}, None),
    ({"delta": [0, -1]}, None),
    ({"probs": 1}, None),
    ({"probs": np.array([0, 1])}, None),  # wrong size lower bounds
    (None, {"probs": np.array([0, 1])}),  # wrong size upper bounds
]


@pytest.mark.parametrize(("lower_bounds", "upper_bounds"), TEST_CASES)
def test_get_bounds_error(pytree_params, lower_bounds, upper_bounds):
    with pytest.raises(InvalidBoundsError):
        get_bounds(pytree_params, lower_bounds=lower_bounds, upper_bounds=upper_bounds)


def test_get_bounds_no_arguments(pytree_params):
    got_lower, got_upper = get_bounds(pytree_params)

    expected_lower = np.array([-np.inf] + 3 * [0] + 4 * [-np.inf])
    expected_upper = np.full(8, np.inf)

    assert_array_equal(got_lower, expected_lower)
    assert_array_equal(got_upper, expected_upper)


def test_get_bounds_with_lower_bounds(pytree_params):
    lower_bounds = {"delta": 0.1}

    got_lower, got_upper = get_bounds(pytree_params, lower_bounds=lower_bounds)

    expected_lower = np.array([0.1] + 3 * [0] + 4 * [-np.inf])
    expected_upper = np.full(8, np.inf)

    assert_array_equal(got_lower, expected_lower)
    assert_array_equal(got_upper, expected_upper)


def test_get_bounds_with_upper_bounds(pytree_params):
    upper_bounds = {
        "utility": pd.DataFrame([[1]] * 3, index=["a", "b", "c"], columns=["value"]),
    }
    got_lower, got_upper = get_bounds(pytree_params, upper_bounds=upper_bounds)

    expected_lower = np.array([-np.inf] + 3 * [0] + 4 * [-np.inf])
    expected_upper = np.array([np.inf] + 3 * [1] + 4 * [np.inf])

    assert_array_equal(got_lower, expected_lower)
    assert_array_equal(got_upper, expected_upper)


def test_get_bounds_numpy(array_params):
    got_lower, got_upper = get_bounds(array_params)

    expected = np.array([np.inf, np.inf])

    assert_array_equal(got_lower, -expected)
    assert_array_equal(got_upper, expected)


def test_get_bounds_numpy_error(array_params):
    with pytest.raises(InvalidBoundsError):
        get_bounds(
            array_params,
            # lower bounds larger than upper bounds
            lower_bounds=np.ones_like(array_params),
            upper_bounds=np.zeros_like(array_params),
        )
