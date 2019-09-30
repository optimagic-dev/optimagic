"""Test the constraints processing."""
import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_array_almost_equal

from estimagic.optimization.optimize import minimize


def rosen(x):
    """The Rosenbrock function

    Args:
        x (pd.Series): Series with the parameters.

    """
    return np_rosen(x["value"].to_numpy())


def np_rosen(x):

    return sum(100.0 * (x[1:] - x[:-1] ** 2.0) ** 2.0 + (1 - x[:-1]) ** 2.0)


params = pd.DataFrame()
params["value"] = np.array([1.3, 0.7, 1.0, 1.9, 1.2])
params["fixed"] = [False, False, True, False, False]
params["lower"] = [-1, -1, -1, -1, -1]
params["upper"] = [5, 5, 5, 5, 5]


def test_single_optimization():
    """
    Test an easy single optimization.
    """
    result = minimize(rosen, params, "nlopt_neldermead")[0]["internal_x"]
    expected_result = [1, 1, 1, 1, 1]

    assert_array_almost_equal(result, expected_result, decimal=4)


def test_lists_same_size():
    """
    Test a parallel optimization: All inputs are a list of the same length.
    """
    result = minimize(
        [rosen, rosen], [params, params], ["nlopt_neldermead", "scipy_L-BFGS-B"]
    )

    result_neldermead = result[0][0]["internal_x"]
    result_BFGS = result[1][0]["internal_x"]
    expected_result = [1, 1, 1, 1, 1]

    assert_array_almost_equal(result_neldermead, expected_result, decimal=4)
    assert_array_almost_equal(result_BFGS, expected_result, decimal=4)


def test_lists_different_size():
    """
    Make sure an error is raised if arguments entered as list are of different length
    """
    with pytest.raises(ValueError):
        result = minimize(
            [rosen, rosen],
            [params, params, params],
            ["nlopt_neldermead", "scipy_L-BFGS-B"],
        )


def test_broadcasting():
    """
    Test if broadcasting of arguments that are not entered as list works.
    """
    result = minimize(rosen, params, ["nlopt_neldermead", "scipy_L-BFGS-B"])
    assert len(result) == 2

    result_neldermead = result[0][0]["internal_x"]
    result_BFGS = result[1][0]["internal_x"]
    expected_result = [1, 1, 1, 1, 1]

    assert_array_almost_equal(result_neldermead, expected_result, decimal=4)
    assert_array_almost_equal(result_BFGS, expected_result, decimal=4)


def test_order_of_results():
    """
    Test if order is contained.
    """
    params_new = params.copy()
    params_new["lower"] = [-1, -1, -1, 1.9, -1]
    result = minimize([rosen, rosen], [params, params_new], "nlopt_neldermead")
    result_unrestricted = result[0][0]["internal_x"]
    result_restricted = result[1][0]["internal_x"]
    expected_result_unrestricted = [1, 1, 1, 1, 1]

    assert_array_almost_equal(
        result_unrestricted, expected_result_unrestricted, decimal=4
    )
    assert_array_almost_equal(result_restricted[3], 1.9, decimal=4)
