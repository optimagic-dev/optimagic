"""Test the constraints processing."""
import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_array_almost_equal

from numba import vectorize, float64
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
params["lower"] = [-1, -1, -1, -1, -1]
params["upper"] = [5, 5, 5, 5, 5]


def rosen_vec(x):
    val_vec = x["value"].to_numpy()
    return np_rosen_vec(val_vec[0], val_vec[1], val_vec[2], val_vec[3], val_vec[4])


@vectorize([float64(float64, float64, float64, float64, float64)])
def np_rosen_vec(x1, x2, x3, x4, x5):
    x = np.array([x1, x2, x3, x4, x5])
    return np.sum(100.0 * (x[1:] - x[:-1] ** 2.0) ** 2.0 + (1 - x[:-1]) ** 2.0)


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
        [rosen, rosen],
        [params, params],
        ["nlopt_neldermead", "scipy_L-BFGS-B"],
        general_options={"n_cores": 4},
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
            general_options={"n_cores": 4},
        )


def test_wrong_type_criterion():
    """
    Make sure an error is raised if an argument has a wrong type.
    """
    with pytest.raises(ValueError):
        result = minimize(
            [rosen, "error"],
            [params, params],
            ["nlopt_neldermead", "scipy_L-BFGS-B"],
            general_options={"n_cores": 4},
        )

    with pytest.raises(ValueError):
        result = minimize(
            "error", params, "nlopt_neldermead", general_options={"n_cores": 4}
        )


def test_wrong_type_algorithm():
    """
    Make sure an error is raised if an argument has a wrong type.
    """
    with pytest.raises(ValueError):
        result = minimize(
            [rosen, rosen],
            [params, params],
            algorithm=["nlopt_neldermead", rosen],
            general_options={"n_cores": 4},
        )

    with pytest.raises(ValueError):
        result = minimize(
            rosen, params, algorithm=rosen, general_options={"n_cores": 4}
        )


def test_wrong_type_constraints():
    """
    Make sure an error is raised if an argument has a wrong type.
    """
    with pytest.raises(ValueError):
        result = minimize(
            rosen,
            params,
            "nlopt_neldermead",
            constraints={},
            general_options={"n_cores": 4},
        )


def test_wrong_type_dashboard():
    """
    Make sure an error is raised if an argument has a wrong type.
    """
    with pytest.raises(ValueError):
        result = minimize(
            [rosen, rosen],
            [params, params],
            algorithm=["nlopt_neldermead", "nlopt_neldermead"],
            dashboard="yes",
            general_options={"n_cores": 4},
        )

    with pytest.raises(ValueError):
        result = minimize(
            rosen,
            params,
            algorithm=rosen,
            dashboard="yes",
            general_options={"n_cores": 4},
        )


def test_n_cores_not_specified():
    """
    Make sure an error is raised if n_cores is not specified and multiple optimizations should be run.
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
    result = minimize(
        rosen,
        params,
        ["nlopt_neldermead", "scipy_L-BFGS-B"],
        general_options={"n_cores": 4},
    )
    assert len(result) == 2

    result_neldermead = result[0][0]["internal_x"]
    result_BFGS = result[1][0]["internal_x"]
    expected_result = [1, 1, 1, 1, 1]

    assert_array_almost_equal(result_neldermead, expected_result, decimal=4)
    assert_array_almost_equal(result_BFGS, expected_result, decimal=4)


def test_broadcasting_list_len1():
    """
    Test if broadcasting of arguments that are not entered 
    as list works if entered as list of length 1.
    """
    result = minimize(
        [rosen],
        [params],
        ["nlopt_neldermead", "scipy_L-BFGS-B"],
        general_options={"n_cores": 4},
    )
    assert len(result) == 2

    result_neldermead = result[0][0]["internal_x"]
    result_BFGS = result[1][0]["internal_x"]
    expected_result = [1, 1, 1, 1, 1]

    assert_array_almost_equal(result_neldermead, expected_result, decimal=4)
    assert_array_almost_equal(result_BFGS, expected_result, decimal=4)


def test_list_length_1():
    """
    Test if broadcasting of arguments that are entered as list of length 1 works.
    """
    result = minimize(
        [rosen],
        [params],
        ["nlopt_neldermead", "scipy_L-BFGS-B"],
        general_options={"n_cores": 4},
    )
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
    result = minimize(
        [rosen, rosen],
        [params, params_new],
        "nlopt_neldermead",
        general_options={"n_cores": 4},
    )
    result_unrestricted = result[0][0]["internal_x"]
    result_restricted = result[1][0]["internal_x"]
    expected_result_unrestricted = [1, 1, 1, 1, 1]

    assert_array_almost_equal(
        result_unrestricted, expected_result_unrestricted, decimal=4
    )
    assert_array_almost_equal(result_restricted[3], 1.9, decimal=4)


def test_list_of_constraints():
    """
    Test if multiple lists of constraints are added.
    """
    constraints = [{"loc": 3, "type": "fixed", "value": 1.9}]
    result = minimize(
        rosen,
        params,
        "nlopt_neldermead",
        constraints=[[], constraints],
        general_options={"n_cores": 4},
    )
    result_unrestricted = result[0][0]["internal_x"]
    result_restricted = result[1][1]["value"]
    expected_result_unrestricted = [1, 1, 1, 1, 1]

    assert_array_almost_equal(
        result_unrestricted, expected_result_unrestricted, decimal=4
    )
    assert_array_almost_equal(result_restricted[3], 1.9, decimal=4)


def test_criterion_including_guvectoring():
    """
    Test if multiple lists of constraints are added.
    """

    result = minimize(
        rosen_vec,
        params,
        ["nlopt_neldermead", "scipy_L-BFGS-B"],
        general_options={"n_cores": 4},
    )
    assert len(result) == 2

    result_neldermead = result[0][0]["internal_x"]
    result_BFGS = result[1][0]["internal_x"]
    expected_result = [1, 1, 1, 1, 1]

    assert_array_almost_equal(result_neldermead, expected_result, decimal=4)
    assert_array_almost_equal(result_BFGS, expected_result, decimal=4)
