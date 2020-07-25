"""Test the constraints processing."""
import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_array_almost_equal

from estimagic.optimization.optimize_new import minimize


def rosen(x):
    """The Rosenbrock function

    Args:
        x (pd.DataFrame): DataFrame with the parameters in the "value" column.

    """
    x = x["value"].to_numpy()
    return sum(100.0 * (x[1:] - x[:-1] ** 2.0) ** 2.0 + (1 - x[:-1]) ** 2.0)


params = pd.DataFrame()
params["value"] = np.array([1.3, 0.7, 1.0, 1.9, 1.2])
params["lower"] = [-1.0, -1, -1, -1, -1]
params["upper"] = [5.0, 5, 5, 5, 5]


def test_single_optimization_with_list_arguments():
    """Test an easy single optimization."""
    batch_options = {"error_handling": "raise"}
    result = minimize(
        criterion=[rosen],
        params=[params],
        algorithm=["scipy_lbfgsb"],
        batch_evaluator_options=batch_options,
        numdiff_options={"error_handling": "raise_strict"},
    )
    expected_result = np.ones(5)
    assert_array_almost_equal(
        result["solution_params"]["value"].to_numpy(), expected_result, decimal=4
    )


def test_parallel_optimizations_all_arguments_have_same_length():
    """Test a parallel optimization: All inputs are a list of the same length."""
    result = minimize(
        [rosen, rosen],
        [params, params],
        ["scipy_lbfgsb", "scipy_lbfgsb"],
        batch_evaluator_options={"n_cores": 4},
        logging=False,
    )

    res1 = result[0]["solution_params"]["value"].to_numpy()
    res2 = result[1]["solution_params"]["value"].to_numpy()
    expected_result = np.ones(5)

    assert_array_almost_equal(res1, expected_result, decimal=4)
    assert_array_almost_equal(res2, expected_result, decimal=4)


def test_parallel_optimizations_with_logging(tmp_path):
    """Test a parallel optimization: All inputs are a list of the same length."""
    paths = [tmp_path / "1.db", tmp_path / "2.db"]
    result = minimize(
        [rosen, rosen],
        [params, params],
        ["scipy_lbfgsb", "scipy_lbfgsb"],
        batch_evaluator_options={"n_cores": 4},
        logging=paths,
    )

    res1 = result[0]["solution_params"]["value"].to_numpy()
    res2 = result[1]["solution_params"]["value"].to_numpy()
    expected_result = np.ones(5)

    assert_array_almost_equal(res1, expected_result, decimal=4)
    assert_array_almost_equal(res2, expected_result, decimal=4)


def test_lists_different_size():
    """Test if error is raised if arguments entered as list are of different length."""
    with pytest.raises(ValueError):
        minimize(
            [rosen, rosen], [params, params, params], ["scipy_lbfgsb", "scipy_lbfgsb"],
        )


def test_missing_argument():
    """Test if error is raised if an important argument is entered as empty list."""
    with pytest.raises(ValueError):
        minimize(criterion=rosen, params=params, algorithm=[])

    with pytest.raises(ValueError):
        minimize(criterion=rosen, params=[], algorithm="scipy_lbfgsb")

    with pytest.raises(ValueError):
        minimize(criterion=[], params=params, algorithm="scipy_lbfgsb")


def test_wrong_type_criterion():
    """Make sure an error is raised if an argument has a wrong type."""
    with pytest.raises(TypeError):
        minimize(
            [rosen, "error"], [params, params], ["scipy_lbfgsb", "scipy_lbfgsb"],
        )

    with pytest.raises(TypeError):
        minimize("error", params, "scipy_lbfgsb")


def test_broadcasting():
    """Test if broadcasting of arguments that are not entered as list works."""
    result = minimize(
        rosen,
        params,
        ["scipy_lbfgsb", "scipy_lbfgsb"],
        batch_evaluator_options={"n_cores": 4},
        logging=False,
    )
    assert len(result) == 2

    res1 = result[0]["solution_params"]["value"].to_numpy()
    res2 = result[1]["solution_params"]["value"].to_numpy()
    expected_result = np.ones(5)

    assert_array_almost_equal(res1, expected_result, decimal=4)
    assert_array_almost_equal(res2, expected_result, decimal=4)


def test_broadcasting_list_len1():
    """
    Test if broadcasting of arguments that are not entered
    as list works if entered as list of length 1.
    """
    result = minimize(
        [rosen],
        [params],
        ["scipy_lbfgsb", "scipy_lbfgsb"],
        batch_evaluator_options={"n_cores": 4},
        logging=False,
    )
    assert len(result) == 2

    res1 = result[0]["solution_params"]["value"].to_numpy()
    res2 = result[1]["solution_params"]["value"].to_numpy()
    expected_result = np.ones(5)

    assert_array_almost_equal(res1, expected_result, decimal=4)
    assert_array_almost_equal(res2, expected_result, decimal=4)


def test_list_of_constraints():
    """Test if multiple lists of constraints are added."""
    constraints = [{"loc": 3, "type": "fixed", "value": 1.9}]
    result = minimize(
        rosen, params, "scipy_lbfgsb", constraints=[[], constraints], logging=False,
    )
    result_unrestricted = result[0]["solution_params"]["value"].to_numpy()
    result_restricted = result[1]["solution_params"]["value"]
    expected_result_unrestricted = [1, 1, 1, 1, 1]

    assert_array_almost_equal(
        result_unrestricted, expected_result_unrestricted, decimal=4
    )
    assert_array_almost_equal(result_restricted[3], 1.9, decimal=4)
