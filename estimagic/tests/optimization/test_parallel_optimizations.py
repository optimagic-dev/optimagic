"""Test the constraints processing."""
import numpy as np
import pandas as pd
import pytest
from numba import guvectorize
from numpy.testing import assert_array_almost_equal

from estimagic.optimization.optimize import minimize


def rosen(x):
    """The Rosenbrock function

    Args:
        x (pd.DataFrame): DataFrame with the parameters in the "value" column.

    """
    return np_rosen(x["value"].to_numpy())


def np_rosen(x):

    return sum(100.0 * (x[1:] - x[:-1] ** 2.0) ** 2.0 + (1 - x[:-1]) ** 2.0)


params = pd.DataFrame()
params["value"] = np.array([1.3, 0.7, 1.0, 1.9, 1.2])
params["lower"] = [-1, -1, -1, -1, -1]
params["upper"] = [5, 5, 5, 5, 5]


@guvectorize(
    ["f8[:], f8[:, :], f8[:]"],
    "(n_choices), (n_draws, n_choices) -> ()",
    nopython=True,
    target="parallel",
)
def simulate_emax(static_utilities, draws, emax):
    n_draws, n_choices = draws.shape
    emax_ = 0

    for i in range(n_draws):
        max_utility = 0
        for j in range(n_choices):
            utility = static_utilities[j] + draws[i, j]

            if utility > max_utility or j == 0:
                max_utility = utility

        emax_ += max_utility

    emax[0] = emax_ / n_draws


def rosen_with_guvectorize(x):
    static_utilities = np.arange(1, 5)
    n_draws = 10000
    draws = np.random.randn(n_draws, 4)
    simulate_emax(static_utilities, draws)

    return np_rosen(x["value"].to_numpy())


def test_single_optimization():
    """Test an easy single optimization."""
    result = minimize(rosen, params, "nlopt_neldermead")[1]["value"].to_numpy()
    expected_result = [1, 1, 1, 1, 1]

    assert_array_almost_equal(result, expected_result, decimal=4)


def test_single_optimization_list_len1():
    """Test an easy single optimization."""
    result = minimize([rosen], [params], ["nlopt_neldermead"])[1]["value"].to_numpy()
    expected_result = [1, 1, 1, 1, 1]

    assert_array_almost_equal(result, expected_result, decimal=4)


def test_lists_same_size():
    """Test a parallel optimization: All inputs are a list of the same length."""
    result = minimize(
        [rosen, rosen],
        [params, params],
        ["nlopt_neldermead", "scipy_L-BFGS-B"],
        general_options={"n_cores": 4},
        logging=False,
    )

    result_neldermead = result[0][1]["value"].to_numpy()
    result_bfgs = result[1][1]["value"].to_numpy()
    expected_result = [1, 1, 1, 1, 1]

    assert_array_almost_equal(result_neldermead, expected_result, decimal=4)
    assert_array_almost_equal(result_bfgs, expected_result, decimal=4)


@pytest.mark.xfail
def test_two_parallel_optimizations_with_logging(tmp_path):
    """Test a parallel optimization: All inputs are a list of the same length."""
    paths = [tmp_path / "neldermead.db", tmp_path / "lbfgsb.db"]
    result = minimize(
        [rosen, rosen],
        [params, params],
        ["nlopt_neldermead", "scipy_L-BFGS-B"],
        general_options={"n_cores": 4},
        logging=paths,
    )

    result_neldermead = result[0][1]["value"].to_numpy()
    result_bfgs = result[1][1]["value"].to_numpy()
    expected_result = [1, 1, 1, 1, 1]

    assert_array_almost_equal(result_neldermead, expected_result, decimal=4)
    assert_array_almost_equal(result_bfgs, expected_result, decimal=4)


def test_lists_different_size():
    """Test if error is raised if arguments entered as list are of different length."""
    with pytest.raises(ValueError):
        minimize(
            [rosen, rosen],
            [params, params, params],
            ["nlopt_neldermead", "scipy_L-BFGS-B"],
            general_options={"n_cores": 4},
        )


def test_missing_argument():
    """Test if error is raised if an important argument is entered as empty list."""
    with pytest.raises(ValueError):
        minimize(criterion=rosen, params=params, algorithm=[])

    with pytest.raises(ValueError):
        minimize(criterion=rosen, params=[], algorithm="nlopt_neldermead")

    with pytest.raises(ValueError):
        minimize(criterion=[], params=params, algorithm="nlopt_neldermead")


def test_wrong_type_criterion():
    """Make sure an error is raised if an argument has a wrong type."""
    with pytest.raises(TypeError):
        minimize(
            [rosen, "error"],
            [params, params],
            ["nlopt_neldermead", "scipy_L-BFGS-B"],
            general_options={"n_cores": 1},
        )

    with pytest.raises(TypeError):
        minimize("error", params, "nlopt_neldermead", general_options={"n_cores": 4})


def test_wrong_type_algorithm():
    """Make sure an error is raised if an argument has a wrong type."""
    with pytest.raises(TypeError):
        minimize(
            [rosen, rosen],
            [params, params],
            algorithm=["nlopt_neldermead", rosen],
            general_options={"n_cores": 4},
        )

    with pytest.raises(TypeError):
        minimize(rosen, params, algorithm=rosen, general_options={"n_cores": 4})


def test_wrong_type_dashboard():
    """Make sure an error is raised if an argument has a wrong type."""
    with pytest.raises(TypeError):
        minimize(
            [rosen, rosen],
            [params, params],
            algorithm=["nlopt_neldermead", "nlopt_neldermead"],
            dashboard="yes",
            general_options={"n_cores": 4},
        )

    with pytest.raises(TypeError):
        minimize(
            rosen,
            params,
            algorithm=rosen,
            dashboard="yes",
            general_options={"n_cores": 4},
        )


def test_broadcasting():
    """Test if broadcasting of arguments that are not entered as list works."""
    result = minimize(
        rosen,
        params,
        ["nlopt_neldermead", "scipy_L-BFGS-B"],
        general_options={"n_cores": 4},
        logging=False,
    )
    assert len(result) == 2

    result_neldermead = result[0][1]["value"].to_numpy()
    result_bfgs = result[1][1]["value"].to_numpy()
    expected_result = [1, 1, 1, 1, 1]

    assert_array_almost_equal(result_neldermead, expected_result, decimal=4)
    assert_array_almost_equal(result_bfgs, expected_result, decimal=4)


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
        logging=False,
    )
    assert len(result) == 2

    result_neldermead = result[0][1]["value"].to_numpy()
    result_bfgs = result[1][1]["value"].to_numpy()
    expected_result = [1, 1, 1, 1, 1]

    assert_array_almost_equal(result_neldermead, expected_result, decimal=4)
    assert_array_almost_equal(result_bfgs, expected_result, decimal=4)


def test_list_length_1():
    """Test if broadcasting of arguments that are entered as list of length 1 works."""
    result = minimize(
        [rosen],
        [params],
        ["nlopt_neldermead", "scipy_L-BFGS-B"],
        general_options={"n_cores": 4},
        logging=False,
    )
    assert len(result) == 2

    result_neldermead = result[0][1]["value"].to_numpy()
    result_bfgs = result[1][1]["value"].to_numpy()
    expected_result = [1, 1, 1, 1, 1]

    assert_array_almost_equal(result_neldermead, expected_result, decimal=4)
    assert_array_almost_equal(result_bfgs, expected_result, decimal=4)


def test_order_of_results():
    """Test if order is contained."""
    params_new = params.copy()
    params_new["lower"] = [-1, -1, -1, 1.9, -1]
    result = minimize(
        [rosen, rosen],
        [params, params_new],
        "nlopt_neldermead",
        general_options={"n_cores": 4},
        logging=False,
    )
    result_unrestricted = result[0][1]["value"].to_numpy()
    result_restricted = result[1][1]["value"].to_numpy()
    expected_result_unrestricted = [1, 1, 1, 1, 1]

    assert_array_almost_equal(
        result_unrestricted, expected_result_unrestricted, decimal=4
    )
    assert_array_almost_equal(result_restricted[3], 1.9, decimal=4)


def test_list_of_constraints():
    """Test if multiple lists of constraints are added."""
    constraints = [{"loc": 3, "type": "fixed", "value": 1.9}]
    result = minimize(
        rosen,
        params,
        "nlopt_neldermead",
        constraints=[[], constraints],
        general_options={"n_cores": 4},
        logging=False,
    )
    result_unrestricted = result[0][1]["value"].to_numpy()
    result_restricted = result[1][1]["value"]
    expected_result_unrestricted = [1, 1, 1, 1, 1]

    assert_array_almost_equal(
        result_unrestricted, expected_result_unrestricted, decimal=4
    )
    assert_array_almost_equal(result_restricted[3], 1.9, decimal=4)


@pytest.mark.skipif(
    'sys.platform == "darwin"', reason="This test doesn't pass Mac azure checks."
)
def test_criterion_including_guvectoring():
    """Test if multiple lists of constraints are added."""
    result = minimize(
        rosen_with_guvectorize,
        params,
        ["nlopt_neldermead", "scipy_L-BFGS-B"],
        general_options={"n_cores": 4},
        logging=False,
    )
    assert len(result) == 2

    result_neldermead = result[0][1]["value"].to_numpy()
    result_bfgs = result[1][1]["value"].to_numpy()
    expected_result = [1, 1, 1, 1, 1]

    assert_array_almost_equal(result_neldermead, expected_result, decimal=4)
    assert_array_almost_equal(result_bfgs, expected_result, decimal=4)
