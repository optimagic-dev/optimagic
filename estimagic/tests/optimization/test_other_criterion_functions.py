""" Test the external interface for optimization with different implementations
of three criterion functions: trid, rotated_hyper_ellipsoid, and rosenbrock,
for a representative subset of available algorithms.

"""
import functools
from itertools import product

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose

from estimagic.config import IS_DFOLS_INSTALLED
from estimagic.config import IS_PETSC4PY_INSTALLED
from estimagic.config import IS_PYBOBYQA_INSTALLED
from estimagic.examples.criterion_functions_optimization_tests import (
    rosenbrock_criterion_and_gradient,
)
from estimagic.examples.criterion_functions_optimization_tests import (
    rosenbrock_dict_criterion,
)
from estimagic.examples.criterion_functions_optimization_tests import (
    rosenbrock_gradient,
)
from estimagic.examples.criterion_functions_optimization_tests import (
    rosenbrock_pandas_gradient,
)
from estimagic.examples.criterion_functions_optimization_tests import (
    rosenbrock_scalar_criterion,
)
from estimagic.examples.criterion_functions_optimization_tests import (
    rotated_hyper_ellipsoid_criterion_and_gradient,
)
from estimagic.examples.criterion_functions_optimization_tests import (
    rotated_hyper_ellipsoid_dict_criterion,
)
from estimagic.examples.criterion_functions_optimization_tests import (
    rotated_hyper_ellipsoid_gradient,
)
from estimagic.examples.criterion_functions_optimization_tests import (
    rotated_hyper_ellipsoid_pandas_gradient,
)
from estimagic.examples.criterion_functions_optimization_tests import (
    rotated_hyper_ellipsoid_scalar_criterion,
)
from estimagic.examples.criterion_functions_optimization_tests import (
    trid_criterion_and_gradient,
)
from estimagic.examples.criterion_functions_optimization_tests import (
    trid_dict_criterion,
)
from estimagic.examples.criterion_functions_optimization_tests import trid_gradient
from estimagic.examples.criterion_functions_optimization_tests import (
    trid_pandas_gradient,
)
from estimagic.examples.criterion_functions_optimization_tests import (
    trid_scalar_criterion,
)
from estimagic.optimization.optimize import maximize
from estimagic.optimization.optimize import minimize


# Take a representative subset of algorithms for running tests - one least_squares
# algorithm (nag_dfols) and few from the scipy library
rep_algo_list = ["scipy_lbfgsb", "scipy_slsqp", "nag_pybobyqa", "nag_dfols"]

# ======================================================================================
# Helper functions for tests
# ======================================================================================


def _skip_tests_with_missing_dependencies(test_cases):
    """Skip tests involving optimizers whose dependencies could not be found."""
    new_test_cases = []
    for test_case in test_cases:
        needs_skipping, reason = _get_skipping_info(test_case)
        if needs_skipping:
            test_case = pytest.param(
                *test_case,
                marks=pytest.mark.skip(reason=reason),
            )

        new_test_cases.append(test_case)

    return new_test_cases


def _get_skipping_info(test_case):
    installation_info = {
        "tao_": IS_PETSC4PY_INSTALLED,
        "nag_pybobyqa": IS_PYBOBYQA_INSTALLED,
        "nag_dfols": IS_DFOLS_INSTALLED,
    }

    reasons = {
        "tao_": "petsc4py is not installed",
        "nag_pybobyqa": "pybobyqa is not installed",
        "nag_dfols": "dfols is not installed",
    }

    algo_name = test_case[0]
    needs_skipping = False
    reason = None
    for substring, is_installed in installation_info.items():
        if algo_name.startswith(substring) and not is_installed:
            needs_skipping = True
            reason = reasons[substring]

    return needs_skipping, reason


# Trid function cannot be represented as a least squares problem. Hence we do not
# generate testcases involving least_squares algorithms
def get_trid_test_cases_for_algorithm(algorithm):
    """Given trid function, generate list of all possible argument combinations
    for each algorithm."""
    is_least_squares = algorithm in ["nag_dfols"]
    is_scalar = not (is_least_squares)

    if is_scalar:
        directions = ["maximize", "minimize"]
    else:
        pass

    crit_funcs = [trid_dict_criterion]
    if is_scalar:
        crit_funcs.append(trid_scalar_criterion)

    if is_scalar:
        derivatives = [trid_gradient, trid_pandas_gradient, None]
    else:
        pass

    if is_scalar:
        crit_and_derivs = [trid_criterion_and_gradient, None]
    else:
        pass

    test_cases = []

    if is_scalar:

        prod_list = [directions, crit_funcs, derivatives, crit_and_derivs]

        for direction, crit, deriv, c_and_d in product(*prod_list):
            if direction == "maximize":
                case = (
                    algorithm,
                    direction,
                    switch_sign(crit),
                    switch_sign(deriv),
                    switch_sign(c_and_d),
                )
            else:
                case = (algorithm, direction, crit, deriv, c_and_d)
            test_cases.append(case)
    else:
        pass
    return test_cases


# Define dictionaries with different implementations of each criterion function as keys
rosenbrock_criterion_functions = {
    "scalar_criterion": rosenbrock_scalar_criterion,
    "dict_criterion": rosenbrock_dict_criterion,
    "criterion_and_derivative": rosenbrock_criterion_and_gradient,
    "gradient": rosenbrock_gradient,
    "pandas_gradient": rosenbrock_pandas_gradient,
}

rhe_criterion_functions = {
    "scalar_criterion": rotated_hyper_ellipsoid_scalar_criterion,
    "dict_criterion": rotated_hyper_ellipsoid_dict_criterion,
    "criterion_and_derivative": rotated_hyper_ellipsoid_criterion_and_gradient,
    "gradient": rotated_hyper_ellipsoid_gradient,
    "pandas_gradient": rotated_hyper_ellipsoid_pandas_gradient,
}


def get_test_cases_for_algorithm_and_criterion(algorithm, criterion_functions):
    """Generate list of all possible argument combinations for each criterion
    function and algorithm.
    Args:
    criterion_functions (dict)
    .....
    """
    is_least_squares = algorithm in ["nag_dfols"]
    is_scalar = not (is_least_squares)

    directions = ["minimize"] if is_least_squares else ["maximize", "minimize"]

    crit_funcs = [criterion_functions["dict_criterion"]]
    if is_scalar:
        crit_funcs.append(criterion_functions["scalar_criterion"])

    if is_scalar:
        derivatives = [
            criterion_functions["gradient"],
            criterion_functions["pandas_gradient"],
            None,
        ]
    else:
        derivatives = [None]

    if is_scalar:
        crit_and_derivs = [criterion_functions["criterion_and_derivative"], None]
    else:
        crit_and_derivs = [None]

    prod_list = [directions, crit_funcs, derivatives, crit_and_derivs]

    test_cases = []
    for direction, crit, deriv, c_and_d in product(*prod_list):
        if direction == "maximize":
            case = (
                algorithm,
                direction,
                switch_sign(crit),
                switch_sign(deriv),
                switch_sign(c_and_d),
            )
        else:
            case = (algorithm, direction, crit, deriv, c_and_d)
        test_cases.append(case)
    return test_cases


def switch_sign(func):
    if func is None:
        wrapper = None
    else:

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            unswitched = func(*args, **kwargs)
            if isinstance(unswitched, dict):
                switched = {key: -val for key, val in unswitched.items()}
            elif isinstance(unswitched, tuple):
                switched = []
                for entry in unswitched:
                    if isinstance(entry, dict):
                        switched.append({key: -val for key, val in entry.items()})
                    else:
                        switched.append(-entry)
                switched = tuple(switched)
            else:
                switched = -unswitched
            return switched

    return wrapper


trid_test_cases = []
rhe_test_cases = []
rosenbrock_test_cases = []
for alg in rep_algo_list:
    trid_test_cases += get_trid_test_cases_for_algorithm(alg)
    rhe_test_cases += get_test_cases_for_algorithm_and_criterion(
        alg, rhe_criterion_functions
    )
    rosenbrock_test_cases += get_test_cases_for_algorithm_and_criterion(
        alg, rosenbrock_criterion_functions
    )


test_cases = trid_test_cases + rhe_test_cases + rosenbrock_test_cases
test_cases = _skip_tests_with_missing_dependencies(test_cases)
test_cases


@pytest.mark.slow
@pytest.mark.parametrize("algo, direction, crit, deriv, crit_and_deriv", test_cases)
def test_without_constraints(algo, direction, crit, deriv, crit_and_deriv):
    params = pd.DataFrame(data=np.array([1, 2, 3]), columns=["value"])
    params["lower_bound"] = -np.inf
    params["upper_bound"] = np.inf

    optimize_func = minimize if direction == "minimize" else maximize

    res = optimize_func(
        criterion=crit,
        params=params,
        algorithm=algo,
        derivative=deriv,
        criterion_and_derivative=crit_and_deriv,
        logging=False,
    )

    if crit.__name__.startswith("trid"):
        expected = np.array([3, 4, 3])
    elif crit.__name__.startswith("rotated_hyper_ellipsoid"):
        expected = np.zeros(3)
    else:
        expected = np.ones(3)

    assert res["success"], f"{algo} did not converge."
    atol = 1e-04
    assert_allclose(
        res["solution_params"]["value"].to_numpy(),
        expected,
        atol=atol,
        rtol=0,
    )


@pytest.mark.slow
@pytest.mark.parametrize("algo, direction, crit, deriv, crit_and_deriv", test_cases)
def test_with_fixed_constraint(algo, direction, crit, deriv, crit_and_deriv):
    params = pd.DataFrame(data=[[1], [3], [2]], columns=["value"])
    params["lower_bound"] = [-5, -5, -5]
    params["upper_bound"] = [5, 5, 5]

    constraints = [{"loc": [0], "type": "fixed", "value": [1]}]

    optimize_func = minimize if direction == "minimize" else maximize

    res = optimize_func(
        criterion=crit,
        params=params,
        algorithm=algo,
        derivative=deriv,
        criterion_and_derivative=crit_and_deriv,
        constraints=constraints,
        logging=False,
    )

    if crit.__name__.startswith("trid"):
        expected = np.array([1, 2.666666667, 2.333333333])
    elif crit.__name__.startswith("rotated_hyper_ellipsoid"):
        expected = np.array([1, 0, 0])
    else:
        expected = np.ones(3)

    assert res["success"], f"{algo} did not converge."
    atol = 1e-04
    assert_allclose(
        res["solution_params"]["value"].to_numpy(), expected, atol=atol, rtol=0
    )


@pytest.mark.slow
@pytest.mark.parametrize("algo, direction, crit, deriv, crit_and_deriv", test_cases)
def test_with_equality_constraint(algo, direction, crit, deriv, crit_and_deriv):
    params = pd.DataFrame(data=[[2], [2], [2]], columns=["value"])
    params["lower_bound"] = [-5, -5, -5]
    params["upper_bound"] = -params["lower_bound"]

    constraints = [{"loc": [0, 1, 2], "type": "equality"}]

    optimize_func = minimize if direction == "minimize" else maximize

    res = optimize_func(
        criterion=crit,
        params=params,
        algorithm=algo,
        derivative=deriv,
        criterion_and_derivative=crit_and_deriv,
        constraints=constraints,
        logging=False,
    )

    if crit.__name__.startswith("trid"):
        expected = np.array([3, 3, 3])
    elif crit.__name__.startswith("rotated_hyper_ellipsoid"):
        expected = np.array([0, 0, 0])
    else:
        expected = np.ones(3)
    assert res["success"], f"{algo} did not converge."

    atol = 1e-04
    assert_allclose(
        res["solution_params"]["value"].to_numpy(), expected, atol=atol, rtol=0
    )


@pytest.mark.slow
@pytest.mark.parametrize("algo, direction, crit, deriv, crit_and_deriv", test_cases)
def test_with_pairwise_equality_constraint(
    algo,
    direction,
    crit,
    deriv,
    crit_and_deriv,
):
    params = pd.DataFrame(data=[[2], [2], [3]], columns=["value"])
    params["lower_bound"] = [-5, -5, -5]
    params["upper_bound"] = [5, 5, 5]

    constraints = [{"locs": [0, 1], "type": "pairwise_equality"}]

    optimize_func = minimize if direction == "minimize" else maximize

    res = optimize_func(
        criterion=crit,
        params=params,
        algorithm=algo,
        derivative=deriv,
        criterion_and_derivative=crit_and_deriv,
        constraints=constraints,
        logging=False,
    )

    if crit.__name__.startswith("trid"):
        expected = np.array([3.333333333, 3.333333333, 2.666666667])
    elif crit.__name__.startswith("rotated_hyper_ellipsoid"):
        expected = np.array([0, 0, 0])
    else:
        expected = np.ones(3)
    assert res["success"], f"{algo} did not converge."

    atol = 1e-04
    assert_allclose(
        res["solution_params"]["value"].to_numpy(), expected, atol=atol, rtol=0
    )


@pytest.mark.slow
@pytest.mark.parametrize("algo, direction, crit, deriv, crit_and_deriv", test_cases)
def test_with_increasing_constraint(algo, direction, crit, deriv, crit_and_deriv):
    params = pd.DataFrame(data=[[1], [2], [3]], columns=["value"])

    constraints = [{"loc": [1, 2], "type": "increasing"}]

    optimize_func = minimize if direction == "minimize" else maximize

    res = optimize_func(
        criterion=crit,
        params=params,
        algorithm=algo,
        derivative=deriv,
        criterion_and_derivative=crit_and_deriv,
        constraints=constraints,
        logging=False,
    )

    if crit.__name__.startswith("trid"):
        expected = np.array([2.666666667, 3.3333333, 3.3333333])
    elif crit.__name__.startswith("rotated_hyper_ellipsoid"):
        expected = np.array([0, 0, 0])
    else:
        expected = np.ones(3)
    assert res["success"], f"{algo} did not converge."

    atol = 1e-04
    assert_allclose(
        res["solution_params"]["value"].to_numpy(), expected, atol=atol, rtol=0
    )


@pytest.mark.slow
@pytest.mark.parametrize("algo, direction, crit, deriv, crit_and_deriv", test_cases)
def test_with_decreasing_constraint(algo, direction, crit, deriv, crit_and_deriv):
    params = pd.DataFrame(data=[[2], [1], [3]], columns=["value"])

    constraints = [{"loc": [0, 1], "type": "decreasing"}]

    optimize_func = minimize if direction == "minimize" else maximize

    res = optimize_func(
        criterion=crit,
        params=params,
        algorithm=algo,
        derivative=deriv,
        criterion_and_derivative=crit_and_deriv,
        constraints=constraints,
        logging=False,
    )

    if crit.__name__.startswith("trid"):
        expected = np.empty(3)
    elif crit.__name__.startswith("rotated_hyper_ellipsoid"):
        expected = np.array([0, 0, 0])
    else:
        expected = np.ones(3)
    assert res["success"], f"{algo} did not converge."

    atol = 1e-04
    assert_allclose(
        res["solution_params"]["value"].to_numpy(), expected, atol=atol, rtol=0
    )


@pytest.mark.slow
@pytest.mark.parametrize("algo, direction, crit, deriv, crit_and_deriv", test_cases)
def test_with_linear_constraint(algo, direction, crit, deriv, crit_and_deriv):
    params = pd.DataFrame(data=[[2], [1], [3]], columns=["value"])

    constraints = [{"loc": [0, 1], "type": "linear", "value": 4, "weights": [1, 2]}]

    optimize_func = minimize if direction == "minimize" else maximize

    res = optimize_func(
        criterion=crit,
        params=params,
        algorithm=algo,
        derivative=deriv,
        criterion_and_derivative=crit_and_deriv,
        constraints=constraints,
        logging=False,
    )

    if crit.__name__.startswith("trid"):
        expected = np.array([1.185185185, 1.4074074069999998, 1.703703704])
    elif crit.__name__.startswith("rotated_hyper_ellipsoid"):
        expected = np.array([0.571428571, 1.714285714, 0])
    else:
        expected = np.empty(3)
    assert res["success"], f"{algo} did not converge."

    atol = 1e-04
    assert_allclose(
        res["solution_params"]["value"].to_numpy(), expected, atol=atol, rtol=0
    )


@pytest.mark.slow
@pytest.mark.parametrize("algo, direction, crit, deriv, crit_and_deriv", test_cases)
def test_with_probability_constraint(algo, direction, crit, deriv, crit_and_deriv):
    params = pd.DataFrame(data=[[0.5], [0.5], [3]], columns=["value"])

    constraints = [{"loc": [0, 1], "type": "probability"}]

    optimize_func = minimize if direction == "minimize" else maximize

    res = optimize_func(
        criterion=crit,
        params=params,
        algorithm=algo,
        derivative=deriv,
        criterion_and_derivative=crit_and_deriv,
        constraints=constraints,
        logging=False,
    )

    if crit.__name__.startswith("trid"):
        expected = np.array([0.272727273, 0.727272727, 1.363636364])
    elif crit.__name__.startswith("rotated_hyper_ellipsoid"):
        expected = np.array([0.4, 0.6, 0])
    else:
        expected = np.empty(3)
    assert res["success"], f"{algo} did not converge."

    atol = 1e-04
    assert_allclose(
        res["solution_params"]["value"].to_numpy(), expected, atol=atol, rtol=0
    )


@pytest.mark.slow
@pytest.mark.parametrize("algo, direction, crit, deriv, crit_and_deriv", test_cases)
def test_with_covariance_constraint_no_bounds_distance(
    algo,
    direction,
    crit,
    deriv,
    crit_and_deriv,
):
    params = pd.DataFrame(data=[[1], [1], [1]], columns=["value"])

    constraints = [{"loc": [0, 1, 2], "type": "covariance"}]

    optimize_func = minimize if direction == "minimize" else maximize

    res = optimize_func(
        criterion=crit,
        params=params,
        algorithm=algo,
        derivative=deriv,
        criterion_and_derivative=crit_and_deriv,
        constraints=constraints,
        logging=False,
    )

    if crit.__name__.startswith("trid"):
        expected = np.empty(3)
    elif crit.__name__.startswith("rotated_hyper_ellipsoid"):
        expected = np.array([0, 0, 0])
    else:
        expected = np.ones(3)
    assert res["success"], f"{algo} did not converge."

    atol = 1e-04
    assert_allclose(
        res["solution_params"]["value"].to_numpy(), expected, atol=atol, rtol=0
    )


@pytest.mark.slow
@pytest.mark.parametrize("algo, direction, crit, deriv, crit_and_deriv", test_cases)
def test_with_sdcorr_constraint_no_bounds_distance(
    algo,
    direction,
    crit,
    deriv,
    crit_and_deriv,
):
    params = pd.DataFrame(data=[[1], [1], [1]], columns=["value"])

    constraints = [{"loc": [0, 1, 2], "type": "sdcorr"}]

    optimize_func = minimize if direction == "minimize" else maximize
    res = optimize_func(
        criterion=crit,
        params=params,
        algorithm=algo,
        derivative=deriv,
        criterion_and_derivative=crit_and_deriv,
        constraints=constraints,
        logging=False,
    )

    if crit.__name__.startswith("trid"):
        expected = np.empty(3)
    elif crit.__name__.startswith("rotated_hyper_ellipsoid"):
        expected = np.array([0, 0, 0])
    else:
        expected = np.ones(3)
    assert res["success"], f"{algo} did not converge."

    atol = 1e-04
    assert_allclose(
        res["solution_params"]["value"].to_numpy(), expected, atol=atol, rtol=0
    )
