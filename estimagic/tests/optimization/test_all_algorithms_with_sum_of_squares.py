"""Test the external interface for optimization with for all algorithms sos.

sum of squares is abbreviated as sos throughout the module.

"""
import functools
import warnings
from itertools import product

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose

from estimagic.config import IS_DFOLS_INSTALLED
from estimagic.config import IS_PETSC4PY_INSTALLED
from estimagic.config import IS_PYBOBYQA_INSTALLED
from estimagic.optimization import AVAILABLE_ALGORITHMS
from estimagic.optimization.optimize import maximize
from estimagic.optimization.optimize import minimize

AVAILABLE_ALGORITHMS = [alg for alg in AVAILABLE_ALGORITHMS if alg.startswith('nlopt')]
BOUNDS_FREE_ALGORITHMS = [
    "scipy_neldermead",
    "scipy_conjugate_gradient",
    "scipy_bfgs",
    "scipy_newton_cg",
    "scipy_cobyla",
]

BOUNDS_SUPPORTING_ALGORITHMS = [
    alg for alg in AVAILABLE_ALGORITHMS if alg not in BOUNDS_FREE_ALGORITHMS
]

IMPRECISE_ALGOS = [
    "scipy_powell",
    "scipy_truncated_newton",
    "scipy_trust_constr",
    "scipy_ls_trf",
    "scipy_ls_dogbox",
]


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


# ======================================================================================
# Define example functions
# ======================================================================================


def sos_dict_criterion(params):
    out = {
        "value": (params["value"].to_numpy() ** 2).sum(),
        "contributions": params["value"].to_numpy() ** 2,
        "root_contributions": params["value"].to_numpy(),
    }
    return out


def sos_dict_criterion_with_pd_objects(params):
    out = {
        "value": (params["value"] ** 2).sum(),
        "contributions": params["value"] ** 2,
        "root_contributions": params["value"],
    }
    return out


def sos_scalar_criterion(params):
    return (params["value"].to_numpy() ** 2).sum()


def sos_gradient(params):
    return 2 * params["value"].to_numpy()


def sos_jacobian(params):
    return np.diag(2 * params["value"])


def sos_pandas_gradient(params):
    return 2 * params["value"]


def sos_pandas_jacobian(params):
    return pd.DataFrame(np.diag(2 * params["value"]))


def sos_criterion_and_gradient(params):
    x = params["value"].to_numpy()
    return (x ** 2).sum(), 2 * x


def sos_criterion_and_jacobian(params):
    x = params["value"].to_numpy()
    return {"contributions": x ** 2, "value": (x ** 2).sum()}, np.diag(2 * x)


# ======================================================================================
# Other helper functions for tests
# ======================================================================================


def get_test_cases_for_algorithm(algorithm):
    """Generate list of all possible argument combinations for algorithm."""
    is_least_squares = algorithm in [
        "tao_pounders",
        "nag_dfols",
        "scipy_ls_trf",
        "scipy_ls_dogbox",
    ]
    is_sum = algorithm in ["bhhh"]
    is_scalar = not (is_least_squares or is_sum)

    directions = ["minimize"] if is_least_squares else ["maximize", "minimize"]

    crit_funcs = [sos_dict_criterion]
    if is_scalar:
        crit_funcs.append(sos_scalar_criterion)

    if is_scalar:
        derivatives = [sos_gradient, sos_pandas_gradient, None]
    elif is_sum:
        derivatives = [sos_jacobian, sos_pandas_jacobian, None]
    else:
        derivatives = [None]

    if is_scalar:
        crit_and_derivs = [sos_criterion_and_gradient, None]
    elif is_sum:
        crit_and_derivs = [sos_criterion_and_jacobian, None]
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


# ======================================================================================
# Actual tests
# ======================================================================================

test_cases = []
for alg in AVAILABLE_ALGORITHMS:
    test_cases += get_test_cases_for_algorithm(alg)
test_cases = _skip_tests_with_missing_dependencies(test_cases)


@pytest.mark.parametrize("algo, direction, crit, deriv, crit_and_deriv", test_cases)
def test_without_constraints(algo, direction, crit, deriv, crit_and_deriv):
    params = pd.DataFrame(data=np.ones((2, 1)), columns=["value"])
    params["lower_bound"] = -np.inf
    params["upper_bound"] = np.inf

    optimize_func = minimize if direction == "minimize" else maximize

    res = optimize_func(
        criterion=crit,
        params=params,
        algorithm=algo,
        derivative=deriv,
        criterion_and_derivative=crit_and_deriv,
        log_options={"save_all_arguments": False},
    )

    assert res["success"], f"{algo} did not converge."
    atol = 1e-02 if algo in IMPRECISE_ALGOS else 1e-04
    assert_allclose(
        res["solution_params"]["value"].to_numpy(),
        np.zeros(2),
        atol=atol,
        rtol=0,
    )


# constraints are only applicable to algorithms that support bounds
bound_cases = []
for alg in BOUNDS_SUPPORTING_ALGORITHMS:
    bound_cases += get_test_cases_for_algorithm(alg)
bound_cases = _skip_tests_with_missing_dependencies(bound_cases)


@pytest.mark.slow
@pytest.mark.parametrize("algo, direction, crit, deriv, crit_and_deriv", bound_cases)
def test_with_binding_bounds(algo, direction, crit, deriv, crit_and_deriv):
    params = pd.DataFrame(data=np.array([5, 8, 8, 8, -5]), columns=["value"])
    # the truncated_newton's line search fails if the lower bound of the first
    # parameter is set to 1.0. With 2.0 truncated_newton also converges.
    params["lower_bound"] = [2, -10, -10, -10, -10]
    params["upper_bound"] = [10, 10, 10, 10, -1]
    expected = np.array([2, 0, 0, 0, -1])

    optimize_func = minimize if direction == "minimize" else maximize

    res = optimize_func(
        criterion=crit,
        params=params,
        algorithm=algo,
        derivative=deriv,
        criterion_and_derivative=crit_and_deriv,
    )

    assert res["success"], f"{algo} did not converge."

    atol = 1e-02 if algo in IMPRECISE_ALGOS else 1e-04
    assert_allclose(
        res["solution_params"]["value"].to_numpy(), expected, atol=atol, rtol=0
    )


@pytest.mark.slow
@pytest.mark.parametrize("algo, direction, crit, deriv, crit_and_deriv", bound_cases)
def test_with_fixed_constraint(algo, direction, crit, deriv, crit_and_deriv):
    params = pd.DataFrame(data=[[1], [7.5], [-1], [-2], [1]], columns=["value"])
    params["lower_bound"] = [-10, -10, -10, -10, -10]
    params["upper_bound"] = [10, 10, 10, 10, 10]

    constraints = [{"loc": [1, 3], "type": "fixed", "value": [7.5, -2]}]

    optimize_func = minimize if direction == "minimize" else maximize

    res = optimize_func(
        criterion=crit,
        params=params,
        algorithm=algo,
        derivative=deriv,
        criterion_and_derivative=crit_and_deriv,
        constraints=constraints,
    )

    assert res["success"], f"{algo} did not converge."

    expected = np.array([0, 7.5, 0, -2, 0.0])
    atol = 1e-02 if algo in IMPRECISE_ALGOS else 1e-04
    assert_allclose(
        res["solution_params"]["value"].to_numpy(), expected, atol=atol, rtol=0
    )


@pytest.mark.slow
@pytest.mark.parametrize("algo, direction, crit, deriv, crit_and_deriv", bound_cases)
def test_with_equality_constraint(algo, direction, crit, deriv, crit_and_deriv):
    params = pd.DataFrame(data=[[1], [7.5], [-1], [-2], [1]], columns=["value"])
    params["lower_bound"] = [-10, -10, -10, -10, -10]
    params["upper_bound"] = -params["lower_bound"]

    constraints = [{"loc": [0, 4], "type": "equality"}]

    optimize_func = minimize if direction == "minimize" else maximize

    res = optimize_func(
        criterion=crit,
        params=params,
        algorithm=algo,
        derivative=deriv,
        criterion_and_derivative=crit_and_deriv,
        constraints=constraints,
    )

    assert res["success"], f"{algo} did not converge."

    expected = np.zeros(5)
    atol = 1e-02 if algo in IMPRECISE_ALGOS else 1e-04
    assert_allclose(
        res["solution_params"]["value"].to_numpy(), expected, atol=atol, rtol=0
    )


@pytest.mark.slow
@pytest.mark.parametrize("algo, direction, crit, deriv, crit_and_deriv", bound_cases)
def test_with_pairwise_equality_constraint(
    algo,
    direction,
    crit,
    deriv,
    crit_and_deriv,
):
    params = pd.DataFrame(data=[[1], [2], [1], [2], [1]], columns=["value"])
    params["lower_bound"] = [-10, -10, -10, -10, -10.0]
    params["upper_bound"] = [10, 10, 10, 10, 10]

    constraints = [{"locs": [[0, 1], [2, 3]], "type": "pairwise_equality"}]

    optimize_func = minimize if direction == "minimize" else maximize

    res = optimize_func(
        criterion=crit,
        params=params,
        algorithm=algo,
        derivative=deriv,
        criterion_and_derivative=crit_and_deriv,
        constraints=constraints,
    )

    assert res["success"], f"{algo} did not converge."

    expected = np.zeros(5)
    atol = 1e-02 if algo in IMPRECISE_ALGOS else 1e-04
    assert_allclose(
        res["solution_params"]["value"].to_numpy(), expected, atol=atol, rtol=0
    )


@pytest.mark.slow
@pytest.mark.parametrize("algo, direction, crit, deriv, crit_and_deriv", bound_cases)
def test_with_increasing_constraint(algo, direction, crit, deriv, crit_and_deriv):
    params = pd.DataFrame(data=[[1], [2], [3], [2], [1]], columns=["value"])

    constraints = [{"loc": [0, 1, 2], "type": "increasing"}]

    optimize_func = minimize if direction == "minimize" else maximize

    res = optimize_func(
        criterion=crit,
        params=params,
        algorithm=algo,
        derivative=deriv,
        criterion_and_derivative=crit_and_deriv,
        constraints=constraints,
    )

    assert res["success"], f"{algo} did not converge."

    expected = np.zeros(5)
    atol = 1e-02 if algo in IMPRECISE_ALGOS else 1e-04
    assert_allclose(
        res["solution_params"]["value"].to_numpy(), expected, atol=atol, rtol=0
    )


@pytest.mark.slow
@pytest.mark.parametrize("algo, direction, crit, deriv, crit_and_deriv", bound_cases)
def test_with_decreasing_constraint(algo, direction, crit, deriv, crit_and_deriv):
    params = pd.DataFrame(data=[[1], [2], [3], [2], [1]], columns=["value"])

    constraints = [{"loc": [2, 3, 4], "type": "decreasing"}]

    optimize_func = minimize if direction == "minimize" else maximize

    res = optimize_func(
        criterion=crit,
        params=params,
        algorithm=algo,
        derivative=deriv,
        criterion_and_derivative=crit_and_deriv,
        constraints=constraints,
    )

    assert res["success"], f"{algo} did not converge."

    expected = np.zeros(5)
    atol = 1e-02 if algo in IMPRECISE_ALGOS else 1e-04
    assert_allclose(
        res["solution_params"]["value"].to_numpy(), expected, atol=atol, rtol=0
    )


@pytest.mark.slow
@pytest.mark.parametrize("algo, direction, crit, deriv, crit_and_deriv", bound_cases)
def test_with_linear_constraint(algo, direction, crit, deriv, crit_and_deriv):
    params = pd.DataFrame(data=[[1], [2], [0.1], [0.3], [0.6]], columns=["value"])

    constraints = [{"loc": [2, 3, 4], "type": "linear", "value": 1, "weights": 1}]

    optimize_func = minimize if direction == "minimize" else maximize

    with warnings.catch_warnings():
        # in the case of the trust_constr algorithm we want to ignore the warning
        # that the approximated function appears linear for the tests.
        warnings.simplefilter(action="ignore", category=UserWarning)

        res = optimize_func(
            criterion=crit,
            params=params,
            algorithm=algo,
            derivative=deriv,
            criterion_and_derivative=crit_and_deriv,
            constraints=constraints,
        )

    assert res["success"], f"{algo} did not converge."

    expected = np.array([0, 0, 1 / 3, 1 / 3, 1 / 3])
    atol = 1e-02 if algo in IMPRECISE_ALGOS else 1e-04
    assert_allclose(
        res["solution_params"]["value"].to_numpy(), expected, atol=atol, rtol=0
    )


@pytest.mark.slow
@pytest.mark.parametrize("algo, direction, crit, deriv, crit_and_deriv", bound_cases)
def test_with_probability_constraint(algo, direction, crit, deriv, crit_and_deriv):
    params = pd.DataFrame(data=[[0.3], [0.0], [0.6], [0.1], [5]], columns=["value"])

    constraints = [{"loc": [0, 1, 2, 3], "type": "probability"}]

    optimize_func = minimize if direction == "minimize" else maximize

    res = optimize_func(
        criterion=crit,
        params=params,
        algorithm=algo,
        derivative=deriv,
        criterion_and_derivative=crit_and_deriv,
        constraints=constraints,
    )

    assert res["success"], f"{algo} did not converge."

    expected = np.array([0.25, 0.25, 0.25, 0.25, 0])
    atol = 1e-02 if algo in IMPRECISE_ALGOS else 1e-04
    assert_allclose(
        res["solution_params"]["value"].to_numpy(), expected, atol=atol, rtol=0
    )


@pytest.mark.slow
@pytest.mark.parametrize("algo, direction, crit, deriv, crit_and_deriv", bound_cases)
def test_with_covariance_constraint_no_bounds_distance(
    algo,
    direction,
    crit,
    deriv,
    crit_and_deriv,
):
    params = pd.DataFrame(data=[[1], [0.1], [2], [3], [2]], columns=["value"])

    constraints = [{"loc": [0, 1, 2], "type": "covariance"}]

    optimize_func = minimize if direction == "minimize" else maximize

    res = optimize_func(
        criterion=crit,
        params=params,
        algorithm=algo,
        derivative=deriv,
        criterion_and_derivative=crit_and_deriv,
        constraints=constraints,
    )

    assert res["success"], f"{algo} did not converge."

    expected = np.zeros(5)
    atol = 1e-02 if algo in IMPRECISE_ALGOS else 1e-04
    assert_allclose(
        res["solution_params"]["value"].to_numpy(), expected, atol=atol, rtol=0
    )


@pytest.mark.slow
@pytest.mark.parametrize("algo, direction, crit, deriv, crit_and_deriv", bound_cases)
def test_with_covariance_constraint_bounds_distance(
    algo,
    direction,
    crit,
    deriv,
    crit_and_deriv,
):
    # Note: Robust bounds only have an effect for 3x3 covariance matrices or larger
    params = pd.DataFrame(data=[[1], [0.1], [2], [0.2], [0.3], [3]], columns=["value"])

    constraints = [
        {
            "loc": [0, 1, 2, 3, 4, 5],
            "type": "covariance",
            "bounds_distance": 0.1,
            "robust_bounds": True,
        }
    ]

    optimize_func = minimize if direction == "minimize" else maximize

    res = optimize_func(
        criterion=crit,
        params=params,
        algorithm=algo,
        derivative=deriv,
        criterion_and_derivative=crit_and_deriv,
        constraints=constraints,
    )

    assert res["success"], f"{algo} did not converge."

    expected = np.array([0.1, 0, 0.1, 0, 0, 0.1])
    atol = 1e-02 if algo in IMPRECISE_ALGOS else 1e-04
    assert_allclose(
        res["solution_params"]["value"].to_numpy(), expected, atol=atol, rtol=0
    )


@pytest.mark.slow
@pytest.mark.parametrize("algo, direction, crit, deriv, crit_and_deriv", bound_cases)
def test_with_sdcorr_constraint_no_bounds_distance(
    algo,
    direction,
    crit,
    deriv,
    crit_and_deriv,
):
    params = pd.DataFrame(data=[[1], [2], [0.1], [3], [2]], columns=["value"])

    constraints = [{"loc": [0, 1, 2], "type": "sdcorr"}]

    optimize_func = minimize if direction == "minimize" else maximize
    res = optimize_func(
        criterion=crit,
        params=params,
        algorithm=algo,
        derivative=deriv,
        criterion_and_derivative=crit_and_deriv,
        constraints=constraints,
    )

    assert res["success"], f"{algo} did not converge."

    expected = np.zeros(5)
    atol = 1e-02 if algo in IMPRECISE_ALGOS else 1e-04
    assert_allclose(
        res["solution_params"]["value"].to_numpy(), expected, atol=atol, rtol=0
    )


@pytest.mark.slow
@pytest.mark.parametrize("algo, direction, crit, deriv, crit_and_deriv", bound_cases)
def test_with_sdcorr_constraint_bounds_distance(
    algo,
    direction,
    crit,
    deriv,
    crit_and_deriv,
):
    # Note: Robust bounds only have an effect for 3x3 sdcorr matrices or larger
    params = pd.DataFrame(data=[[1], [2], [3], [0.1], [0.2], [0.3]], columns=["value"])

    constraints = [
        {
            "loc": [0, 1, 2, 3, 4, 5],
            "type": "sdcorr",
            "bounds_distance": 0.1,
            "robust_bounds": True,
        }
    ]

    optimize_func = minimize if direction == "minimize" else maximize

    # filter ignored algo_options warnings to pass the same tolerances to all algorithms
    with warnings.catch_warnings():
        hess_warn_msg = (
            "delta_grad == 0.0. Check if the approximated function is linear. "
            + "If the function is linear better results can be obtained by defining "
            + "the Hessian as zero instead of using quasi-Newton approximations."
        )
        warnings.filterwarnings("ignore", message=hess_warn_msg)

        res = optimize_func(
            criterion=crit,
            params=params,
            algorithm=algo,
            derivative=deriv,
            criterion_and_derivative=crit_and_deriv,
            constraints=constraints,
        )

    assert res["success"], f"{algo} did not converge."

    expected = np.array([0.1, 0.1, 0.1, 0, 0, 0.0])
    atol = 1e-02 if algo in IMPRECISE_ALGOS else 1e-04
    assert_allclose(
        res["solution_params"]["value"].to_numpy(), expected, atol=atol, rtol=0
    )


def test_scipy_lbfgsb_actually_calls_criterion_and_derivative():
    params = pd.DataFrame(data=np.ones((10, 1)), columns=["value"])

    def raising_crit_and_deriv(params):
        raise Exception()

    with pytest.raises(Exception):
        minimize(
            criterion=sos_scalar_criterion,
            params=params,
            algorithm="scipy_lbfgsb",
            criterion_and_derivative=raising_crit_and_deriv,
        )
