""" Test the external interface for optimization with different implementations
of three criterion functions: trid, rotated_hyper_ellipsoid, and rosenbrock,
for a subset of/all algorithms.

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
    rosenbrock_contributions,
)
from estimagic.examples.criterion_functions_optimization_tests import (
    rosenbrock_criterion_and_gradient,
)
from estimagic.examples.criterion_functions_optimization_tests import (
    rosenbrock_dict_criterion,
)
from estimagic.examples.criterion_functions_optimization_tests import (
    rosenbrock_dict_criterion_with_pd_objects,
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
    rotated_hyper_ellipsoid_contributions,
)
from estimagic.examples.criterion_functions_optimization_tests import (
    rotated_hyper_ellipsoid_criterion_and_gradient,
)
from estimagic.examples.criterion_functions_optimization_tests import (
    rotated_hyper_ellipsoid_dict_criterion,
)
from estimagic.examples.criterion_functions_optimization_tests import (
    rotated_hyper_ellipsoid_dict_criterion_with_pd_objects,
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
from estimagic.examples.criterion_functions_optimization_tests import (
    trid_dict_criterion_with_pd_objects,
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


# Running all took ~7 hrs. Running one ~50 minutes. Hence, run tests on a representative
# subset of algorithms.
# 1 supporting bounds, 1 bounds-free, 1 least squares
rep_algo_list = ["scipy_bfgs", "scipy_neldermead", "nag_dfols"]

BOUNDS_FREE_ALGORITHMS = ["scipy_neldermead"]
BOUNDS_SUPPORTING_ALGORITHMS = [
    alg for alg in rep_algo_list if alg not in BOUNDS_FREE_ALGORITHMS
]
BOUNDS_SUPPORTING_ALGORITHMS

# ======================================================================================
# Helper functions for tests
# ======================================================================================


def _skip_tests_with_missing_dependencies(test_cases):
    """Skip tests involving optimizers whose dependencies could not be found."""
    dependency_present_to_start_str = {
        IS_PETSC4PY_INSTALLED: "tao_",
        IS_PYBOBYQA_INSTALLED: "nag_pybobyqa",
        IS_DFOLS_INSTALLED: "nag_dfols",
    }

    new_test_cases = []
    for test_case in test_cases:
        for dependency_present, start_str in dependency_present_to_start_str.items():
            if test_case[0].startswith(start_str) and not dependency_present:
                test_case = pytest.param(
                    *test_case,
                    marks=pytest.mark.skip(reason="petsc4py is not installed."),
                )
            else:
                print(f"Skipping {start_str}")
            new_test_cases.append(test_case)

    return new_test_cases


def get_test_cases_for_algorithm(algorithm):
    """Generate list of all possible argument combinations for algorithm."""
    is_least_squares = algorithm in ["tao_pounders", "nag_dfols"]
    is_scalar = not (is_least_squares)

    directions = ["minimize"] if is_least_squares else ["maximize", "minimize"]

    crit_funcs = [rotated_hyper_ellipsoid_dict_criterion]
    if is_scalar:
        crit_funcs.append(rotated_hyper_ellipsoid_scalar_criterion)

    if is_scalar:
        derivatives = [
            rotated_hyper_ellipsoid_gradient,
            rotated_hyper_ellipsoid_pandas_gradient,
            None,
        ]

    else:
        derivatives = [None]

    if is_scalar:
        crit_and_derivs = [rotated_hyper_ellipsoid_criterion_and_gradient, None]
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
for alg in rep_algo_list:
    test_cases += get_test_cases_for_algorithm(alg)
test_cases = _skip_tests_with_missing_dependencies(test_cases)


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
        log_options={"save_all_arguments": False},
    )

    assert res["success"], f"{algo} did not converge."
    atol = 1e-04
    assert_allclose(
        res["solution_params"]["value"].to_numpy(),
        np.zeros(3),
        atol=atol,
        rtol=0,
    )
