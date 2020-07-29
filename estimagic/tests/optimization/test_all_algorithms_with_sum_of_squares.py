"""Test the external interface for optimization with for all algorithms sos.

sum of squares is abbreviated as sos throughout the module.

"""
import functools
from itertools import product

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_array_almost_equal as aaae

from estimagic.config import AVAILABLE_ALGORITHMS
from estimagic.optimization.optimize import maximize
from estimagic.optimization.optimize import minimize


# ======================================================================================
# Define example functions
# ======================================================================================


def sos_dict_criterion(params):
    out = {
        "value": (params["value"] ** 2).sum(),
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
    return (x ** 2), np.diag(2 * x)


# ======================================================================================
# Other helper functions for tests
# ======================================================================================


def get_test_cases_for_algorithm(algorithm):
    is_least_squares = algorithm in ["tao_pounders"]
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


@pytest.mark.parametrize("algo, direction, crit, deriv, crit_and_deriv", test_cases)
def test_algorithm_without_constraints(algo, direction, crit, deriv, crit_and_deriv):
    """Simple test with and without closed form derivatives.

    Basically we just test many ways of specifying the same problem.

    """
    params = pd.DataFrame(data=np.ones((10, 1)), columns=["value"])
    params["lower"] = -10
    params["upper"] = 10

    optimize_func = minimize if direction == "minimize" else maximize

    res = optimize_func(
        criterion=crit, params=params, algorithm=algo, error_handling="raise",
    )
    aaae(res["solution_params"]["value"].to_numpy(), np.zeros(10))
