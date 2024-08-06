"""Test different versions of specifying a criterion functions.

Here we want to take:
- Few representative algorithms (derivative based, derivative free, least squares)
- One basic criterion function (sum of squares)
- Many ways of specifying derivatives or not.
- Maximize and minimize since this might make a difference

"""

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_array_almost_equal as aaae
from optimagic.examples.criterion_functions import (
    sos_criterion_and_gradient,
    sos_criterion_and_jacobian,
    sos_dict_criterion,
    sos_dict_derivative,
    sos_gradient,
    sos_jacobian,
    sos_pandas_gradient,
    sos_pandas_jacobian,
)
from optimagic.optimization.optimize import maximize, minimize

algorithms = ["scipy_lbfgsb", "scipy_neldermead", "scipy_ls_dogbox"]

ls_algorithms = {"scipy_ls_dogbox"}


scalar_derivatives = [None, sos_gradient, sos_pandas_gradient]

scalar_criterion_and_derivtives = [None, sos_criterion_and_gradient]

ls_derivatives = [None, sos_jacobian, sos_pandas_jacobian]


ls_criterion_and_derivatives = [sos_criterion_and_jacobian]

dict_criterion_and_derivatives = [
    None,
]

MIN_CASES = []
for algo in algorithms:
    if algo in ls_algorithms:
        for deriv in ls_derivatives:
            for crit_and_deriv in ls_criterion_and_derivatives:
                MIN_CASES.append((algo, deriv, crit_and_deriv))
    else:
        for deriv in scalar_derivatives:
            for crit_and_deriv in scalar_criterion_and_derivtives:
                MIN_CASES.append((algo, deriv, crit_and_deriv))


@pytest.mark.parametrize("algorithm, derivative, criterion_and_derivative", MIN_CASES)
def test_derivative_versions_in_minimize(
    algorithm, derivative, criterion_and_derivative
):
    start_params = pd.DataFrame()
    start_params["value"] = [1, 2, 3]

    res = minimize(
        fun=sos_dict_criterion,
        params=start_params,
        algorithm=algorithm,
        jac=derivative,
        fun_and_jac=criterion_and_derivative,
        error_handling="raise",
    )

    aaae(res.params["value"].to_numpy(), np.zeros(3), decimal=4)


def test_dict_derivative():
    start_params = pd.DataFrame()
    start_params["value"] = [1, 2, 3]

    res = minimize(
        fun=sos_dict_criterion,
        params=start_params,
        algorithm="scipy_lbfgsb",
        jac=sos_dict_derivative,
    )

    aaae(res.params["value"].to_numpy(), np.zeros(3))


def neg_sos_criterion(params):
    x = params["value"].to_numpy()
    return -x @ x


def neg_sos_gradient(params):
    grad = params.copy()
    grad["value"] = -2 * grad["value"]
    return grad


def neg_sos_crit_and_grad(params):
    return neg_sos_criterion(params), neg_sos_gradient(params)


neg_derivatives = [None, neg_sos_gradient]
neg_criterion_and_derivatives = [None, neg_sos_crit_and_grad]

MAX_CASES = []
for algo in ["scipy_lbfgsb", "scipy_neldermead"]:
    for deriv in neg_derivatives:
        for crit_and_deriv in neg_criterion_and_derivatives:
            MAX_CASES.append((algo, deriv, crit_and_deriv))


@pytest.mark.parametrize("algorithm, derivative, criterion_and_derivative", MAX_CASES)
def test_derivative_versions_in_maximize(
    algorithm, derivative, criterion_and_derivative
):
    start_params = pd.DataFrame()
    start_params["value"] = [1, 2, 3]

    res = maximize(
        fun=neg_sos_criterion,
        params=start_params,
        algorithm=algorithm,
        jac=derivative,
        fun_and_jac=criterion_and_derivative,
        error_handling="raise",
    )

    aaae(res.params["value"].to_numpy(), np.zeros(3), decimal=4)
