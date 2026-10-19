"""End-to-end tests for ``error_handling="continue"``.

The test problem has the residuals :math:`r(x) = e^x - e` with optimum
:math:`x^* = (1, 1)` and optimal value 0. Starting at :math:`x_0 = (-2, -2)`, the first
steps of all tested optimizers overshoot into the region :math:`x_i > 1.5`, where the
objective raises an error. The tests cover all supported combinations of function
type, solver type and direction.

"""

from dataclasses import dataclass, field

import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal as aaae

import optimagic as om
from optimagic.exceptions import InvalidFunctionError

START_PARAMS = np.full(2, -2.0)
OPTIMUM = np.ones(2)
FAILURE_THRESHOLD = 1.5

BHHH_XFAIL = pytest.mark.xfail(
    strict=True,
    reason="bhhh steps uphill if the derivative is consistent with the objective.",
)


@dataclass
class FailureCounter:
    n_failures: int = field(default=0)


def _raise_in_failure_region(x, counter):
    if np.any(x > FAILURE_THRESHOLD):
        counter.n_failures += 1
        raise ValueError("Evaluation failed.")


def _get_fun_and_jac(fun_type, direction, counter):
    """Return the objective and its derivative for the test problem.

    Args:
        fun_type: One of "scalar", "likelihood" or "least_squares".
        direction: One of "minimize" or "maximize".
        counter: Counts the evaluations in the failure region.

    Returns:
        The marked objective and its derivative. The derivative has shape (2,) for
        scalar and shape (2, 2) for likelihood and least-squares functions.

    """
    sign = -1.0 if direction == "maximize" else 1.0

    def residuals(x):
        _raise_in_failure_region(x, counter)
        return np.exp(x) - np.e

    if fun_type == "scalar":

        @om.mark.scalar
        def fun(x):
            r = residuals(x)
            return sign * 0.5 * r @ r

        @om.mark.scalar
        def jac(x):
            return sign * residuals(x) * np.exp(x)

    elif fun_type == "likelihood":

        @om.mark.likelihood
        def fun(x):
            return sign * 0.5 * residuals(x) ** 2

        @om.mark.likelihood
        def jac(x):
            return sign * np.diag(residuals(x) * np.exp(x))

    elif fun_type == "least_squares":
        fun = om.mark.least_squares(residuals)

        @om.mark.least_squares
        def jac(x):
            _raise_in_failure_region(x, counter)
            return np.diag(np.exp(x))

    return fun, jac


CASES = [
    # scalar functions
    ("scalar", "scipy_lbfgsb", "minimize", False),
    ("scalar", "scipy_lbfgsb", "maximize", False),
    ("scalar", "scipy_lbfgsb", "minimize", True),
    ("scalar", "scipy_lbfgsb", "maximize", True),
    ("scalar", "scipy_neldermead", "minimize", False),
    ("scalar", "scipy_neldermead", "maximize", False),
    # likelihood functions
    ("likelihood", "scipy_lbfgsb", "minimize", False),
    ("likelihood", "scipy_lbfgsb", "maximize", False),
    pytest.param("likelihood", "bhhh", "minimize", False, marks=BHHH_XFAIL),
    pytest.param("likelihood", "bhhh", "maximize", False, marks=BHHH_XFAIL),
    # least-squares functions
    ("least_squares", "scipy_lbfgsb", "minimize", False),
    pytest.param("least_squares", "bhhh", "minimize", False, marks=BHHH_XFAIL),
    ("least_squares", "scipy_ls_trf", "minimize", False),
    ("least_squares", "scipy_ls_trf", "minimize", True),
    ("least_squares", "pounders", "minimize", False),
]


@pytest.mark.parametrize("fun_type, algorithm, direction, with_jac", CASES)
def test_optimizer_recovers_from_errors(fun_type, algorithm, direction, with_jac):
    counter = FailureCounter()
    fun, jac = _get_fun_and_jac(fun_type, direction, counter)
    optimize = om.maximize if direction == "maximize" else om.minimize

    res = optimize(
        fun=fun,
        jac=jac if with_jac else None,
        params=START_PARAMS,
        algorithm=algorithm,
        error_handling="continue",
    )

    assert counter.n_failures > 0, "The failure region was never evaluated."
    aaae(res.params, OPTIMUM, decimal=4)
    # Penalty values are at least 100 away from the start value, so an optimal value
    # close to 0 means that no penalty value leaked into the result.
    assert abs(res.fun) < 1e-6
    assert res.history is not None
    history_values = [v for v in res.history.fun if v is not None]
    best = max(history_values) if direction == "maximize" else min(history_values)
    assert abs(best) < 1e-6


def test_least_squares_function_cannot_be_maximized():
    fun, _ = _get_fun_and_jac("least_squares", "minimize", FailureCounter())
    with pytest.raises(InvalidFunctionError, match="cannot be maximized"):
        om.maximize(
            fun=fun,
            params=START_PARAMS,
            algorithm="scipy_lbfgsb",
            error_handling="continue",
        )


@pytest.mark.parametrize(
    "fun_type, algorithm",
    [
        ("scalar", "bhhh"),
        ("scalar", "scipy_ls_trf"),
        ("likelihood", "scipy_ls_trf"),
    ],
)
def test_unsupported_solver_type_raises(fun_type, algorithm):
    fun, _ = _get_fun_and_jac(fun_type, "minimize", FailureCounter())
    with pytest.raises(InvalidFunctionError):
        om.minimize(
            fun=fun,
            params=START_PARAMS,
            algorithm=algorithm,
            error_handling="continue",
        )
