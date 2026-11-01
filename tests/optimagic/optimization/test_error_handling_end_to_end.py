r"""End-to-end tests for ``error_handling="continue"``.

All tests use one nonlinear regression with the mean function
:math:`\mu_i(x) = e^{x_0 + x_1 t_i}`, the residuals :math:`r_i = y_i - \mu_i(x)` and the
four observations ``Y`` at ``T``. Least-squares functions return the residuals,
likelihood functions the contributions :math:`r_i^2` and scalar functions their sum.
For maximization, the contributions are multiplied by -1.

Since ``T`` only takes the values 0 and 1, the optimal means are the means of ``Y`` in
both groups, 2.5 and 4.5. Hence, the optimal parameters are
:math:`(\log 2.5, \log 1.8)` and the optimal sum of squared residuals is 1. At the
optimum, all residuals are :math:`\pm 0.5`, so the outer product of gradients is half
the Hessian and bhhh overshoots.

Evaluating the functions fails if :math:`x_0 + x_1` exceeds its value at the optimum
by more than ``FAILURE_MARGIN``, similar to an overflow of the exponential. Starting at
``START_PARAMS``, all tested optimizers step into that region on their way to the
optimum.

"""

from dataclasses import dataclass, field

import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal as aaae

import optimagic as om

T = np.array([0.0, 0.0, 1.0, 1.0])
Y = np.array([2.0, 3.0, 4.0, 5.0])
OPTIMUM = np.array([np.log(2.5), np.log(1.8)])
START_PARAMS = np.full(2, -1.0)
FAILURE_MARGIN = 0.01

BHHH_XFAIL = pytest.mark.xfail(
    strict=True,
    reason="bhhh steps uphill and accepts worse points; fixed in #719.",
)


@dataclass
class FailureCounter:
    n_failures: int = field(default=0)


def _raise_in_failure_region(x, counter):
    if x.sum() > OPTIMUM.sum() + FAILURE_MARGIN:
        counter.n_failures += 1
        raise ValueError("Evaluation failed.")


def _problem(kind, direction, counter):
    """Return the objective function and its derivative for a function kind.

    Least-squares and likelihood functions return arrays of shape (n_obs,) and
    derivatives of shape (n_obs, 2). Scalar functions return floats and derivatives of
    shape (2,).

    """
    sign = 1.0 if direction == "minimize" else -1.0

    def residuals(x):
        _raise_in_failure_region(x, counter)
        return Y - np.exp(x[0] + x[1] * T)

    def residuals_jac(x):
        _raise_in_failure_region(x, counter)
        mean = np.exp(x[0] + x[1] * T)
        return -mean[:, None] * np.column_stack([np.ones_like(T), T])

    def contributions(x):
        return sign * residuals(x) ** 2

    def contributions_jac(x):
        return sign * 2 * residuals(x)[:, None] * residuals_jac(x)

    if kind == "least_squares":
        return om.mark.least_squares(residuals), om.mark.least_squares(residuals_jac)
    if kind == "likelihood":
        return om.mark.likelihood(contributions), om.mark.likelihood(contributions_jac)
    return (
        om.mark.scalar(lambda x: contributions(x).sum()),
        om.mark.scalar(lambda x: contributions_jac(x).sum(axis=0)),
    )


CASES = [
    ("scalar", "minimize", "scipy_lbfgsb", False),
    ("scalar", "maximize", "scipy_lbfgsb", False),
    ("scalar", "minimize", "scipy_lbfgsb", True),
    ("scalar", "maximize", "scipy_lbfgsb", True),
    ("scalar", "minimize", "scipy_neldermead", False),
    ("scalar", "maximize", "scipy_neldermead", False),
    ("likelihood", "minimize", "scipy_lbfgsb", False),
    ("likelihood", "maximize", "scipy_lbfgsb", False),
    pytest.param("likelihood", "minimize", "bhhh", False, marks=BHHH_XFAIL),
    pytest.param("likelihood", "maximize", "bhhh", False, marks=BHHH_XFAIL),
    ("least_squares", "minimize", "scipy_lbfgsb", False),
    pytest.param("least_squares", "minimize", "bhhh", False, marks=BHHH_XFAIL),
    ("least_squares", "minimize", "scipy_ls_trf", False),
    ("least_squares", "minimize", "scipy_ls_trf", True),
    ("least_squares", "minimize", "pounders", False),
]


@pytest.mark.parametrize(("kind", "direction", "algorithm", "with_jac"), CASES)
def test_error_handling_continue(kind, direction, algorithm, with_jac):
    counter = FailureCounter()
    fun, jac = _problem(kind, direction, counter)
    optimize = om.minimize if direction == "minimize" else om.maximize
    res = optimize(
        fun,
        START_PARAMS,
        algorithm=algorithm,
        jac=jac if with_jac else None,
        error_handling="continue",
    )

    assert counter.n_failures > 0, "The failure region was never evaluated."
    aaae(res.params, OPTIMUM, decimal=4)
    # Penalty values are worse than the start value by at least 100, so an optimal
    # value equal to the true one means that no penalty value leaked into the result.
    optimal_value = 1.0 if direction == "minimize" else -1.0
    assert np.isclose(res.fun, optimal_value, rtol=1e-8)
    assert res.history is not None
    history_values = [v for v in res.history.fun if v is not None]
    best = min(history_values) if direction == "minimize" else max(history_values)
    assert np.isclose(best, optimal_value, rtol=1e-8)
