r"""End-to-end tests for ``error_handling="continue"``.

There are two test problems, both maximum likelihood estimations with the linear
predictor :math:`\eta_i(x) = x_0 + x_1 t_i`:

- Poisson regression for scalar and likelihood functions. The loglikelihood
  contributions are :math:`y_i \eta_i - e^{\eta_i} - \log(y_i!)`. The data has a smaller
  variance than the Poisson model implies, so the outer product of gradients
  underestimates the Hessian and bhhh overshoots.
- Gaussian nonlinear regression for least-squares functions, with residuals
  :math:`y_i - e^{\eta_i}`.

Evaluating either objective fails if :math:`x_0 + x_1` exceeds its value at the optimum
by more than ``FAILURE_MARGIN``, similar to an overflow of the exponential. Starting at
``START_PARAMS``, all tested optimizers step into that region on their way to the
optimum. The tests cover all supported combinations of function type, solver type and
direction.

"""

from dataclasses import dataclass, field

import numpy as np
import pytest
import scipy.optimize
from numpy.testing import assert_array_almost_equal as aaae
from scipy.special import gammaln

import optimagic as om
from optimagic.exceptions import InvalidFunctionError
from optimagic.utilities import get_rng

START_PARAMS = np.full(2, -1.0)
FAILURE_MARGIN = 0.01


def _generate_data():
    rng = get_rng(seed=0)
    t = rng.uniform(size=200)
    mean = np.exp(1 + t)
    y_poisson = rng.binomial(10, mean / 10).astype(float)
    y_gaussian = mean + rng.normal(scale=0.2, size=200)
    return t, y_poisson, y_gaussian


T, Y_POISSON, Y_GAUSSIAN = _generate_data()


def _poisson_loglike(x):
    eta = x[0] + x[1] * T
    return Y_POISSON * eta - np.exp(eta) - gammaln(Y_POISSON + 1)


def _poisson_score(x):
    """Return the derivative of `_poisson_loglike` of shape (n_obs, 2)."""
    residuals = Y_POISSON - np.exp(x[0] + x[1] * T)
    return residuals[:, None] * np.column_stack([np.ones_like(T), T])


def _gaussian_residuals(x):
    return Y_GAUSSIAN - np.exp(x[0] + x[1] * T)


def _gaussian_residuals_jac(x):
    """Return the derivative of `_gaussian_residuals` of shape (n_obs, 2)."""
    mean = np.exp(x[0] + x[1] * T)
    return -mean[:, None] * np.column_stack([np.ones_like(T), T])


POISSON_OPTIMUM = scipy.optimize.minimize(
    lambda x: -_poisson_loglike(x).sum(),
    x0=np.zeros(2),
    jac=lambda x: -_poisson_score(x).sum(axis=0),
    method="BFGS",
    options={"gtol": 1e-10},
).x

GAUSSIAN_OPTIMUM = scipy.optimize.least_squares(
    _gaussian_residuals,
    x0=np.zeros(2),
    jac=_gaussian_residuals_jac,
    xtol=1e-15,
    ftol=1e-15,
    gtol=1e-15,
).x


@dataclass
class FailureCounter:
    n_failures: int = field(default=0)


def _raise_in_failure_region(x, optimum, counter):
    if x.sum() > optimum.sum() + FAILURE_MARGIN:
        counter.n_failures += 1
        raise ValueError("Evaluation failed.")


def _get_problem(fun_type, direction, counter):
    """Return the objective, its derivative, the optimum and the optimal value.

    Args:
        fun_type: One of "scalar", "likelihood" or "least_squares".
        direction: One of "minimize" or "maximize".
        counter: Counts the evaluations in the failure region.

    Returns:
        The marked objective, its marked derivative, the optimal parameters of shape
        (2,) and the aggregated objective value at the optimum. The derivative has
        shape (2,) for scalar and shape (n_obs, 2) for likelihood and least-squares
        functions.

    """
    # The loglikelihood is maximized; minimization uses its negative.
    sign = 1.0 if direction == "maximize" else -1.0

    if fun_type in ("scalar", "likelihood"):
        optimum = POISSON_OPTIMUM

        def contributions(x):
            _raise_in_failure_region(x, optimum, counter)
            return sign * _poisson_loglike(x)

        def contributions_jac(x):
            _raise_in_failure_region(x, optimum, counter)
            return sign * _poisson_score(x)

        if fun_type == "scalar":
            fun = om.mark.scalar(lambda x: contributions(x).sum())
            jac = om.mark.scalar(lambda x: contributions_jac(x).sum(axis=0))
        else:
            fun = om.mark.likelihood(contributions)
            jac = om.mark.likelihood(contributions_jac)
        optimal_value = sign * _poisson_loglike(optimum).sum()

    elif fun_type == "least_squares":
        optimum = GAUSSIAN_OPTIMUM

        @om.mark.least_squares
        def fun(x):
            _raise_in_failure_region(x, optimum, counter)
            return _gaussian_residuals(x)

        @om.mark.least_squares
        def jac(x):
            _raise_in_failure_region(x, optimum, counter)
            return _gaussian_residuals_jac(x)

        optimal_value = (_gaussian_residuals(optimum) ** 2).sum()

    return fun, jac, optimum, optimal_value


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
    ("likelihood", "bhhh", "minimize", False),
    ("likelihood", "bhhh", "maximize", False),
    # least-squares functions
    ("least_squares", "scipy_lbfgsb", "minimize", False),
    ("least_squares", "bhhh", "minimize", False),
    ("least_squares", "scipy_ls_trf", "minimize", False),
    ("least_squares", "scipy_ls_trf", "minimize", True),
    ("least_squares", "pounders", "minimize", False),
]


@pytest.mark.parametrize("fun_type, algorithm, direction, with_jac", CASES)
def test_optimizer_recovers_from_errors(fun_type, algorithm, direction, with_jac):
    counter = FailureCounter()
    fun, jac, optimum, optimal_value = _get_problem(fun_type, direction, counter)
    optimize = om.maximize if direction == "maximize" else om.minimize

    res = optimize(
        fun=fun,
        jac=jac if with_jac else None,
        params=START_PARAMS,
        algorithm=algorithm,
        error_handling="continue",
    )

    assert counter.n_failures > 0, "The failure region was never evaluated."
    aaae(res.params, optimum, decimal=4)
    # Penalty values are worse than the start value by at least 100, so an optimal
    # value equal to the true one means that no penalty value leaked into the result.
    assert np.isclose(res.fun, optimal_value, rtol=1e-8)
    assert res.history is not None
    history_values = [v for v in res.history.fun if v is not None]
    best = max(history_values) if direction == "maximize" else min(history_values)
    assert np.isclose(best, optimal_value, rtol=1e-8)


def test_least_squares_function_cannot_be_maximized():
    fun, *_ = _get_problem("least_squares", "minimize", FailureCounter())
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
    fun, *_ = _get_problem(fun_type, "minimize", FailureCounter())
    with pytest.raises(InvalidFunctionError):
        om.minimize(
            fun=fun,
            params=START_PARAMS,
            algorithm=algorithm,
            error_handling="continue",
        )
