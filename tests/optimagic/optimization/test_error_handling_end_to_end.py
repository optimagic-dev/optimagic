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
optimum.

"""

from dataclasses import dataclass, field

import numpy as np
import pytest
import scipy.optimize
from numpy.testing import assert_array_almost_equal as aaae
from scipy.special import gammaln

import optimagic as om
from optimagic.utilities import get_rng

START_PARAMS = np.full(2, -1.0)
FAILURE_MARGIN = 0.01

BHHH_XFAIL = pytest.mark.xfail(
    strict=True,
    reason="bhhh steps uphill and accepts worse points; fixed in #719.",
)


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


def _poisson_functions(direction, counter):
    """Return the loglikelihood contributions and their derivative of shape (n_obs, 2).

    For minimization, both are multiplied by -1.

    """
    sign = 1.0 if direction == "maximize" else -1.0

    def contributions(x):
        _raise_in_failure_region(x, POISSON_OPTIMUM, counter)
        return sign * _poisson_loglike(x)

    def contributions_jac(x):
        _raise_in_failure_region(x, POISSON_OPTIMUM, counter)
        return sign * _poisson_score(x)

    return contributions, contributions_jac


def _scalar_problem(direction, counter):
    contributions, contributions_jac = _poisson_functions(direction, counter)
    fun = om.mark.scalar(lambda x: contributions(x).sum())
    jac = om.mark.scalar(lambda x: contributions_jac(x).sum(axis=0))
    return fun, jac


def _likelihood_problem(direction, counter):
    contributions, contributions_jac = _poisson_functions(direction, counter)
    return om.mark.likelihood(contributions), om.mark.likelihood(contributions_jac)


def _least_squares_problem(counter):
    @om.mark.least_squares
    def fun(x):
        _raise_in_failure_region(x, GAUSSIAN_OPTIMUM, counter)
        return _gaussian_residuals(x)

    @om.mark.least_squares
    def jac(x):
        _raise_in_failure_region(x, GAUSSIAN_OPTIMUM, counter)
        return _gaussian_residuals_jac(x)

    return fun, jac


def _assert_poisson_recovered(res, counter):
    sign = 1.0 if res.direction == "maximize" else -1.0
    optimal_value = sign * _poisson_loglike(POISSON_OPTIMUM).sum()
    _assert_recovered(res, counter, POISSON_OPTIMUM, optimal_value)


def _assert_gaussian_recovered(res, counter):
    optimal_value = (_gaussian_residuals(GAUSSIAN_OPTIMUM) ** 2).sum()
    _assert_recovered(res, counter, GAUSSIAN_OPTIMUM, optimal_value)


def _assert_recovered(res, counter, optimum, optimal_value):
    assert counter.n_failures > 0, "The failure region was never evaluated."
    aaae(res.params, optimum, decimal=4)
    # Penalty values are worse than the start value by at least 100, so an optimal
    # value equal to the true one means that no penalty value leaked into the result.
    assert np.isclose(res.fun, optimal_value, rtol=1e-8)
    assert res.history is not None
    history_values = [v for v in res.history.fun if v is not None]
    best = max(history_values) if res.direction == "maximize" else min(history_values)
    assert np.isclose(best, optimal_value, rtol=1e-8)


# ======================================================================================
# scalar functions
# ======================================================================================


def test_scalar_lbfgsb_minimize():
    counter = FailureCounter()
    fun, _ = _scalar_problem("minimize", counter)
    res = om.minimize(
        fun, START_PARAMS, algorithm="scipy_lbfgsb", error_handling="continue"
    )
    _assert_poisson_recovered(res, counter)


def test_scalar_lbfgsb_maximize():
    counter = FailureCounter()
    fun, _ = _scalar_problem("maximize", counter)
    res = om.maximize(
        fun, START_PARAMS, algorithm="scipy_lbfgsb", error_handling="continue"
    )
    _assert_poisson_recovered(res, counter)


def test_scalar_lbfgsb_minimize_with_jac():
    counter = FailureCounter()
    fun, jac = _scalar_problem("minimize", counter)
    res = om.minimize(
        fun, START_PARAMS, algorithm="scipy_lbfgsb", jac=jac, error_handling="continue"
    )
    _assert_poisson_recovered(res, counter)


def test_scalar_lbfgsb_maximize_with_jac():
    counter = FailureCounter()
    fun, jac = _scalar_problem("maximize", counter)
    res = om.maximize(
        fun, START_PARAMS, algorithm="scipy_lbfgsb", jac=jac, error_handling="continue"
    )
    _assert_poisson_recovered(res, counter)


def test_scalar_neldermead_minimize():
    counter = FailureCounter()
    fun, _ = _scalar_problem("minimize", counter)
    res = om.minimize(
        fun, START_PARAMS, algorithm="scipy_neldermead", error_handling="continue"
    )
    _assert_poisson_recovered(res, counter)


def test_scalar_neldermead_maximize():
    counter = FailureCounter()
    fun, _ = _scalar_problem("maximize", counter)
    res = om.maximize(
        fun, START_PARAMS, algorithm="scipy_neldermead", error_handling="continue"
    )
    _assert_poisson_recovered(res, counter)


# ======================================================================================
# likelihood functions
# ======================================================================================


def test_likelihood_lbfgsb_minimize():
    counter = FailureCounter()
    fun, _ = _likelihood_problem("minimize", counter)
    res = om.minimize(
        fun, START_PARAMS, algorithm="scipy_lbfgsb", error_handling="continue"
    )
    _assert_poisson_recovered(res, counter)


def test_likelihood_lbfgsb_maximize():
    counter = FailureCounter()
    fun, _ = _likelihood_problem("maximize", counter)
    res = om.maximize(
        fun, START_PARAMS, algorithm="scipy_lbfgsb", error_handling="continue"
    )
    _assert_poisson_recovered(res, counter)


@BHHH_XFAIL
def test_likelihood_bhhh_minimize():
    counter = FailureCounter()
    fun, _ = _likelihood_problem("minimize", counter)
    res = om.minimize(fun, START_PARAMS, algorithm="bhhh", error_handling="continue")
    _assert_poisson_recovered(res, counter)


@BHHH_XFAIL
def test_likelihood_bhhh_maximize():
    counter = FailureCounter()
    fun, _ = _likelihood_problem("maximize", counter)
    res = om.maximize(fun, START_PARAMS, algorithm="bhhh", error_handling="continue")
    _assert_poisson_recovered(res, counter)


# ======================================================================================
# least-squares functions
# ======================================================================================


def test_least_squares_lbfgsb():
    counter = FailureCounter()
    fun, _ = _least_squares_problem(counter)
    res = om.minimize(
        fun, START_PARAMS, algorithm="scipy_lbfgsb", error_handling="continue"
    )
    _assert_gaussian_recovered(res, counter)


@BHHH_XFAIL
def test_least_squares_bhhh():
    counter = FailureCounter()
    fun, _ = _least_squares_problem(counter)
    res = om.minimize(fun, START_PARAMS, algorithm="bhhh", error_handling="continue")
    _assert_gaussian_recovered(res, counter)


def test_least_squares_ls_trf():
    counter = FailureCounter()
    fun, _ = _least_squares_problem(counter)
    res = om.minimize(
        fun, START_PARAMS, algorithm="scipy_ls_trf", error_handling="continue"
    )
    _assert_gaussian_recovered(res, counter)


def test_least_squares_ls_trf_with_jac():
    counter = FailureCounter()
    fun, jac = _least_squares_problem(counter)
    res = om.minimize(
        fun, START_PARAMS, algorithm="scipy_ls_trf", jac=jac, error_handling="continue"
    )
    _assert_gaussian_recovered(res, counter)


def test_least_squares_pounders():
    counter = FailureCounter()
    fun, _ = _least_squares_problem(counter)
    res = om.minimize(
        fun, START_PARAMS, algorithm="pounders", error_handling="continue"
    )
    _assert_gaussian_recovered(res, counter)
