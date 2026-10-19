"""Test the internal BHHH algorithm."""

import numpy as np
import pytest
import statsmodels.api as sm
from numpy.testing import assert_array_almost_equal as aaae
from scipy.stats import norm

from optimagic import mark, maximize, minimize
from optimagic.differentiation.derivatives import first_derivative
from optimagic.optimizers.bhhh import bhhh_internal
from optimagic.utilities import get_rng


def generate_test_data():
    rng = get_rng(seed=12)

    num_observations = 5000
    x1 = rng.multivariate_normal([0, 0], [[1, 0.75], [0.75, 1]], num_observations)
    x2 = rng.multivariate_normal([1, 4], [[1, 0.75], [0.75, 1]], num_observations)

    endog = np.hstack((np.zeros(num_observations), np.ones(num_observations)))

    simulated_exog = np.vstack((x1, x2)).astype(np.float32)
    exog = simulated_exog
    intercept = np.ones((exog.shape[0], 1))
    exog = np.hstack((intercept, exog))

    return endog, exog


ENDOG, EXOG = generate_test_data()


def _cdf_logit(x):
    return 1 / (1 + np.exp(-x))


def loglike_logit(x):
    """Return the logit loglikelihood contributions of shape (n_obs,)."""
    q = 2 * ENDOG - 1
    return np.log(_cdf_logit(q * (EXOG @ x)))


def score_logit(x):
    """Return the derivative of `loglike_logit` of shape (n_obs, n_params)."""
    return (ENDOG - _cdf_logit(EXOG @ x))[:, None] * EXOG


def loglike_probit(x):
    """Return the probit loglikelihood contributions of shape (n_obs,)."""
    q = 2 * ENDOG - 1
    return np.log(norm.cdf(q * (EXOG @ x)))


def score_probit(x):
    """Return the derivative of `loglike_probit` of shape (n_obs, n_params)."""
    q = 2 * ENDOG - 1
    linear_prediction = EXOG @ x
    derivative = q * norm.pdf(q * linear_prediction) / norm.cdf(q * linear_prediction)
    return derivative[:, None] * EXOG


MODELS = {
    "logit": (loglike_logit, score_logit, sm.Logit),
    "probit": (loglike_probit, score_probit, sm.Probit),
}


@pytest.fixture(params=list(MODELS), scope="module")
def model(request):
    loglike, score, statsmodels_model = MODELS[request.param]
    expected = statsmodels_model(ENDOG, EXOG).fit(disp=0).params
    return loglike, score, expected


def test_score_is_derivative_of_loglike(model):
    loglike, score, _ = model
    x = np.array([0.1, -0.2, 0.3])
    numerical = first_derivative(loglike, params=x).derivative
    aaae(score(x), numerical, decimal=5)


def test_bhhh_internal_minimizes_negative_loglike(model):
    loglike, score, expected = model

    def fun_and_jac(x):
        return -loglike(x), -score(x)

    res = bhhh_internal(fun_and_jac, x=np.zeros(3), gtol_abs=1e-8, maxiter=200)
    aaae(res.x, expected, decimal=4)


@pytest.mark.parametrize("direction", ["minimize", "maximize"])
@pytest.mark.parametrize("closed_form_derivative", [True, False])
def test_bhhh_external_interface(model, direction, closed_form_derivative):
    loglike, score, expected = model
    sign = 1.0 if direction == "maximize" else -1.0

    @mark.likelihood
    def fun(x):
        return sign * loglike(x)

    @mark.likelihood
    def fun_and_jac(x):
        return sign * loglike(x), sign * score(x)

    optimize = maximize if direction == "maximize" else minimize
    res = optimize(
        fun=fun,
        fun_and_jac=fun_and_jac if closed_form_derivative else None,
        params=np.zeros(3),
        algorithm="bhhh",
    )
    aaae(res.params, expected, decimal=4)
