from functools import partial

import numpy as np
import pytest
import statsmodels.api as sm
from estimagic.optimization.bhhh import minimize_bhhh
from numpy.testing import assert_array_almost_equal as aaae


def _cdf(x):
    return 1 / (1 + np.exp(-x))


def scoreobs(endog, exog, params):
    return (endog - _cdf(np.dot(exog, params)))[:, None] * exog


def loglikeobs(endog, exog, params):
    q = 2 * endog - 1
    return np.log(_cdf(q * np.dot(exog, params)))


@pytest.fixture()
def data():
    np.random.seed(12)

    num_observations = 5000
    x1 = np.random.multivariate_normal([0, 0], [[1, 0.75], [0.75, 1]], num_observations)
    x2 = np.random.multivariate_normal([1, 4], [[1, 0.75], [0.75, 1]], num_observations)

    endog = np.hstack((np.zeros(num_observations), np.ones(num_observations)))

    simulated_exog = np.vstack((x1, x2)).astype(np.float32)
    exog = simulated_exog
    intercept = np.ones((exog.shape[0], 1))
    exog = np.hstack((intercept, exog))

    return endog, exog


def test_logit_compare_bhhh_and_sm(data):
    endog, exog = data
    scoreobs_p = partial(scoreobs, endog, exog)
    loglikeobs_p = partial(loglikeobs, endog, exog)

    def logit_criterion(params, task="criterion_and_derivative"):
        """Logit criterion function.

        Args:
            params (np.ndarray): Parameter vector of shape (n_obs,).
            task (str): If task=="criterion", compute log-likelihood.
                If task=="derivative", compute jacobian.
                If task="criterion_and_derivative", compute both.

        Returns:
            np.ndarray or tuple: If task=="criterion" it returns the output of
                criterion, which is a 1d numpy array.
                If task=="derivative" it returns the first derivative of criterion,
                which is a numpy array.
                If task=="criterion_and_derivative" it returns both as a tuple.
        """
        res = ()

        if "criterion" in task:
            res += (-loglikeobs_p(params),)
        if "derivative" in task:
            res += (scoreobs_p(params),)

        if len(res) == 1:
            (res,) = res

        return res

    params = np.zeros(exog.shape[1])
    calculated = minimize_bhhh(criterion_and_derivative=logit_criterion, x=params)

    sm_res = sm.Logit(endog, exog).fit()
    expected = sm_res.params

    aaae(calculated, expected, decimal=4)
