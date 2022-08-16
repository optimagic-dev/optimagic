"""Test the internal BHHH algorithm."""
from functools import partial

import numpy as np
import pytest
import statsmodels.api as sm
from estimagic.optimization.bhhh import bhhh_internal
from estimagic.utilities import get_rng
from numpy.testing import assert_array_almost_equal as aaae
from scipy.stats import norm


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


def _cdf_logit(x):
    return 1 / (1 + np.exp(-x))


def get_loglikelihood_logit(endog, exog, x):
    q = 2 * endog - 1
    linear_prediction = np.dot(exog, x)

    return np.log(_cdf_logit(q * linear_prediction))


def get_score_logit(endog, exog, x):
    linear_prediction = np.dot(exog, x)

    return (endog - _cdf_logit(linear_prediction))[:, None] * exog


def get_loglikelihood_probit(endog, exog, x):
    q = 2 * endog - 1
    linear_prediction = np.dot(exog, x[: exog.shape[1]])

    return np.log(norm.cdf(q * linear_prediction))


def get_score_probit(endog, exog, x):
    q = 2 * endog - 1
    linear_prediction = np.dot(exog, x[: exog.shape[1]])

    derivative_loglikelihood = (
        q * norm.pdf(q * linear_prediction) / norm.cdf(q * linear_prediction)
    )

    return derivative_loglikelihood[:, None] * exog


def criterion_and_derivative_logit(x):
    """Return Logit criterion and derivative.

    Args:
        x (np.ndarray): Parameter vector of shape (n_obs,).

    Returns:
        tuple: first entry is the criterion, second entry is the score

    """
    endog, exog = generate_test_data()
    score = partial(get_score_logit, endog, exog)
    loglike = partial(get_loglikelihood_logit, endog, exog)

    return -loglike(x), score(x)


def criterion_and_derivative_probit(x):
    """Return Probit criterion and derivative.

    Args:
        x (np.ndarray): Parameter vector of shape (n_obs,).

    Returns:
        tuple: first entry is the criterion, second entry is the score
    """
    endog, exog = generate_test_data()

    score = partial(get_score_probit, endog, exog)
    loglike = partial(get_loglikelihood_probit, endog, exog)

    return -loglike(x), score(x)


# =====================================================================================
# Fixtures
# =====================================================================================


@pytest.fixture
def result_logit_unbounded():
    endog, exog = generate_test_data()
    result_unbounded = sm.Logit(endog, exog).fit()

    return result_unbounded.params


@pytest.fixture
def result_probit_unbounded():
    endog, exog = generate_test_data()
    result_unbounded = sm.Probit(endog, exog).fit()

    return result_unbounded.params


@pytest.fixture
def result_logit_bounded():
    return np.array([-5.0, -3.93455447, 6.49171767])


@pytest.fixture
def result_probit_bounded():
    return np.array([-5.0, -2.56431715, 4.18468707])


# =====================================================================================
# Tests
# =====================================================================================
TEST_CASES = [
    (
        criterion_and_derivative_logit,
        np.array([-np.inf, -np.inf, -np.inf]),
        np.array([np.inf, np.inf, np.inf]),
        "result_logit_unbounded",
    ),
    (
        criterion_and_derivative_probit,
        np.array([-np.inf, -np.inf, -np.inf]),
        np.array([np.inf, np.inf, np.inf]),
        "result_probit_unbounded",
    ),
    (
        criterion_and_derivative_logit,
        np.array([-5, -10, -10]),
        np.array([10, 10, 10]),
        "result_logit_bounded",
    ),
    (
        criterion_and_derivative_probit,
        np.array([-5, -10, -10]),
        np.array([10, 10, 10]),
        "result_probit_bounded",
    ),
]


@pytest.mark.parametrize(
    "criterion_and_derivative, lower_bounds, upper_bounds, result_statsmodels",
    TEST_CASES,
)
def test_maximum_likelihood(
    criterion_and_derivative, lower_bounds, upper_bounds, result_statsmodels, request
):
    params_expected = request.getfixturevalue(result_statsmodels)

    x0 = np.zeros(3)

    result_bhhh = bhhh_internal(
        criterion_and_derivative,
        x=x0,
        lower_bounds=lower_bounds,
        upper_bounds=upper_bounds,
        convergence_absolute_gradient_tolerance=1e-8,
        stopping_max_iterations=200,
    )

    aaae(result_bhhh["solution_x"], params_expected, decimal=4)
