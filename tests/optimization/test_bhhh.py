"""Test the internal BHHH algorithm."""
from functools import partial

import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm
from estimagic.config import TEST_FIXTURES_DIR
from estimagic.optimization.bhhh import bhhh_box_constrained
from estimagic.optimization.bhhh import bhhh_unconstrained
from estimagic.utilities import get_rng
from numpy.testing import assert_array_almost_equal as aaae
from scipy.stats import norm


def generate_test_data():
    rng = get_rng(seed=12)

    num_observations = 5_000
    x1 = rng.multivariate_normal([0, 0], [[1, 0.75], [0.75, 1]], num_observations)
    x2 = rng.multivariate_normal([1, 4], [[1, 0.75], [0.75, 1]], num_observations)

    endog = np.hstack((np.zeros(num_observations), np.ones(num_observations)))

    simulated_exog = np.vstack((x1, x2)).astype(np.float32)
    exog = simulated_exog
    intercept = np.ones((exog.shape[0], 1))
    exog = np.hstack((intercept, exog))

    return endog, exog


def ibm_test_data():
    df = pd.read_csv(TEST_FIXTURES_DIR / "telco_churn_clean.csv")

    exog = df.drop(columns="Churn").to_numpy()
    intercept = np.ones((exog.shape[0], 1))
    exog = np.hstack((intercept, exog))
    endog = df[["Churn"]].to_numpy()
    endog = endog.reshape(
        len(endog),
    )

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


def criterion_and_derivative_logit(x, task="criterion_and_derivative"):
    """Return Logit criterion and derivative.

    Args:
        x (np.ndarray): Parameter vector of shape (n_obs,).
        task (str): If task=="criterion", compute log-likelihood.
            If task=="derivative", compute derivative.
            If task="criterion_and_derivative", compute both.

    Returns:
        (np.ndarray or tuple): If task=="criterion" it returns the output of
            criterion, which is a 1d numpy array.
            If task=="derivative" it returns the first derivative of criterion,
            which is 2d numpy array.
            If task=="criterion_and_derivative" it returns both as a tuple.
    """
    endog, exog = generate_test_data()
    result = ()

    if "criterion" in task:
        loglike = partial(get_loglikelihood_logit, endog, exog)
        result += (-loglike(x),)
    if "derivative" in task:
        score = partial(get_score_logit, endog, exog)
        result += (score(x),)

    if len(result) == 1:
        (result,) = result

    return result


def criterion_and_derivative_logit_ibm(x, task="criterion_and_derivative"):
    """Return Logit criterion and derivative.

    Args:
        x (np.ndarray): Parameter vector of shape (n_obs,).
        task (str): If task=="criterion", compute log-likelihood.
            If task=="derivative", compute derivative.
            If task="criterion_and_derivative", compute both.

    Returns:
        (np.ndarray or tuple): If task=="criterion" it returns the output of
            criterion, which is a 1d numpy array.
            If task=="derivative" it returns the first derivative of criterion,
            which is 2d numpy array.
            If task=="criterion_and_derivative" it returns both as a tuple.
    """
    endog, exog = ibm_test_data()
    result = ()

    if "criterion" in task:
        loglike = partial(get_loglikelihood_logit, endog, exog)
        result += (-loglike(x),)
    if "derivative" in task:
        score = partial(get_score_logit, endog, exog)
        result += (score(x),)

    if len(result) == 1:
        (result,) = result

    return result


def criterion_and_derivative_probit(x, task="criterion_and_derivative"):
    """Return Probit criterion and derivative.

    Args:
        x (np.ndarray): Parameter vector of shape (n_obs,).
        task (str): If task=="criterion", compute log-likelihood.
            If task=="derivative", compute derivative.
            If task="criterion_and_derivative", compute both.

    Returns:
        (np.ndarray or tuple): If task=="criterion" it returns the output of
            criterion, which is a 1d numpy array.
            If task=="derivative" it returns the first derivative of criterion,
            which is 2d numpy array.
            If task=="criterion_and_derivative" it returns both as a tuple.
    """
    endog, exog = generate_test_data()
    result = ()

    if "criterion" in task:
        loglike = partial(get_loglikelihood_probit, endog, exog)
        result += (-loglike(x),)
    if "derivative" in task:
        score = partial(get_score_probit, endog, exog)
        result += (score(x),)

    if len(result) == 1:
        (result,) = result

    return result


def criterion_and_derivative_probit_ibm(x, task="criterion_and_derivative"):
    """Return Probit criterion and derivative.

    Args:
        x (np.ndarray): Parameter vector of shape (n_obs,).
        task (str): If task=="criterion", compute log-likelihood.
            If task=="derivative", compute derivative.
            If task="criterion_and_derivative", compute both.

    Returns:
        (np.ndarray or tuple): If task=="criterion" it returns the output of
            criterion, which is a 1d numpy array.
            If task=="derivative" it returns the first derivative of criterion,
            which is 2d numpy array.
            If task=="criterion_and_derivative" it returns both as a tuple.
    """
    endog, exog = ibm_test_data()
    result = ()

    if "criterion" in task:
        loglike = partial(get_loglikelihood_probit, endog, exog)
        result += (-loglike(x),)
    if "derivative" in task:
        score = partial(get_score_probit, endog, exog)
        result += (score(x),)

    if len(result) == 1:
        (result,) = result

    return result


# =====================================================================================
# Fixtures
# =====================================================================================


@pytest.fixture
def result_logit_unbounded():
    endog, exog = generate_test_data()
    result_unbounded = sm.Logit(endog, exog).fit(disp=True)

    return result_unbounded


@pytest.fixture
def result_probit_unbounded():
    endog, exog = generate_test_data()
    result_unbounded = sm.Probit(endog, exog).fit(disp=True)

    return result_unbounded


@pytest.fixture
def result_logit_bounded():
    endog, exog = generate_test_data()
    result_bounded = sm.Logit(endog, exog).fit(
        method="lbfgs", bounds=((-5, np.inf), (-10, 10), (-10, 10)), disp=False
    )

    return result_bounded


@pytest.fixture
def result_probit_bounded():
    endog, exog = generate_test_data()
    result_bounded = sm.Logit(endog, exog).fit(
        method="lbfgs", bounds=((-5, np.inf), (-10, 10), (-10, 10)), disp=False
    )

    return result_bounded


@pytest.fixture
def result_logit_ibm():
    endog, exog = ibm_test_data()
    result_unbounded = sm.Logit(endog, exog).fit(disp=False)

    return result_unbounded


@pytest.fixture
def result_probit_ibm():
    endog, exog = ibm_test_data()
    result_unbounded = sm.Probit(endog, exog).fit(disp=False)

    return result_unbounded


# =====================================================================================
# Tests
# =====================================================================================

TEST_CASES_UNBOUNDED = [
    (
        criterion_and_derivative_logit,
        np.zeros(3),
        -np.ones(3) * np.inf,
        np.ones(3) * np.inf,
        "result_logit_unbounded",
    ),
    (
        criterion_and_derivative_probit,
        np.zeros(3),
        -np.ones(3) * np.inf,
        np.ones(3) * np.inf,
        "result_probit_unbounded",
    ),
    (
        criterion_and_derivative_logit_ibm,
        np.zeros(9),
        -np.ones(9) * np.inf,
        np.ones(9) * np.inf,
        "result_logit_ibm",
    ),
    (
        criterion_and_derivative_probit_ibm,
        np.zeros(9),
        -np.ones(9) * np.inf,
        np.ones(9) * np.inf,
        "result_probit_ibm",
    ),
]


@pytest.mark.parametrize(
    "criterion_and_derivative, x0, lower_bounds, upper_bounds, expected",
    TEST_CASES_UNBOUNDED,
)
def test_maximum_likelihood(
    criterion_and_derivative,
    x0,
    lower_bounds,
    upper_bounds,
    expected,
    request,
):
    params_expected = request.getfixturevalue(expected)

    result_bhhh = bhhh_unconstrained(
        criterion_and_derivative,
        x=x0,
        lower_bounds=lower_bounds,
        upper_bounds=upper_bounds,
        convergence_absolute_gradient_tolerance=1e-8,
        convergence_relative_gradient_tolerance=1e-8,
        stopping_max_iterations=200,
    )

    aaae(result_bhhh["solution_x"], params_expected.params, decimal=4)


TEST_CASES_BOUNDED = [
    (
        criterion_and_derivative_logit,
        np.zeros(3),
        np.array([-5, -100, -100]),
        np.array([100, 100, 100]),
        "result_logit_bounded",
        4,
    ),
    (
        criterion_and_derivative_probit,
        np.zeros(3),
        np.array([-5, -np.inf, -np.inf]),
        np.array([10, np.inf, np.inf]),
        "result_probit_bounded",
        0,
    ),
]


@pytest.mark.parametrize(
    "criterion_and_derivative, x0, lower_bounds, upper_bounds, expected, digits",
    TEST_CASES_BOUNDED,
)
def test_maximum_likelihood_bounded(
    criterion_and_derivative,
    x0,
    lower_bounds,
    upper_bounds,
    expected,
    decimals,
    request,
):
    params_expected = request.getfixturevalue(expected)

    result_bhhh = bhhh_box_constrained(
        criterion_and_derivative,
        x=x0,
        lower_bounds=lower_bounds,
        upper_bounds=upper_bounds,
        convergence_absolute_gradient_tolerance=1e-4,
        stopping_max_iterations=200,
    )

    aaae(result_bhhh["solution_x"], params_expected.params, decimal=decimals)
