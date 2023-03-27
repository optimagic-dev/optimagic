import itertools

import numpy as np
import pytest
from estimagic.optimization.optimize import minimize
from estimagic.optimization.tranquilo.tranquilo import (
    tranquilo,
    tranquilo_ls,
)
from numpy.testing import assert_array_almost_equal as aaae

# ======================================================================================
# Test tranquilo end-to-end
# ======================================================================================


def _product(sample_filter, model_fitter, model_type):
    # is used to create products of test cases
    return list(itertools.product(sample_filter, model_fitter, model_type))


# ======================================================================================
# Scalar Tranquilo
# ======================================================================================

TEST_CASES = {
    "ols": {
        "sample_filter": ["discard_all", "keep_all"],
        "model_fitter": ["ols"],
        "model_type": ["quadratic"],
    },
    "ols_keep_all": {
        "sample_filter": ["keep_all"],
        "model_fitter": ["ols"],
        "model_type": ["quadratic"],
    },
    "pounders_discard_all": {
        "sample_filter": ["discard_all"],
        "model_fitter": ["powell"],
        "model_type": ["quadratic"],
    },
    "pounders_keep_all": {
        "sample_filter": ["keep_all"],
        "model_fitter": ["powell"],
        "model_type": ["quadratic"],
    },
}

TEST_CASES = [_product(**kwargs) for kwargs in TEST_CASES.values()]
TEST_CASES = itertools.chain.from_iterable(TEST_CASES)


@pytest.mark.parametrize("sample_filter, model_fitter, model_type", TEST_CASES)
def test_internal_tranquilo_scalar_sphere_defaults(
    sample_filter,
    model_fitter,
    model_type,
):
    res = tranquilo(
        criterion=lambda x: x @ x,
        x=np.arange(4),
        sample_filter=sample_filter,
        model_fitter=model_fitter,
        model_type=model_type,
    )
    aaae(res["solution_x"], np.zeros(4), decimal=4)


# ======================================================================================
# Imprecise options for scalar tranquilo
# ======================================================================================

TEST_CASES = {
    "ls_keep": {
        "sample_filter": ["keep_all"],
        "model_fitter": ["ols"],
        "model_type": ["quadratic"],
    },
    "pounders_discard_all": {
        "sample_filter": ["discard_all"],
        "model_fitter": ["powell"],
        "model_type": ["quadratic"],
    },
}

TEST_CASES = [_product(**kwargs) for kwargs in TEST_CASES.values()]
TEST_CASES = itertools.chain.from_iterable(TEST_CASES)


@pytest.mark.parametrize("sample_filter, model_fitter, model_type", TEST_CASES)
def test_internal_tranquilo_scalar_sphere_imprecise_defaults(
    sample_filter,
    model_fitter,
    model_type,
):
    res = tranquilo(
        criterion=lambda x: x @ x,
        x=np.arange(4),
        sample_filter=sample_filter,
        model_fitter=model_fitter,
        model_type=model_type,
    )
    aaae(res["solution_x"], np.zeros(4), decimal=3)


# ======================================================================================
# External
# ======================================================================================


def test_external_tranquilo_scalar_sphere_defaults():
    res = minimize(
        criterion=lambda x: x @ x,
        params=np.arange(4),
        algorithm="tranquilo",
    )

    aaae(res.params, np.zeros(4), decimal=4)


# ======================================================================================
# Least-squares Tranquilo
# ======================================================================================


TEST_CASES = {
    "ols": {
        "sample_filter": ["keep_all", "discard_all"],
        "model_fitter": ["ols"],
        "model_type": ["linear"],
    },
    "pounders_filtering": {
        "sample_filter": ["drop_pounders"],
        "model_fitter": ["ols"],
        "model_type": ["linear"],
    },
}

TEST_CASES = [_product(**kwargs) for kwargs in TEST_CASES.values()]
TEST_CASES = itertools.chain.from_iterable(TEST_CASES)


@pytest.mark.parametrize("sample_filter, model_fitter, model_type", TEST_CASES)
def test_internal_tranquilo_ls_sphere_defaults(
    sample_filter,
    model_fitter,
    model_type,
):
    res = tranquilo_ls(
        criterion=lambda x: x,
        x=np.arange(5),
        sample_filter=sample_filter,
        model_fitter=model_fitter,
        model_type=model_type,
    )
    aaae(res["solution_x"], np.zeros(5), decimal=5)


# ======================================================================================
# External
# ======================================================================================


def test_external_tranquilo_ls_sphere_defaults():
    res = minimize(
        criterion=lambda x: x,
        params=np.arange(5),
        algorithm="tranquilo_ls",
    )

    aaae(res.params, np.zeros(5), decimal=5)


# ======================================================================================
# Noisy case
# ======================================================================================


@pytest.mark.parametrize("algo", ["tranquilo", "tranquilo_ls"])
def test_tranquilo_with_noise_handling_and_deterministic_function(algo):
    def _f(x):
        return {"root_contributions": x, "value": x @ x}

    res = minimize(
        criterion=_f,
        params=np.arange(5),
        algorithm=algo,
        algo_options={"noisy": True},
    )

    aaae(res.params, np.zeros(5), decimal=3)


@pytest.mark.slow()
def test_tranquilo_ls_with_noise_handling_and_noisy_function():
    rng = np.random.default_rng(123)

    def _f(x):
        x_n = x + rng.normal(0, 0.05, size=x.shape)
        return {"root_contributions": x_n, "value": x_n @ x_n}

    res = minimize(
        criterion=_f,
        params=np.ones(3),
        algorithm="tranquilo",
        algo_options={"noisy": True, "n_evals_per_point": 10},
    )

    aaae(res.params, np.zeros(3), decimal=1)


# ======================================================================================
# Bounded case
# ======================================================================================


def sum_of_squares(x):
    contribs = x**2
    return {"value": contribs.sum(), "contributions": contribs, "root_contributions": x}


@pytest.mark.parametrize("algorithm", ["tranquilo", "tranquilo_ls"])
def test_tranquilo_with_binding_bounds(algorithm):
    res = minimize(
        criterion=sum_of_squares,
        params=np.array([3, 2, -3]),
        lower_bounds=np.array([1, -np.inf, -np.inf]),
        upper_bounds=np.array([np.inf, np.inf, -1]),
        algorithm=algorithm,
        collect_history=True,
        skip_checks=True,
    )
    assert res.success in [True, None]
    aaae(res.params, np.array([1, 0, -1]), decimal=3)
