import itertools

import numpy as np
import pytest
from estimagic.optimization.optimize import minimize
from estimagic.optimization.tranquilo.tranquilo import tranquilo
from estimagic.optimization.tranquilo.tranquilo import tranquilo_ls
from numpy.testing import assert_array_almost_equal as aaae


def _product(sample_filter, fitter, surrogate_model, sample_size):
    # is used to create products of test cases
    return list(itertools.product(sample_filter, fitter, surrogate_model, sample_size))


# ======================================================================================
# Scalar Tranquilo
# ======================================================================================

TEST_CASES = {
    "ols": {
        "sample_filter": ["discard_all", "keep_all"],
        "fitter": ["ols"],
        "surrogate_model": ["quadratic"],
        "sample_size": ["quadratic"],
    },
    "ols_keep_all": {
        "sample_filter": ["keep_all"],
        "fitter": ["ols"],
        "surrogate_model": ["quadratic"],
        "sample_size": ["powell"],
    },
    "pounders_discard_all": {
        "sample_filter": ["discard_all"],
        "fitter": ["powell"],
        "surrogate_model": ["quadratic"],
        "sample_size": ["quadratic"],
    },
    "pounders_keep_all": {
        "sample_filter": ["keep_all"],
        "fitter": ["powell"],
        "surrogate_model": ["quadratic"],
        "sample_size": ["linear", "powell", "quadratic"],
    },
    "ols_pounders_filtering": {
        "sample_filter": ["drop_pounders"],
        "fitter": ["ols"],
        "surrogate_model": ["quadratic"],
        "sample_size": ["powell", "quadratic"],
    },
    "pounders_filtering": {
        "sample_filter": ["drop_pounders"],
        "fitter": ["powell"],
        "surrogate_model": ["quadratic"],
        "sample_size": ["linear", "powell", "quadratic"],
    },
}

TEST_CASES = [_product(**kwargs) for kwargs in TEST_CASES.values()]
TEST_CASES = itertools.chain.from_iterable(TEST_CASES)


@pytest.mark.parametrize(
    "sample_filter, fitter, surrogate_model, sample_size", TEST_CASES
)
def test_internal_tranquilo_scalar_sphere_defaults(
    sample_filter, fitter, surrogate_model, sample_size
):
    res = tranquilo(
        criterion=lambda x: x @ x,
        x=np.arange(5),
        sample_filter=sample_filter,
        fitter=fitter,
        surrogate_model=surrogate_model,
        sample_size=sample_size,
    )
    aaae(res["solution_x"], np.zeros(5), decimal=5)


# ======================================================================================
# Imprecise options for scalar tranquilo
# ======================================================================================

TEST_CASES = {
    "ls_keep": {
        "sample_filter": ["keep_all"],
        "fitter": ["ols"],
        "surrogate_model": ["quadratic"],
        "sample_size": ["linear"],
    },
    "pounders_discard_all": {
        "sample_filter": ["discard_all"],
        "fitter": ["powell"],
        "surrogate_model": ["quadratic"],
        "sample_size": ["powell"],
    },
}

TEST_CASES = [_product(**kwargs) for kwargs in TEST_CASES.values()]
TEST_CASES = itertools.chain.from_iterable(TEST_CASES)


@pytest.mark.parametrize(
    "sample_filter, fitter, surrogate_model, sample_size", TEST_CASES
)
def test_internal_tranquilo_scalar_sphere_imprecise_defaults(
    sample_filter, fitter, surrogate_model, sample_size
):
    res = tranquilo(
        criterion=lambda x: x @ x,
        x=np.arange(5),
        sample_filter=sample_filter,
        fitter=fitter,
        surrogate_model=surrogate_model,
        sample_size=sample_size,
    )
    aaae(res["solution_x"], np.zeros(5), decimal=3)


# ======================================================================================
# External
# ======================================================================================


def test_external_tranquilo_scalar_sphere_defaults():
    res = minimize(
        criterion=lambda x: x @ x,
        params=np.arange(5),
        algorithm="tranquilo",
    )

    aaae(res.params, np.zeros(5), decimal=5)


# ======================================================================================
# Least-squares Tranquilo
# ======================================================================================


TEST_CASES = {
    "ols": {
        "sample_filter": ["keep_all", "discard_all"],
        "fitter": ["ols"],
        "surrogate_model": ["linear"],
        "sample_size": ["linear"],
    },
    "pounders_filtering": {
        "sample_filter": ["drop_pounders"],
        "fitter": ["ols"],
        "surrogate_model": ["linear"],
        "sample_size": ["linear"],
    },
}

TEST_CASES = [_product(**kwargs) for kwargs in TEST_CASES.values()]
TEST_CASES = itertools.chain.from_iterable(TEST_CASES)


@pytest.mark.parametrize(
    "sample_filter, fitter, surrogate_model, sample_size", TEST_CASES
)
def test_internal_tranquilo_ls_sphere_defaults(
    sample_filter, fitter, surrogate_model, sample_size
):
    res = tranquilo_ls(
        criterion=lambda x: x,
        x=np.arange(5),
        sample_filter=sample_filter,
        fitter=fitter,
        surrogate_model=surrogate_model,
        sample_size=sample_size,
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
