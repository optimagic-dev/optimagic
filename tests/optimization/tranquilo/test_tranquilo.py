import itertools

import numpy as np
import pytest
from estimagic.optimization.optimize import minimize
from estimagic.optimization.tranquilo.models import ModelInfo
from estimagic.optimization.tranquilo.tranquilo import (
    _process_sample_size,
    _process_surrogate_model,
    tranquilo,
    tranquilo_ls,
)
from numpy.testing import assert_array_almost_equal as aaae

# ======================================================================================
# Test tranquilo end-to-end
# ======================================================================================


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
        "sample_size": ["powell", "quadratic"],
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
        "sample_size": ["powell"],
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
    aaae(res["solution_x"], np.zeros(5), decimal=4)


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

    aaae(res.params, np.zeros(5), decimal=4)


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


# ======================================================================================
# Test input processing functions
# ======================================================================================


def test_process_surrogate_model_none_scalar():
    got = _process_surrogate_model(None, functype="scalar")
    assert got.has_interactions is True
    assert got.has_squares is True


@pytest.mark.parametrize("functype", ["least_squares", "likelihood"])
def test_process_surrogate_model_none_not_scalar(functype):
    got = _process_surrogate_model(None, functype=functype)
    assert got.has_interactions is False
    assert got.has_squares is False


@pytest.mark.parametrize("has_interactions", [True, False])
@pytest.mark.parametrize("has_squares", [True, False])
def test_process_surrogate_model_info(has_interactions, has_squares):
    model_info = ModelInfo(has_squares=has_squares, has_interactions=has_interactions)
    got = _process_surrogate_model(model_info, functype="whatever")
    assert got == model_info


def test_process_surrogate_model_str_linear():
    got = _process_surrogate_model("linear", functype="scalar")
    assert got.has_interactions is False
    assert got.has_squares is False


def test_process_surrogate_model_str_diagonal():
    got = _process_surrogate_model("diagonal", functype="least_squares")
    assert got.has_interactions is False
    assert got.has_squares is True


def test_process_surrogate_model_str_quadratic():
    got = _process_surrogate_model("quadratic", functype="likelihood")
    assert got.has_interactions is True
    assert got.has_squares is True


def test_process_surrogate_model_str_invalid():
    with pytest.raises(ValueError):
        _process_surrogate_model("whatever", None)


@pytest.mark.parametrize("functype", ["scalar", "least_squares"])
def test_process_surrogate_model_invalid(functype):
    surrogate_model = np.linalg.lstsq
    with pytest.raises(TypeError):
        _process_surrogate_model(surrogate_model, functype=functype)


@pytest.mark.parametrize("has_interactions", [True, False])
@pytest.mark.parametrize("has_squares", [True, False])
def test_process_sample_size_none_linear(has_interactions, has_squares):
    model_info = ModelInfo(has_interactions=has_interactions, has_squares=has_squares)
    x = np.ones((3, 2))
    got = _process_sample_size(
        None, model_info=model_info, x=x, sample_size_factor=None
    )
    if has_interactions or has_squares:
        assert got == 7
    else:
        assert got == 4


STR_TEST_CASES = [  # assume len(x) = 3
    # (user_sample_size, expected)  # noqa: ERA001
    ("linear", 3 + 1),
    ("n+1", 3 + 1),
    ("n + 1", 3 + 1),
    ("powell", 2 * 3 + 1),
    ("2n+1", 2 * 3 + 1),
    ("2 n + 1", 2 * 3 + 1),
    ("2*n + 1", 2 * 3 + 1),
    ("quadratic", 6 + 3 + 1),
]


@pytest.mark.parametrize("user_sample_size, expected", STR_TEST_CASES)
def test_process_sample_size_str(user_sample_size, expected):
    x = np.ones((3, 2))
    got = _process_sample_size(user_sample_size, None, x=x, sample_size_factor=None)
    assert got == expected


def test_process_sample_size_str_invalid():
    with pytest.raises(ValueError):
        _process_sample_size("n**2", None, None, None)


@pytest.mark.parametrize("user_sample_size", [1, 10, -100, 10.5])
def test_process_sample_size_number(user_sample_size):
    got = _process_sample_size(user_sample_size, None, None, None)
    assert got == int(user_sample_size)


def test_process_sample_size_invalid():
    x = np.ones((3, 2))
    with pytest.raises(TypeError):
        _process_sample_size(np.zeros_like(x), None, x, None)
