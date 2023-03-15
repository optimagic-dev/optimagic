import itertools

import numpy as np
import pytest
from estimagic.optimization.optimize import minimize
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
        x=np.arange(4),
        sample_filter=sample_filter,
        fitter=fitter,
        surrogate_model=surrogate_model,
        sample_size=sample_size,
    )
    aaae(res["solution_x"], np.zeros(4), decimal=4)


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
        x=np.arange(4),
        sample_filter=sample_filter,
        fitter=fitter,
        surrogate_model=surrogate_model,
        sample_size=sample_size,
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
    assert got == "quadratic"


@pytest.mark.parametrize("functype", ["least_squares", "likelihood"])
def test_process_surrogate_model_none_not_scalar(functype):
    got = _process_surrogate_model(None, functype=functype)
    assert got == "linear"


@pytest.mark.parametrize("model_type", ("linear", "quadratic"))
def test_process_surrogate_model_info(model_type):
    got = _process_surrogate_model(model_type, functype="whatever")
    assert got == model_type


def test_process_surrogate_model_str_linear():
    got = _process_surrogate_model("linear", functype="scalar")
    assert got == "linear"


def test_process_surrogate_model_str_quadratic():
    got = _process_surrogate_model("quadratic", functype="likelihood")
    assert got == "quadratic"


def test_process_surrogate_model_str_invalid():
    with pytest.raises(ValueError):
        _process_surrogate_model("whatever", None)


@pytest.mark.parametrize("functype", ["scalar", "least_squares"])
def test_process_surrogate_model_invalid(functype):
    surrogate_model = np.linalg.lstsq
    with pytest.raises(TypeError):
        _process_surrogate_model(surrogate_model, functype=functype)


@pytest.mark.parametrize("model_type", ("linear", "quadratic"))
def test_process_sample_size_none_linear(model_type):
    x = np.ones((3, 2))
    got = _process_sample_size(
        None, model_type=model_type, x=x, sample_size_factor=None
    )
    if model_type == "quadratic":
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
