from itertools import product

import numpy as np
import pytest
from estimagic.optimization.optimize import minimize
from estimagic.optimization.tranquilo.tranquilo import tranquilo
from estimagic.optimization.tranquilo.tranquilo import tranquilo_ls
from numpy.testing import assert_array_almost_equal as aaae


# ======================================================================================
# Scalar Tranquilo
# ======================================================================================


_sample_filter = ["keep_all"]
_fitter = ["ols"]
_surrogate_model = ["quadratic"]
_sample_size = ["linear", "pounders", "quadratic"]
test_cases_ols = list(product(_sample_filter, _fitter, _surrogate_model, _sample_size))

_sample_filter = ["discard_all"]
_fitter = ["pounders"]
_surrogate_model = ["quadratic"]
_sample_size = ["pounders", "quadratic"]
test_cases_pounders = list(
    product(_sample_filter, _fitter, _surrogate_model, _sample_size)
)


TEST_CASES = test_cases_ols + test_cases_pounders


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
    aaae(res["solution_x"], np.zeros(5), decimal=3)


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

_sample_filter = ["keep_all", "discard_all"]
_fitter = ["ols"]
_surrogate_model = ["linear", "quadratic"]
_sample_size = ["linear", "pounders", "quadratic"]
test_cases_ols = list(product(_sample_filter, _fitter, _surrogate_model, _sample_size))

_sample_filter = ["discard_all"]
_fitter = ["pounders"]
_surrogate_model = ["linear", "quadratic"]
_sample_size = ["linear", "pounders", "quadratic"]
test_cases_pounders = list(
    product(_sample_filter, _fitter, _surrogate_model, _sample_size)
)


TEST_CASES = test_cases_ols + test_cases_pounders


@pytest.mark.parametrize(
    "sample_filter, fitter, surrogate_model, sample_size", TEST_CASES
)
def test_internal_tranquilo_ls_sphere_defaults_ls(
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


def test_internal_tranquilo_ls_sphere_defaults():
    res = tranquilo_ls(
        criterion=lambda x: x,
        x=np.arange(5),
    )
    aaae(res["solution_x"], np.zeros(5), decimal=5)


def test_external_tranquilo_ls_sphere_defaults():
    res = minimize(
        criterion=lambda x: x,
        params=np.arange(5),
        algorithm="tranquilo_ls",
    )

    aaae(res.params, np.zeros(5), decimal=5)
