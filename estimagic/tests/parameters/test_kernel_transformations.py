from functools import partial
from itertools import product

import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal as aaae

import estimagic.parameters.kernel_transformations as kt
from estimagic.differentiation.derivatives import first_derivative
from estimagic.parameters.kernel_transformations import cov_matrix_to_sdcorr_params

to_test = list(product(range(10, 30, 5), [1234, 5471]))


def get_internal_cholesky(dim, seed=0):
    """Return random internal cholesky values given dimension."""
    np.random.seed(seed)
    chol = np.tril(np.random.randn(dim, dim))
    internal = chol[np.tril_indices(len(chol))]
    return internal


def get_external_covariance(dim, seed=0):
    """Return random external covariance values given dimension."""
    np.random.seed(seed)
    data = np.random.randn(dim, 1000)
    cov = np.cov(data)
    external = cov[np.tril_indices(dim)]
    return external


def get_internal_probability(dim, seed=0):
    """Return random internal positive values given dimension."""
    np.random.seed(seed)
    internal = np.random.uniform(size=dim)
    return internal


def get_external_probability(dim, seed=0):
    """Return random internal positive values that sum to one."""
    internal = get_internal_probability(dim, seed)
    external = internal / internal.sum()
    return external


def get_external_sdcorr(dim, seed=0):
    """Return random external sdcorr values given dimension."""
    np.random.seed(seed)
    data = np.random.randn(dim, 1000)
    cov = np.cov(data)
    external = cov_matrix_to_sdcorr_params(cov)
    return external


@pytest.mark.parametrize("dim, seed", to_test)
def test_covariance_from_internal_jacobian(dim, seed):
    internal = get_internal_cholesky(dim)

    func = partial(kt.covariance_from_internal, **{"constr": None})
    numerical_deriv = first_derivative(func, internal)
    deriv = kt.covariance_from_internal_jacobian(internal, None)

    aaae(deriv, numerical_deriv["derivative"], decimal=3)


@pytest.mark.parametrize("dim, seed", to_test)
def test_covariance_to_internal_jacobian(dim, seed):
    external = get_external_covariance(dim)

    func = partial(kt.covariance_to_internal, **{"constr": None})
    numerical_deriv = first_derivative(func, external)
    deriv = kt.covariance_to_internal_jacobian(external, None)

    aaae(deriv, numerical_deriv["derivative"], decimal=3)


@pytest.mark.parametrize("dim, seed", to_test)
def test_probability_from_internal_jacobian(dim, seed):
    internal = get_internal_probability(dim)

    func = partial(kt.probability_from_internal, **{"constr": None})
    numerical_deriv = first_derivative(func, internal)
    deriv = kt.probability_from_internal_jacobian(internal, None)

    aaae(deriv, numerical_deriv["derivative"], decimal=3)


@pytest.mark.parametrize("dim, seed", to_test)
def test_probability_to_internal_jacobian(dim, seed):
    external = get_external_probability(dim)

    func = partial(kt.probability_to_internal, **{"constr": None})
    numerical_deriv = first_derivative(func, external)
    deriv = kt.probability_to_internal_jacobian(external, None)

    aaae(deriv, numerical_deriv["derivative"], decimal=3)


@pytest.mark.parametrize("dim, seed", to_test)
def test_sdcorr_from_internal_jacobian(dim, seed):
    internal = get_internal_cholesky(dim)

    func = partial(kt.sdcorr_from_internal, **{"constr": None})
    numerical_deriv = first_derivative(func, internal)
    deriv = kt.sdcorr_from_internal_jacobian(internal, None)

    aaae(deriv, numerical_deriv["derivative"], decimal=3)


@pytest.mark.parametrize("dim, seed", to_test)
def test_sdcorr_to_internal_jacobian(dim, seed):
    external = get_external_sdcorr(dim)

    func = partial(kt.sdcorr_to_internal, **{"constr": None})
    numerical_deriv = first_derivative(func, external)
    deriv = kt.sdcorr_to_internal_jacobian(external, None)

    aaae(deriv, numerical_deriv["derivative"], decimal=3)
