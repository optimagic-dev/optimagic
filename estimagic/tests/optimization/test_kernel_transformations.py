from itertools import product

import numpy as np
import pytest
from jax import jacfwd
from numpy.testing import assert_array_almost_equal

import estimagic.optimization.kernel_transformations as kt
import estimagic.optimizations.kernel_transformation_jax as ktj
from estimagic.optimizations.kernel_transformations import cov_matrix_to_sdcorr_params

to_test = list(product(range(10, 30), range(5)))


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


@pytest.fixture
def jax_derivatives():
    out = {
        "covariance_from": jacfwd(ktj.covariance_from_internal),
        "covariance_to": jacfwd(ktj.covariance_to_internal),
        "probability_from": jacfwd(ktj.probability_from_internal),
        "probability_to": jacfwd(ktj.probability_to_internal),
        "sdcorr_from": jacfwd(ktj.sdcorr_from_internal),
        "sdcorr_to": jacfwd(ktj.sdcorr_to_internal),
    }
    return out


@pytest.mark.parametrize("dim, seed", to_test)
def test_covariance_from_internal_jacobian(dim, seed, jax_derivatives):
    internal = get_internal_cholesky(dim)
    jax_deriv = jax_derivatives["covariance_from"](internal)
    deriv = kt.covariance_from_internal_jacobian(internal)
    assert_array_almost_equal(deriv, jax_deriv, decimal=5)


@pytest.mark.parametrize("dim, seed", to_test)
def test_covariance_to_internal_jacobian(dim, seed, jax_derivatives):
    external = get_external_covariance(dim)
    jax_deriv = jax_derivatives["covariance_to"](external)
    deriv = kt.covariance_to_internal_jacobian(external)
    assert_array_almost_equal(deriv, jax_deriv, decimal=5)


@pytest.mark.parametrize("dim, seed", to_test)
def test_probability_from_internal_jacobian(dim, seed, jax_derivatives):
    internal = get_internal_probability(dim)
    jax_deriv = jax_derivatives["probability_from"](internal)
    deriv = kt.probability_from_internal_jacobian(internal)
    assert_array_almost_equal(deriv, jax_deriv, decimal=5)


@pytest.mark.parametrize("dim, seed", to_test)
def test_probability_to_internal_jacobian(dim, seed, jax_derivatives):
    external = get_external_probability(dim)
    jax_deriv = jax_derivatives["probability_to"](external)
    deriv = kt.probability_to_internal_jacobian(external)
    assert_array_almost_equal(deriv, jax_deriv, decimal=3)


@pytest.mark.parametrize("dim, seed", to_test)
def test_sdcorr_from_internal_jacobian(dim, seed, jax_derivatives):
    internal = get_internal_cholesky(dim)
    jax_deriv = jax_derivatives["sdcorr_from"](internal)
    deriv = kt.sdcorr_from_internal_jacobian(internal)
    assert_array_almost_equal(deriv, jax_deriv, decimal=5)


@pytest.mark.parametrize("dim, seed", to_test)
def test_sdcorr_to_internal_jacobian(dim, seed, jax_derivatives):
    external = get_external_sdcorr(dim)
    jax_deriv = jax_derivatives["sdcorr_to"](external)
    deriv = kt.sdcorr_to_internal_jacobian(external)
    assert_array_almost_equal(deriv, jax_deriv, decimal=5)
