import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal

from estimagic.inference.moment_covs import _covariance_moments
from estimagic.inference.moment_covs import gmm_cov
from estimagic.inference.moment_covs import sandwich_cov


def test_covariance_moments_random():
    nobs, nmoms = np.random.randint(1, 50, size=2)
    mom_cond = np.random.rand(nobs, nmoms)
    dev = (mom_cond - np.mean(mom_cond, axis=0)).reshape(nobs, nmoms, 1)
    cov = np.zeros(shape=(nmoms, nmoms), dtype=float)
    for i in range(nobs):
        cov += dev[i, :, :] @ dev[i, :, :].T
    cov = cov / nobs
    assert_array_almost_equal(_covariance_moments(mom_cond), cov)


def test_covariance_moments_unit():
    moment_cond = np.reshape(np.arange(12), (3, 4))
    control = np.full((4, 4), 32, dtype=np.float) / 3
    assert_array_almost_equal(_covariance_moments(moment_cond), control)


@pytest.fixture
def fixtures_gmm_cov():
    """ The fixture contains a test case for our functions. The expected result was
    calculated by hand.s"""
    fix = {}
    fix["mom_cond"] = np.array([[0.1, 0.3], [0.7, 1.3]])
    fix["mom_cond_jacob"] = np.array(
        [[[3.0, 3.6], [4.0, 4.5]], [[3.2, 3.8], [4.2, 4.9]]]
    )
    fix["mom_weight"] = np.array([[13, 17], [23, 29]])
    fix["cov_result"] = np.array([[0.26888897, -0.19555563], [-0.19555563, 0.14222229]])
    return fix


def test_gmm_cov(fixtures_gmm_cov):
    fix = fixtures_gmm_cov
    assert_array_almost_equal(
        gmm_cov(fix["mom_cond"], fix["mom_cond_jacob"], fix["mom_weight"]),
        fix["cov_result"],
    )


def test_sandwich_cov(fixtures_gmm_cov):
    fix = fixtures_gmm_cov
    cov_moments = _covariance_moments(fix["mom_cond"])  # noqa: N806
    mean_mom_jacobi = np.mean(fix["mom_cond_jacob"], axis=0)  # noqa: N806
    assert_array_almost_equal(
        sandwich_cov(
            mean_mom_jacobi, fix["mom_weight"], cov_moments, fix["mom_cond"].shape[0]
        ),
        fix["cov_result"],
    )
