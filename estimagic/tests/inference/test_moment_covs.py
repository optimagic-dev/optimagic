import numpy as np
from numpy.testing import assert_array_almost_equal

from estimagic.inference.moment_covs import covariance_moments


def test_covariance_moments_random():
    nobs, nmoms = np.random.randint(1, 50, size=2)
    mom_cond = np.random.rand(nobs, nmoms)
    dev = (mom_cond - np.mean(mom_cond, axis=0)).reshape(nobs, nmoms, 1)
    cov = np.zeros(shape=(nmoms, nmoms), dtype=float)
    for i in range(nobs):
        cov += dev[i, :, :] @ dev[i, :, :].T
    cov = cov / nobs
    assert_array_almost_equal(covariance_moments(mom_cond), cov)


def test_covariance_moments_unit():
    moment_cond = np.reshape(np.arange(12), (3, 4))
    control = np.full((4, 4), 32, dtype=np.float) / 3
    assert_array_almost_equal(covariance_moments(moment_cond), control)
