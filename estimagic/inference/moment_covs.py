"""
This module provides functions for the calculation of standard errors for anon linear
GMM estimator. The source of these calculations is
Bruce E. Hansen - Econometrics (https://www.ssc.wisc.edu/~bhansen/econometrics).
"""
import numpy as np

from estimagic.exceptions import INVALID_INFERENCE_MSG
from estimagic.optimization.utilities import robust_inverse


def gmm_cov(mom_cond, mom_cond_jacob, weighting_matrix):
    """

    Args:
        mom_cond (np.array): 2d array matrix of the moment conditions of
            dimension (nobs, nmoms).

        mom_cond_jacob (np.array): 3d array of the moment condition derivatives
            w.r.t. the parameters of dimension (nobs, nmoms, nparams).

        weighting_matrix (np.array):
            2d array weighting matrix for the moments of dimension (nmoms, nmoms)

    Returns:
        sandwich (np.array):
        2d array variance-covariance matrix of the GMM estimator of dimension
        (nparams, nparams)

    """

    # Use notation from Hansen book everywhere.
    omega = _covariance_moments(mom_cond)
    q_hat = np.mean(mom_cond_jacob, axis=0)

    return sandwich_cov(q_hat, weighting_matrix, omega, mom_cond.shape[0])


def _covariance_moments(mom_cond):
    """
    Calculate the standard covariance matrix Omega.

    Args:
        mom_cond (np.array): 2d array matrix of the moment conditions of
            dimension (nobs, nmoms).

    Returns:
        cov (np.array): 2d array covariance matrix of the moments (nmoms, nmoms)

    """

    dev = mom_cond - np.mean(mom_cond, axis=0)
    cov = dev.T @ dev / mom_cond.shape[0]
    return cov


def sandwich_cov(q_hat, weighting_matrix, omega, nobs):
    bread = robust_inverse(
        q_hat.T @ weighting_matrix @ q_hat, msg=INVALID_INFERENCE_MSG
    )
    butter = q_hat.T @ weighting_matrix @ omega @ weighting_matrix @ q_hat
    return bread @ butter @ bread / nobs
