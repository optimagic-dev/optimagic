"""
This module provides functions for the calculation of standard errors for anon linear
GMM estimator. The source of these calculations is
Bruce E. Hansen - Econometrics (https://www.ssc.wisc.edu/~bhansen/econometrics).
"""
import numpy as np


def gmm_cov(mom_cond, mom_cond_jacob, mom_weight):
    """

    Args:
        mom_cond (np.array): 2d array matrix of the moment conditions of
            dimension (nobs, nmoms).

        mom_cond_jacob (np.array): 3d array of the moment condition derivatives
            w.r.t. the parameters of dimension (nobs, nmoms, nparams).

        mom_weight (np.array):
            2d array weighting matrix for the moments of dimension (nmoms, nmoms)

    Returns:
        sandwich (np.array):
        2d array variance-covariance matrix of the GMM estimator of dimension
        (nparams, nparams)

    """

    # Use notation from Hansen book everywhere. Tell flake8 to ignore capital notation
    W = mom_weight  # noqa: N806
    Omega = _covariance_moments(mom_cond)  # noqa: N806
    Q = np.mean(mom_cond_jacob, axis=0)  # noqa: N806

    return _sandwich_cov(Q, W, Omega, mom_cond.shape[0])


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


def _sandwich_cov(Q, W, Omega, nobs):  # noqa: N803
    bread = np.linalg.inv(Q.T @ W @ Q)
    butter = Q.T @ W @ Omega @ W @ Q
    return bread @ butter @ bread / nobs
