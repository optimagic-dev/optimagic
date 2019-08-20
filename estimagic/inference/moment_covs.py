"""
This module provides functions for the calculation of standard errors for anon linear
GMM estimator. The source of these calculations is
Bruce E. Hansen - Econometrics (https://www.ssc.wisc.edu/~bhansen/econometrics).
We also adopted the notation, where
    - nobs: number of observations
    - l: number of moment conditions
    - k: size of estimated parameter vector
"""
import numpy as np



def covariance_moments(mom_cond):
    """
    Calculate the standard covariance matrix Omega.

    Args:
        mom_cond (np.array): 2d array matrix of the moment conditions of
        dimension (nobs, nmoms).

    Returns:
        cov (np.array): 2d array covariance matrix of the moments (nmoms, nmoms)

    """

    dev = mom_cond - np.mean(mom_cond, axis=0)
    cov = dev @ dev.T / mom_cond.shape[0]

    return cov


def gmm_cov(mom_cond, mom_cond_jacob, mom_weight):
    """

    Args:
        mom_cond (np.array):
            n x l array containing for n individuals the values of the
            l moment conditions.

        mom_cond_jacob (np.array):
            n x l x k array containing for n individuals the derivatives
            of l moments with respect to k parameters.

        mom_weight (np.array):
            Weighting matrix for the moments of dimension (nmoms, nmoms)

    Returns:

        sandwich (np.array):
            Variance-covariance matrix of the GMM estimator of dimension
            (nparams, nparams)

    """

    # Use notation from Hansen book everywhere
    W = mom_weight
    Omega = covariance_moments(mom_cond)
    Q = np.mean(mom_cond_jacob, axis=0)

    bread = np.linalg.inv(Q.T @ W @ Q)
    butter = Q.T @ W @ Omega @ W @ Q
    sandwich = bread @ butter @ bread

    return sandwich
