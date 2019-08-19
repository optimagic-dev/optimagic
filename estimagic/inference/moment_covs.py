"""
This module provides functions for the calculation of standard errors for anon linear
GMM estimator. The source of these calculations is
Bruce E. Hansen - Econometrics.
We also adopted the notation, where
    - n: number of observations
    - l: number of moment conditions
    - k: size of estimated parameter vector
"""
import numpy as np


def average_contribution(mom_cond):
    """
    Calculates the average condition value for l moment conditions of n
    observations.
    Args:
        mom_cond (np.array):
            3d array containing for each of n observations
            either lx1 moment condition values or lxk moment condition derivatives

    Returns:
        mom_cond_mean (np.array):
    """
    n = mom_cond.shape[0]
    mom_cond_mean = np.sum(mom_cond, axis=0) / n
    return mom_cond_mean


def covariance_moments(mom_cond):
    """
    Calculates the standard covariance matrix Omega.
    Args:
        mom_cond (np.array):
            3d array containing for each of n observations lx1
            moment condition values

    Returns:
        cov (np.array):
    """
    n = mom_cond.shape[0]
    mom_cond_mean = average_contribution(mom_cond)
    cov = np.zeros(shape=(mom_cond.shape[1], mom_cond.shape[1]))
    mom_cond_dev = mom_cond - mom_cond_mean
    for i in range(n):
        cov += np.dot(mom_cond_dev[i, :, :], mom_cond_dev[i, :, :].transpose())
    cov = cov / n
    return cov


def gmm_cov(mom_cond, mom_cond_diff, weight_m):
    """

    Args:
        mom_cond (np.array):
            3d array containing for each of n observations
            either lx1 moment condition values

        mom_cond_diff (np.array):
            3d array containing for each of n observations the lxk jacobian of the
            moment conditions

        weight_m (np.array):
            A lxl weighting matrix.

    Returns:

    """
    omega = covariance_moments(mom_cond)
    q = average_contribution(mom_cond_diff)
    qwq_inv = np.linalg.inv(np.linalg.multi_dot([q.T, weight_m, q]))
    aux = np.linalg.multi_dot([q.T, weight_m, omega, weight_m, q])
    return np.linalg.multi_dot([qwq_inv, aux, qwq_inv])
