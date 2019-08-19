import numpy as np


def average_contribution(mom_cond):
    """
    Calculates the average condition value for l moment conditions of n
    observations.
    Args:
        mom_cond (np.array): 3d array moment conditions values

    Returns:
        mom_cond_mean (np.array): 3d array moment conditions means
    """
    n = mom_cond.shape[0]
    mom_cond_mean = np.sum(mom_cond, axis=0) / n
    return mom_cond_mean


def calc_standard_cov(mom_cond):
    """
    Calculates the standard covariance matrix Omega.
    Args:
        mom_cond (np.array): 3d array moment conditions values

    Returns:
        cov (np.array):
    """
    n = mom_cond.shape[0]
    mom_cond_mean = average_contribution(mom_cond)
    cov = np.zeros(shape=(mom_cond.shape[1:3]))
    for i in range(n):
        mom_cond_dev = mom_cond[i, :, :] - mom_cond_mean
        cov = cov + np.dot(mom_cond_dev, mom_cond_dev.transpose())
    cov = cov / n
    return cov
