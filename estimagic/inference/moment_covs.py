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
