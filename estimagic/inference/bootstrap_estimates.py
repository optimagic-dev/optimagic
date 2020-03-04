import random

import numpy as np
import pandas as pd
from joblib import delayed
from joblib import Parallel

from estimagic.inference.bootstrap_ci import _check_inputs
from estimagic.inference.bootstrap_ci import _concatenate_functions
from estimagic.inference.bootstrap_samples import _get_cluster_index
from estimagic.inference.bootstrap_samples import get_seeds


def get_bootstrap_estimates(
    data, f, cluster_by=None, seeds=None, ndraws=1000, num_threads=1
):
    """Calculate the statistic f for every bootstrap sample, either by specified seeds
    or for ndraws random samples.

    Args:
        data (pandas.DataFrame): original dataset.
        f (callable): function of the dataset calculating statistic of interest or list
            of functions.
        cluster_by (str): column name of the variable to cluster by.
        seeds (numpy.array): Size ndraws vector of drawn seeds or None.
        ndraws (int): number of draws, only relevant if seeds is None.
        num_threads (int): number of jobs for parallelization.

    Returns:
        estimates (pandas.DataFrame): DataFrame estimates for different bootstrap
            samples.

    """

    _check_inputs(data=data, cluster_by=cluster_by)
    if isinstance(f, list):
        f = _concatenate_functions(f, data)

    if seeds is None:
        seeds = get_seeds(ndraws)

    if cluster_by is None:

        estimates = _get_uniform_estimates(data, seeds, num_threads, f)

    else:

        estimates = _get_clustered_estimates(data, cluster_by, seeds, num_threads, f)

    return pd.DataFrame(estimates)


def _get_uniform_estimates(data, seeds, num_threads=1, f=None):
    """Calculate non-clustered bootstrap estimates. If f is None, return a list of the
    samples.

    Args:
        data (pandas.DataFrame): original dataset.
        seeds (numpy.array): Size ndraws vector of drawn seeds or None.
        num_threads (int): number of jobs for parallelization.
        f (callable): function of the dataset calculating statistic of interest.

     Returns:
         estimates (list): list of estimates for different bootstrap samples.

     """

    n = len(data)

    def loop(s):

        np.random.seed(s)
        draw_ids = np.random.randint(0, n, size=n)
        draw = data.iloc[draw_ids]

        if f is None:
            res = draw
        else:
            res = f(draw)

        return res

    estimates = Parallel(n_jobs=num_threads)(delayed(loop)(s) for s in seeds)

    return estimates


def _get_clustered_estimates(data, cluster_by, seeds, num_threads=1, f=None):
    """Calculate clustered bootstrap estimates. If f is None, return a list of the
    samples.

    Args:
        data (pandas.DataFrame): original dataset.
        cluster_by (str): column name of the variable to cluster by.
        seeds (numpy.array): Size ndraws vector of drawn seeds or None.
        num_threads (int): number of jobs for parallelization.
        f (callable): function of the dataset calculating statistic of interest.

     Returns:
         estimates (list): list of estimates for different bootstrap samples.

     """

    clusters = _get_cluster_index(data, cluster_by)
    nclusters = len(clusters)

    def loop(s):
        random.seed(s)
        draw_ids = np.concatenate(random.choices(clusters, k=nclusters))
        draw = data.iloc[draw_ids]

        if f is None:
            res = draw
        else:
            res = f(draw)

        return res

    estimates = Parallel(n_jobs=num_threads)(delayed(loop)(s) for s in seeds)

    return estimates
