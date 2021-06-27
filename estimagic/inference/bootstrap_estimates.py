import random

import numpy as np
import pandas as pd
from joblib import delayed
from joblib import Parallel

from estimagic.inference.bootstrap_helpers import check_inputs
from estimagic.inference.bootstrap_helpers import concatenate_functions
from estimagic.inference.bootstrap_helpers import get_cluster_index
from estimagic.inference.bootstrap_helpers import get_seeds


def get_bootstrap_estimates(
    data, f, cluster_by=None, seeds=None, n_draws=1000, n_cores=1
):
    """Calculate the statistic f for every bootstrap sample, either by specified seeds
    or for n_draws random samples.

    Args:
        data (pandas.DataFrame): original dataset.
        f (callable): function of the dataset calculating statistic of interest or list
            of functions. Needs to return array-like object or pd.Series.
        cluster_by (str): column name of the variable to cluster by.
        seeds (numpy.array): Size n_draws vector of drawn seeds or None.
        n_draws (int): number of draws, only relevant if seeds is None.
        n_cores (int): number of jobs for parallelization.

    Returns:
        estimates (pandas.DataFrame): DataFrame estimates for different bootstrap
            samples.

    """

    check_inputs(data=data, cluster_by=cluster_by)
    if isinstance(f, list):
        f = concatenate_functions(f, data)

    if seeds is None:
        seeds = get_seeds(n_draws)

    df = data.reset_index(drop=True)

    if cluster_by is None:

        estimates = get_uniform_estimates(df, seeds, n_cores, f)

    else:

        estimates = get_clustered_estimates(df, cluster_by, seeds, n_cores, f)

    return pd.DataFrame(estimates)


def get_uniform_estimates(data, seeds, n_cores=1, f=None):
    """Calculate non-clustered bootstrap estimates. If f is None, return a list of the
    samples.

    Args:
        data (pandas.DataFrame): original dataset.
        seeds (numpy.array): Size n_draws vector of drawn seeds or None.
        n_cores (int): number of jobs for parallelization.
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
            res = draw_ids
        else:
            res = f(draw)

        return res

    estimates = Parallel(n_jobs=n_cores)(delayed(loop)(s) for s in seeds)

    return estimates


def get_clustered_estimates(data, cluster_by, seeds, n_cores=1, f=None):
    """Calculate clustered bootstrap estimates. If f is None, return a list of the
    samples.

    Args:
        data (pandas.DataFrame): original dataset.
        cluster_by (str): column name of the variable to cluster by.
        seeds (numpy.array): Size n_draws vector of drawn seeds or None.
        n_cores (int): number of jobs for parallelization.
        f (callable): function of the dataset calculating statistic of interest.

     Returns:
         estimates (list): list of estimates for different bootstrap samples.

    """

    clusters = get_cluster_index(data, cluster_by)
    nclusters = len(clusters)

    def loop(s):
        random.seed(s)
        draw_ids = np.concatenate(random.choices(clusters, k=nclusters))
        draw = data.iloc[draw_ids]

        if f is None:
            res = draw_ids
        else:
            res = f(draw)

        return res

    estimates = Parallel(n_jobs=n_cores)(delayed(loop)(s) for s in seeds)

    return estimates


def mean(df):
    return df.mean(axis=0)
