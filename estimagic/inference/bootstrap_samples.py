import random

import numpy as np
from joblib import delayed
from joblib import Parallel


def get_bootstrap_sample_seeds(ndraws):
    """Draw seeds for bootstrap resampling.

    Args:
        ndraws (int): number of bootstrap draws.

    Returns:
        seeds (np.array): vector of randomly drawn seeds.

    """

    return np.random.randint(0, 2 ** 31, size=ndraws)


# REWRITE THIS TO RETURN EITHER INDEXES OR ACTUAL SAMPLES OR DELETE
# CAN'T I WRITE THIS WITH get_bootstrap_estimates?
def get_bootstrap_samples(data, cluster_by=None, seeds=None, num_threads=1):
    """Calculate the statistic for every drawn sample.

    Args:
        data (pd.DataFrame): original dataset.
        cluster_by (str): column name of the variable to cluster by.
        seeds (np.array): Size ndraws vector of drawn seeds or None.
        num_threads (int): number of jobs for parallelization.

    Returns:
        samples (list): list of DataFrames containing resampled data.

    """
    if seeds is None:
        seeds = get_bootstrap_sample_seeds(1000)

    n = len(data)

    if cluster_by is None:

        def loop(s):

            np.random.seed(s)
            draw_ids = np.random.randint(0, n, size=n)
            draw = data.iloc[draw_ids]
            return draw

        estimates = Parallel(n_jobs=num_threads)(delayed(loop)(s) for s in seeds)

    else:
        clusters = _get_cluster_index(data, cluster_by)
        nclusters = len(clusters)

        estimates = []

        for s in seeds:
            random.seed(s)
            draw_ids = np.concatenate(random.choices(clusters, k=nclusters))
            draw = data.iloc[draw_ids]
            estimates.append(draw)

    return estimates


def _get_cluster_index(data, cluster_by):
    """Divide up the dataframe into clusters by variable cluster_by.

    Args:
        data (pd.DataFrame): original dataset.
        cluster_by (str): column name of variable to cluster by.

    Returns:
        clusters (list): list of arrays of row numbers belonging
        to the different clusters.

    """

    cluster_vals = data[cluster_by].unique()

    clusters = [
        np.array(data[data[cluster_by] == val].index.values.tolist())
        for val in cluster_vals
    ]

    return clusters
