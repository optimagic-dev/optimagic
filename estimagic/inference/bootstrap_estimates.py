import random

import numpy as np
import pandas as pd
from joblib import delayed
from joblib import Parallel

from estimagic.inference.bootstrap_samples import _get_cluster_index
from estimagic.inference.bootstrap_samples import get_seeds


def get_bootstrap_estimates(data, f, cluster_by=None, seeds=None, num_threads=1):
    """Calculate the statistic for every drawn sample.

    Args:
        data (pd.DataFrame): original dataset.
        f (callable): function of the dataset calculating statistic of interest.
        cluster_by (str): column name of the variable to cluster by.
        seeds (np.array): Size ndraws vector of drawn seeds or None.
        num_threads (int): number of jobs for parallelization.

    Returns:
        estimates (pd.DataFrame): DataFrame estimates for different bootstrap samples.

    """
    if seeds is None:
        seeds = get_seeds(1000)

    n = len(data)

    if cluster_by is None:

        def loop(s):

            np.random.seed(s)
            draw_ids = np.random.randint(0, n, size=n)
            draw = data.iloc[draw_ids]
            return f(draw)

        estimates = Parallel(n_jobs=num_threads)(delayed(loop)(s) for s in seeds)

    else:
        clusters = _get_cluster_index(data, cluster_by)
        nclusters = len(clusters)

        estimates = []

        for s in seeds:
            random.seed(s)
            draw_ids = np.concatenate(random.choices(clusters, k=nclusters))
            draw = data.iloc[draw_ids]
            estimates.append(f(draw))

    # need to get column names for estimates dataframe from f?

    return pd.DataFrame(estimates)
