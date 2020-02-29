import numpy as np

import estimagic.inference.bootstrap_estimates as est


def get_seeds(ndraws):
    """Draw seeds for bootstrap resampling.

    Args:
        ndraws (int): number of bootstrap draws.

    Returns:
        seeds (np.array): vector of randomly drawn seeds.

    """

    return np.random.randint(0, 2 ** 31, size=ndraws)


def get_bootstrap_samples(
    data, cluster_by=None, seeds=None, ndraws=1000, num_threads=1
):
    """Return the drawn bootstrap samples.

    Args:
        data (pd.DataFrame): original dataset.
        cluster_by (str): column name of the variable to cluster by.
        seeds (np.array): Size ndraws vector of drawn seeds or None.
        ndraws (int): number of draws, only relevant if seeds is None.
        num_threads (int): number of jobs for parallelization.

    Returns:
        samples (list): list of DataFrames containing resampled data.

    """
    if seeds is None:
        seeds = get_seeds(ndraws)

    if cluster_by is None:

        samples = est._get_uniform_estimates(data, seeds, num_threads, f=None)

    else:

        samples = est._get_clustered_estimates(
            data, cluster_by, seeds, num_threads, f=None
        )

    return samples


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
