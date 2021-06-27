import numpy as np

from estimagic.inference.bootstrap_ci import check_inputs


def get_seeds(n_draws=1000):
    """Draw seeds for bootstrap resampling.

    Args:
        n_draws (int): number of bootstrap draws.

    Returns:
        seeds (numpy.array): vector of randomly drawn seeds.

    """

    return np.random.randint(0, 2 ** 31, size=n_draws)


def get_bootstrap_samples(
    data, cluster_by=None, seeds=None, n_draws=1000, n_cores=1, return_samples=False
):
    """Draw and return bootstrap samples, either by specified seeds or number of draws.

    Args:
        data (pandas.DataFrame): original dataset.
        cluster_by (str): column name of the variable to cluster by.
        seeds (numpy.array): Size n_draws vector of drawn seeds or None.
        n_draws (int): number of draws, only relevant if seeds is None.
        n_cores (int): number of jobs for parallelization.
        return_samples (bool): If true, return samples, else return indices.

    Returns:
        samples (list): list of DataFrames containing resampled data or ids.

    """

    check_inputs(data=data, cluster_by=cluster_by)

    if seeds is None:
        seeds = get_seeds(n_draws)

    if cluster_by is None:

        from estimagic.inference.bootstrap_estimates import get_uniform_estimates

        sample_ids = get_uniform_estimates(data, seeds, n_cores, f=None)

    else:

        from estimagic.inference.bootstrap_estimates import get_clustered_estimates

        sample_ids = get_clustered_estimates(data, cluster_by, seeds, n_cores, f=None)

    if return_samples is True:

        result = [data.iloc[ids] for ids in sample_ids]

    else:

        result = sample_ids

    return result


def get_cluster_index(data, cluster_by):
    """Divide up the dataframe into clusters by variable cluster_by.

    Args:
        data (pandas.DataFrame): original dataset.
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
