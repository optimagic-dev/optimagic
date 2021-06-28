import numpy as np
import pandas as pd


def check_inputs(data, cluster_by=None, ci_method="percentile", alpha=0.05):
    """Check validity of inputs.
    Args:
        data (pd.DataFrame): original dataset.
        cluster_by (str): column name of variable to cluster by.
        ci_method (str): method of choice for confidence interval computation.
        alpha (float): significance level of choice.

    """

    ci_method_list = ["percentile", "bca", "bc", "t", "normal", "basic"]

    if not isinstance(data, pd.DataFrame):
        raise ValueError("Input 'data' must be DataFrame.")

    elif (cluster_by is not None) and (cluster_by not in data.columns.tolist()):
        raise ValueError(
            "Input 'cluster_by' must be None or a column name of DataFrame."
        )

    elif ci_method not in ci_method_list:
        raise ValueError(
            "ci_method must be 'percentile', 'bc',"
            " 'bca', 't', 'basic' or 'normal', '{method}'"
            " was supplied".format(method=ci_method)
        )

    elif alpha > 1 or alpha < 0:
        raise ValueError("Input 'alpha' must be in [0,1].")


def get_seeds(n_draws=1000):
    """Draw seeds for bootstrap resampling.

    Args:
        n_draws (int): number of bootstrap draws.

    Returns:
        seeds (numpy.array): vector of randomly drawn seeds.

    """

    return np.random.randint(0, 2 ** 31, size=n_draws)


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
