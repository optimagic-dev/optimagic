import pandas as pd

from estimagic.inference.bootstrap_ci import _check_inputs
from estimagic.inference.bootstrap_ci import compute_ci
from estimagic.inference.bootstrap_estimates import get_bootstrap_estimates
from estimagic.inference.bootstrap_samples import get_seeds


def _mean(df):
    return df.mean(axis=0)


def bootstrap(
    data,
    f=_mean,
    ndraws=1000,
    cluster_by=None,
    ci_method="percentile",
    alpha=0.05,
    seeds=None,
    return_seeds=False,
    num_threads=1,
):
    """Calculate bootstrap estimates, standard errors and confidence intervals
    for statistic of interest in given original sample.

    Args:
        data (pandas.DataFrame): original dataset.
        f (callable): function of the data calculating statistic of interest.
        ndraws (int): number of bootstrap samples to draw.
        cluster_by (str): column name of variable to cluster by or None.
        ci_method (str): method of choice for confidence interval computation.
        alpha (float): significance level of choice.
        return_seeds (bool): specify whether to return the drawn seeds as 2nd argument.
        seeds (numpy.array): array of seeds for bootstrap samples, default is none.
        num_threads (int): number of jobs for parallelization.

    Returns:
        results (pd.DataFrame): DataFrame where k'th row contains mean estimate,
        standard error, and confidence interval of k'th parameter.

    """

    _check_inputs(data, cluster_by, ci_method, alpha)

    df = data.reset_index(drop=True)

    if seeds is None:
        seeds = get_seeds(ndraws)

    estimates = get_bootstrap_estimates(df, f, cluster_by, seeds, num_threads)

    results = get_results_table(df, f, estimates, ci_method, alpha, num_threads)

    if return_seeds is False:
        return results

    elif return_seeds is True:
        return results, seeds


def get_results_table(
    data, f, estimates, ci_method="percentile", alpha=0.05, num_threads=1
):
    """Set up results table containing mean, standard deviation and confidence interval
    for each estimated parameter.

    Args:
        data (pandas.DataFrame): original dataset.
        f (callable): function of the data calculating statistic of interest.
        estimates (data.Frame): DataFrame of estimates in the bootstrap samples.
        ci_method (str): method of choice for confidence interval computation.
        num_threads (int): number of jobs for parallelization.
        alpha (float): significance level of choice.

    Returns:
        results (pd.DataFrame): table of results.

    """

    _check_inputs(data=data, ci_method=ci_method, alpha=alpha)

    results = pd.DataFrame(estimates.mean(axis=0), columns=["mean"])

    results["std"] = estimates.std(axis=0)

    cis = compute_ci(data, f, estimates, ci_method, alpha, num_threads)
    results["lower_ci"] = cis["lower_ci"]
    results["upper_ci"] = cis["upper_ci"]

    return results
