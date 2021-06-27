import pandas as pd

from estimagic.inference.bootstrap_ci import check_inputs
from estimagic.inference.bootstrap_ci import compute_ci
from estimagic.inference.bootstrap_ci import concatenate_functions
from estimagic.inference.bootstrap_estimates import get_bootstrap_estimates
from estimagic.inference.bootstrap_estimates import mean
from estimagic.inference.bootstrap_samples import get_seeds


def bootstrap(
    data,
    f=mean,
    n_draws=1_000,
    cluster_by=None,
    ci_method="percentile",
    alpha=0.05,
    seeds=None,
    n_cores=1,
):
    """Calculate bootstrap estimates, standard errors and confidence intervals
    for statistic of interest in given original sample.

    Args:
        data (pandas.DataFrame): original dataset.
        f (callable): function of the data calculating statistic of interest or list of
            functions. Needs to return array-like object or pd.Series.
        n_draws (int): number of bootstrap samples to draw.
        cluster_by (str): column name of variable to cluster by or None.
        ci_method (str): method of choice for confidence interval computation.
        alpha (float): significance level of choice.
        seeds (numpy.array): array of seeds for bootstrap samples, default is none.
        n_cores (int): number of jobs for parallelization.

    Returns:
        results (pandas.DataFrame): DataFrame where k'th row contains mean estimate,
        standard error, and confidence interval of k'th parameter.

    """

    check_inputs(data, cluster_by, ci_method, alpha)
    if isinstance(f, list):
        f = concatenate_functions(f, data)

    df = data.reset_index(drop=True)

    if seeds is None:
        seeds = get_seeds(n_draws)

    estimates = get_bootstrap_estimates(df, f, cluster_by, seeds, n_cores)

    table = get_results_table(df, f, estimates, ci_method, alpha, n_cores)

    return table


def get_results_table(
    data, f, estimates, ci_method="percentile", alpha=0.05, n_cores=1
):
    """Set up results table containing mean, standard deviation and confidence interval
    for each estimated parameter.

    Args:
        data (pandas.DataFrame): original dataset.
        f (callable): function of the data calculating statistic of interest or list of
            functions. Needs to return array-like object or pd.Series.
        estimates (pandas.DataFrame): DataFrame of estimates in the bootstrap samples.
        ci_method (str): method of choice for confidence interval computation.
        n_cores (int): number of jobs for parallelization.
        alpha (float): significance level of choice.

    Returns:
        results (pandas.DataFrame): table of results.

    """

    check_inputs(data=data, ci_method=ci_method, alpha=alpha)
    if isinstance(f, list):
        f = concatenate_functions(f, data)

    results = pd.DataFrame(estimates.mean(axis=0), columns=["mean"])

    results["std"] = estimates.std(axis=0)

    cis = compute_ci(data, f, estimates, ci_method, alpha, n_cores)
    results["lower_ci"] = cis["lower_ci"]
    results["upper_ci"] = cis["upper_ci"]

    return results
