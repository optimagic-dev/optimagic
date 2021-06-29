import pandas as pd

from estimagic.batch_evaluators import joblib_batch_evaluator
from estimagic.inference.bootstrap_ci import compute_ci
from estimagic.inference.bootstrap_helpers import check_inputs
from estimagic.inference.bootstrap_outcomes import get_bootstrap_outcomes


def bootstrap(
    data,
    outcome,
    n_draws=1_000,
    cluster_by=None,
    ci_method="percentile",
    alpha=0.05,
    seed=None,
    n_cores=1,
    error_handling="continue",
    batch_evaluator=joblib_batch_evaluator,
):
    """Calculate bootstrap estimates, standard errors and confidence intervals
    for statistic of interest in given original sample.

    Args:
        data (pandas.DataFrame): original dataset.
        outcome (callable): function of the data calculating statistic of interest.
            Needs to return array-like object or pd.Series.
        n_draws (int): number of bootstrap samples to draw.
        cluster_by (str): column name of variable to cluster by or None.
        ci_method (str): method of choice for confidence interval computation.
        alpha (float): significance level of choice.
        seeds (numpy.array): array of seeds for bootstrap samples, default is none.
        n_cores (int): number of jobs for parallelization.
        error_handling (str): One of "continue", "raise". Default "continue" which means
            that bootstrap estimates are only calculated for those samples where no
            errors occur and a warning is produced if any error occurs.
        batch_evaluator (str or Callable): Name of a pre-implemented batch evaluator
            (currently 'joblib' and 'pathos_mp') or Callable with the same interface
            as the estimagic batch_evaluators. See :ref:`batch_evaluators`.

    Returns:
        results (pandas.DataFrame): DataFrame where k'th row contains mean estimate,
        standard error, and confidence interval of k'th parameter.

    """

    check_inputs(data, cluster_by, ci_method, alpha)

    estimates = get_bootstrap_outcomes(
        data=data,
        outcome=outcome,
        cluster_by=cluster_by,
        seed=seed,
        n_cores=n_cores,
        error_handling=error_handling,
        batch_evaluator=batch_evaluator,
    )

    table = get_results_table(data, outcome, estimates, ci_method, alpha, n_cores)

    return table


def get_results_table(
    data, outcome, estimates, ci_method="percentile", alpha=0.05, n_cores=1
):
    """Set up results table containing mean, standard deviation and confidence interval
    for each estimated parameter.

    Args:
        data (pandas.DataFrame): original dataset.
        outcome (callable): function of the data calculating statistic of interest.
            Needs to return array-like object or pd.Series.
        estimates (pandas.DataFrame): DataFrame of estimates in the bootstrap samples.
        ci_method (str): method of choice for confidence interval computation.
        n_cores (int): number of jobs for parallelization.
        alpha (float): significance level of choice.

    Returns:
        results (pandas.DataFrame): table of results.

    """

    check_inputs(data=data, ci_method=ci_method, alpha=alpha)

    results = pd.DataFrame(estimates.mean(axis=0), columns=["mean"])

    results["std"] = estimates.std(axis=0)

    cis = compute_ci(data, outcome, estimates, ci_method, alpha, n_cores)
    results["lower_ci"] = cis["lower_ci"]
    results["upper_ci"] = cis["upper_ci"]

    return results
