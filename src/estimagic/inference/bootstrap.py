import functools
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import pandas as pd
from estimagic.batch_evaluators import joblib_batch_evaluator
from estimagic.inference.bootstrap_ci import compute_ci
from estimagic.inference.bootstrap_helpers import check_inputs
from estimagic.inference.bootstrap_outcomes import get_bootstrap_outcomes
from estimagic.parameters.tree_registry import get_registry
from pybaum import tree_just_flatten


def bootstrap(
    data,
    outcome,
    *,
    outcome_kwargs=None,
    n_draws=1_000,
    cluster_by=None,
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
            Returns a general pytree (e.g. pandas Series, dict, numpy array, etc.).
        outcome_kwargs (dict): Additional keyword arguments for outcome.
        n_draws (int): number of bootstrap samples to draw.
        cluster_by (str): column name of variable to cluster by or None.
        seeds (numpy.array): array of seeds for bootstrap samples, default is none.
        n_cores (int): number of jobs for parallelization.
        error_handling (str): One of "continue", "raise". Default "continue" which means
            that bootstrap estimates are only calculated for those samples where no
            errors occur and a warning is produced if any error occurs.
        batch_evaluator (str or Callable): Name of a pre-implemented batch evaluator
            (currently 'joblib' and 'pathos_mp') or Callable with the same interface
            as the estimagic batch_evaluators. See :ref:`batch_evaluators`.

    Returns:
        BootstrapResult: A BootstrapResult object storing information on summary
            statistics, the covariance matrix, and the estimated boostrap outcomes.
    """
    check_inputs(data, cluster_by)

    registry = get_registry(extended=True)
    if outcome_kwargs is not None:
        outcome = functools.partial(outcome, **outcome_kwargs)

    @functools.wraps(outcome)
    def outcome_flat(data):
        raw = outcome(data)
        out = tree_just_flatten(raw, registry=registry)
        return out

    estimates = get_bootstrap_outcomes(
        data=data,
        outcome=outcome_flat,
        cluster_by=cluster_by,
        seed=seed,
        n_draws=n_draws,
        n_cores=n_cores,
        error_handling=error_handling,
        batch_evaluator=batch_evaluator,
    )

    out = bootstrap_from_outcomes(
        data,
        outcome,
        estimates,
    )

    return out


def bootstrap_from_outcomes(data, outcome, bootstrap_outcomes):
    """Set up results table containing mean, standard deviation and confidence interval
    for each estimated parameter.

    Args:
        data (pandas.DataFrame): original dataset.
        outcome (callable): function of the data calculating statistic of interest.
            Returns a general pytree (e.g. pandas Series, dict, numpy array, etc.).
        bootstrap_outcomes (pandas.DataFrame): DataFrame of bootstrap_outcomes in the
            bootstrap samples.

    Returns:
        BootstrapResult: A BootstrapResult object storing information on summary
            statistics, the covariance matrix, and the estimated boostrap outcomes.
    """
    out = BootstrapResult(
        params=data,
        outcome=outcome,
        bootstrap_outcomes=bootstrap_outcomes,
    )

    return out


@dataclass
class BootstrapResult:
    params: pd.DataFrame
    outcome: Callable
    bootstrap_outcomes: Any

    @property
    def _summary(self):
        return self.summary()

    @property
    def _cov(self):
        return self.cov()

    def summary(
        self,
        ci_method="percentile",
        alpha=0.05,
        n_cores=1,
    ):
        """Create a summary of estimation results.

        Args:
           ci_method (str): method of choice for confidence interval computation.
            alpha (float): significance level of choice.
            n_cores (int): number of jobs for parallelization.

        Returns:
            pd.DataFrame: The estimation summary as a DataFrame containing information
                on the mean, standard errors, as well as the confindence interval.
                Soon this will be a pytree of DataFrames.
        """
        check_inputs(data=self.params, ci_method=ci_method, alpha=alpha)

        cis = compute_ci(
            self.params,
            self.outcome,
            self.bootstrap_outcomes,
            ci_method,
            alpha,
            n_cores,
        )

        summary = pd.DataFrame(self.bootstrap_outcomes.mean(axis=0), columns=["mean"])
        summary["std"] = self.bootstrap_outcomes.std(axis=0)

        summary["lower_ci"] = cis["lower_ci"]
        summary["upper_ci"] = cis["upper_ci"]

        return summary

    def cov(self):
        """Calculate the variance-covariance matrix of the estimated parameters.

        Returns:
            pd.DataFrame: The covariance matrix of the estimated parameters as a
                pandas.DataFrame. Soon, block-pytree or numpy array will also be
                supported.
        """
        return self.bootstrap_outcomes.cov()
