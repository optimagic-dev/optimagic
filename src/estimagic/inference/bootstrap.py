import functools
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from estimagic.batch_evaluators import joblib_batch_evaluator
from estimagic.inference.bootstrap_ci import compute_ci
from estimagic.inference.bootstrap_helpers import check_inputs
from estimagic.inference.bootstrap_outcomes import get_bootstrap_outcomes
from estimagic.parameters.block_trees import matrix_to_block_tree
from estimagic.parameters.tree_registry import get_registry
from pybaum import leaf_names
from pybaum import tree_flatten
from pybaum import tree_just_flatten
from pybaum import tree_unflatten


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
            statistics, the covariance matrix, and estimated boostrap outcomes.
    """
    check_inputs(data, cluster_by)

    if outcome_kwargs is not None:
        outcome = functools.partial(outcome, **outcome_kwargs)

    estimates = get_bootstrap_outcomes(
        data=data,
        outcome=outcome,
        cluster_by=cluster_by,
        seed=seed,
        n_draws=n_draws,
        n_cores=n_cores,
        error_handling=error_handling,
        batch_evaluator=batch_evaluator,
    )

    base_outcomes = outcome(data)

    out = bootstrap_from_outcomes(data, base_outcomes, estimates)

    return out


def bootstrap_from_outcomes(data, base_outcomes, bootstrap_outcomes):
    """Create BootstrapResults object.

    Args:
        data (pandas.DataFrame): original dataset.
        base_outcomes (pytree): Pytree of the base outomes, i.e. the outcomes
            evaluated on the original data set.
        bootstrap_outcomes (list or pytree): List of pytrees or pytree of estimated
            bootstrap outcomes.

    Returns:
        BootstrapResult: A BootstrapResult object storing information on summary
            statistics, the covariance matrix, and the estimated boostrap outcomes.
    """
    check_inputs(data)

    if isinstance(bootstrap_outcomes, list):  # Write test cases on this!
        registry = get_registry(extended=True)

        flat_outcomes = [
            tree_just_flatten(est, registry=registry) for est in bootstrap_outcomes
        ]
        internal_outcomes = np.array(flat_outcomes)
    elif isinstance(bootstrap_outcomes, dict):
        internal_outcomes = np.array(list(bootstrap_outcomes.values()))
    else:
        internal_outcomes = np.array(bootstrap_outcomes)

    out = BootstrapResult(
        data=data,
        base_outcomes=base_outcomes,
        bootstrap_outcomes=bootstrap_outcomes,
        _internal_outcomes=internal_outcomes,
    )  # write more tests on retrieval

    return out


@dataclass
class BootstrapResult:
    data: pd.DataFrame
    base_outcomes: Any  # rename to params?
    bootstrap_outcomes: Any
    _internal_outcomes: np.ndarray

    @property
    def _outcomes(self):
        return self.outcomes()

    @property
    def _summary(self):
        return self.summary()

    @property
    def _se(self):
        return self.se()

    @property
    def _cov(self):
        return self.cov()

    def outcomes(self, return_type="pytree"):
        """Returns the estimated bootstrap outcomes.

        Args:
            return_type (str): One of "array", "dataframe" or "pytree". Default pytree.
                If your bootstrap outcomes have a very nested format, return_type
                "dataframe" might be the better choice.

        Returns:
            Any: The boostrap outcomes as a pytree, numpy array or DataFrame.
        """
        if return_type == "array":
            out = self._internal_outcomes
        elif return_type == "data_frame":
            registry = get_registry(extended=True)
            leafnames = leaf_names(self.base_outcomes, registry=registry)
            free_index = np.array(leafnames)
            out = pd.DataFrame(data=self._internal_outcomes, columns=free_index)
        elif return_type == "pytree":
            out = self.bootstrap_outcomes

        return out

    def se(self):
        """Calculate standard errors.

        Returns:
            Any: A pytree with the same structure as base_outcomes containing standard
                errors for the parameter estimates.
        """
        free_cov = np.cov(self._internal_outcomes, rowvar=False)
        helper = np.sqrt(np.diagonal(free_cov))

        registry = get_registry(extended=True)
        _, treedef = tree_flatten(self.base_outcomes, registry=registry)

        out = tree_unflatten(treedef, helper, registry=registry)

        return out

    def cov(self, return_type="pytree"):
        """Calculate the variance-covariance matrix of the estimated parameters.

        Args:
            return_type (str): One of "pytree", "array" or "dataframe". Default pytree.
                If "array", a 2d numpy array with the covariance is returned. If
                "dataframe", a pandas DataFrame with parameter names in the
                index and columns are returned.

        Returns:
            Any: The covariance matrix of the estimated parameters as a block-pytree,
                numpy array, or pandas DataFrame.
        """
        free_cov = np.cov(self._internal_outcomes, rowvar=False)

        if return_type == "array":
            out = free_cov
        elif return_type == "dataframe":
            registry = get_registry(extended=True)
            leafnames = leaf_names(self.base_outcomes, registry=registry)
            free_index = np.array(leafnames)
            out = pd.DataFrame(data=free_cov, columns=free_index, index=free_index)
        elif return_type == "pytree":
            out = matrix_to_block_tree(free_cov, self.base_outcomes, self.base_outcomes)

        return out

    def summary(self, ci_method="percentile", alpha=0.05):
        """Create a summary of bootstrap results.

        Args:
            ci_method (str): method of choice for confidence interval computation.
            alpha (float): significance level of choice.

        Returns:
            pd.DataFrame: The estimation summary as a DataFrame containing information
                on the mean, standard errors, as well as the confindence interval.
                Soon this will be a pytree of DataFrames.
        """
        check_inputs(ci_method=ci_method, alpha=alpha)

        cis = compute_ci(
            self.base_outcomes,
            self.bootstrap_outcomes,
            ci_method,
            alpha,
        )

        summary = pd.DataFrame(self.bootstrap_outcomes.mean(axis=0), columns=["mean"])
        summary["std"] = self.bootstrap_outcomes.std(axis=0)

        summary["lower_ci"] = cis["lower_ci"]
        summary["upper_ci"] = cis["upper_ci"]

        return summary
