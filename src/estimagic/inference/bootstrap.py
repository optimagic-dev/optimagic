import functools
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from estimagic.batch_evaluators import joblib_batch_evaluator
from estimagic.inference.bootstrap_ci import compute_ci
from estimagic.inference.bootstrap_ci import compute_p_values
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

    bootstrap_outcomes = get_bootstrap_outcomes(
        data=data,
        outcome=outcome,
        cluster_by=cluster_by,
        seed=seed,
        n_draws=n_draws,
        n_cores=n_cores,
        error_handling=error_handling,
        batch_evaluator=batch_evaluator,
    )

    base_outcome = outcome(data)
    out = bootstrap_from_outcomes(base_outcome, bootstrap_outcomes)

    return out


def bootstrap_from_outcomes(base_outcome, bootstrap_outcomes):
    """Create BootstrapResults object.

    Args:
        base_outcome (pytree): Pytree of base outcomes, i.e. the outcome statistics
            evaluated on the original data set.
        bootstrap_outcomes (list): List of pytrees of estimated
            bootstrap outcomes.

    Returns:
        BootstrapResult: A BootstrapResult object storing information on summary
            statistics, the covariance matrix, and the estimated boostrap outcomes.
    """
    if isinstance(bootstrap_outcomes, list):
        registry = get_registry(extended=True)

        flat_outcomes = [
            tree_just_flatten(est, registry=registry) for est in bootstrap_outcomes
        ]
        internal_outcomes = np.array(flat_outcomes)
    else:
        raise TypeError(
            "bootstrap_outcomes must be a list of pytrees, "
            f"not {type(bootstrap_outcomes)}."
        )

    out = BootstrapResult(
        base_outcome=base_outcome,
        _internal_outcomes=internal_outcomes,
    )

    return out


@dataclass
class BootstrapResult:
    base_outcome: Any
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

    def outcomes(self):
        """Returns the estimated bootstrap outcomes.

        Returns:
            Any: The boostrap outcomes as a list of pytrees.
        """
        registry = get_registry(extended=True)
        _, treedef = tree_flatten(self.base_outcome, registry=registry)

        outcomes = [
            tree_unflatten(treedef, out, registry=registry)
            for out in self._internal_outcomes
        ]

        return outcomes

    def se(self):
        """Calculate standard errors.

        Returns:
            list: A list of pytrees containing standard errors for the bootstrapped
                statistic.
        """
        free_cov = np.cov(self._internal_outcomes, rowvar=False)
        free_se = np.sqrt(np.diagonal(free_cov))

        registry = get_registry(extended=True)
        _, treedef = tree_flatten(self.base_outcome, registry=registry)

        out = tree_unflatten(treedef, free_se, registry=registry)

        return out

    def p_values(self, alpha=0.05):
        """Calculate p-values.

        Args:
            alpha (float): Significance level of choice.

        Returns:
            Any: A pytree with the same structure as base_outcomes containing p-values
                for the parameter estimates.
        """
        registry = get_registry(extended=True)
        base_outcome_flat, treedef = tree_flatten(self.base_outcome, registry=registry)

        free_p_values = compute_p_values(
            base_outcome_flat, self._internal_outcomes, alpha
        )
        out = tree_unflatten(treedef, free_p_values, registry=registry)

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
            leafnames = leaf_names(self.base_outcome, registry=registry)
            free_index = np.array(leafnames)
            out = pd.DataFrame(data=free_cov, columns=free_index, index=free_index)
        elif return_type == "pytree":
            out = matrix_to_block_tree(free_cov, self.base_outcome, self.base_outcome)
        else:
            raise TypeError(
                "return_type must be one of pytree, array, or dataframe, "
                f"not {return_type}."
            )

        return out

    def ci(self, ci_method="percentile", alpha=0.05):
        """Calculate confidence intervals.

        Args:
            ci_method (str): method of choice for confidence interval computation.
            alpha (float): Significance level of choice.

        Returns:
            Any: Pytree with the same structure as base_outcomes containing lower
                bounds of confidence intervals.
            Any: Pytree with the same structure as base_outcomes containing upper
                bounds of confidence intervals.
        """
        check_inputs(ci_method=ci_method, alpha=alpha)

        registry = get_registry(extended=True)
        base_outcome_flat, treedef = tree_flatten(self.base_outcome, registry=registry)

        lower_flat, upper_flat = compute_ci(
            base_outcome_flat, self._internal_outcomes, ci_method, alpha
        )

        lower = tree_unflatten(treedef, lower_flat, registry=registry)
        upper = tree_unflatten(treedef, upper_flat, registry=registry)

        return lower, upper

    def summary(self, ci_method="percentile", alpha=0.05):
        """Create a summary of bootstrap results.

        Args:
            ci_method (str): method of choice for confidence interval computation.
            alpha (float): Significance level of choice.

        Returns:
            pd.DataFrame: The estimation summary as a DataFrame containing information
                on the mean, standard errors, as well as the confidence intervals.
                Soon this will be a pytree.
        """
        check_inputs(ci_method=ci_method, alpha=alpha)

        lower, upper = compute_ci(
            self.base_outcome, self._outcomes_internal, ci_method, alpha
        )

        cis = pd.DataFrame(
            np.stack([lower, upper], axis=1),
            columns=["lower_ci", "upper_ci"],
        )

        summary = pd.DataFrame(
            np.mean(self._internal_outcomes, axis=0), columns=["mean"]
        )
        summary["std"] = np.std(self._internal_outcomes, axis=0)

        summary["lower_ci"] = cis["lower_ci"]
        summary["upper_ci"] = cis["upper_ci"]

        return summary
