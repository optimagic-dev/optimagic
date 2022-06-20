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
        data (pandas.DataFrame): Original dataset.
        outcome (callable): Function of the data calculating statistic of interest.
            Returns a general pytree (e.g. pandas Series, dict, numpy array, etc.).
        outcome_kwargs (dict): Additional keyword arguments for outcome.
        n_draws (int): number of bootstrap samples to draw.
        cluster_by (str): Column name of variable to cluster by or None.
        seeds (numpy.array): Array of seeds for bootstrap samples, default is none.
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
        base_outcome (pytree): Pytree of base outcomes, i.e. the outcome
            statistic(s) evaluated on the original data set.
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
        _base_outcome=base_outcome,
        _internal_outcomes=internal_outcomes,
    )

    return out


@dataclass
class BootstrapResult:
    _base_outcome: Any
    _internal_outcomes: np.ndarray

    @property
    def _outcomes(self):
        return self.outcomes()

    @property
    def _se(self):
        return self.se()

    @property
    def _cov(self):
        return self.cov()

    @property
    def _ci(self):
        return self.ci()

    @property
    def _p_values(self):
        return self.p_values()

    @property
    def _summary(self):
        return self.summary()

    @property
    def base_outcome(self):
        """Returns the base outcome statistic(s).

        Returns:
            pytree: Pytree of base outcomes, i.e. the outcome statistic(s) evaluated
                on the original data set.
        """
        return self._base_outcome

    def outcomes(self):
        """Returns the estimated bootstrap outcomes.

        Returns:
            Any: The boostrap outcomes as a list of pytrees.
        """
        registry = get_registry(extended=True)
        _, treedef = tree_flatten(self._base_outcome, registry=registry)

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
        cov = np.cov(self._internal_outcomes, rowvar=False)
        se = np.sqrt(np.diagonal(cov))

        registry = get_registry(extended=True)
        _, treedef = tree_flatten(self._base_outcome, registry=registry)

        out = tree_unflatten(treedef, se, registry=registry)

        return out

    def cov(self, return_type="pytree"):
        """Calculate the variance-covariance matrix of the estimated parameters.

        Args:
            return_type (str): One of "pytree", "array" or "dataframe". Default pytree.
                If "array", a 2d numpy array with the covariance is returned. If
                "dataframe", a pandas DataFrame with parameter names in the
                index and columns are returned.
                The default is "pytree".

        Returns:
            Any: The covariance matrix of the estimated parameters as a block-pytree,
                numpy array, or pandas DataFrame.
        """
        cov = np.cov(self._internal_outcomes, rowvar=False)

        if return_type == "array":
            out = cov
        elif return_type == "dataframe":
            registry = get_registry(extended=True)
            leafnames = leaf_names(self._base_outcome, registry=registry)
            free_index = np.array(leafnames)
            out = pd.DataFrame(data=cov, columns=free_index, index=free_index)
        elif return_type == "pytree":
            out = matrix_to_block_tree(cov, self._base_outcome, self._base_outcome)
        else:
            raise TypeError(
                "return_type must be one of pytree, array, or dataframe, "
                f"not {return_type}."
            )

        return out

    def ci(self, ci_method="percentile", ci_level=0.95):
        """Calculate confidence intervals.

        Args:
            ci_method (str): Method of choice for confidence interval computation.
                The default is "percentile".
            ci_level (float): Confidence level for the calculation of confidence
                intervals. The default is 0.95.

        Returns:
            Any: Pytree with the same structure as base_outcome containing lower
                bounds of confidence intervals.
            Any: Pytree with the same structure as base_outcome containing upper
                bounds of confidence intervals.
        """
        registry = get_registry(extended=True)
        base_outcome_flat, treedef = tree_flatten(self._base_outcome, registry=registry)

        lower_flat, upper_flat = compute_ci(
            base_outcome_flat, self._internal_outcomes, ci_method, ci_level
        )

        lower = tree_unflatten(treedef, lower_flat, registry=registry)
        upper = tree_unflatten(treedef, upper_flat, registry=registry)

        return lower, upper

    def p_values(self):
        """Calculate p-values.

        Returns:
            Any: A pytree with the same structure as base_outcome containing p-values
                for the parameter estimates.
        """
        raise NotImplementedError("Bootstrapped p-values are not implemented yet.")

    def summary(self, ci_method="percentile", ci_level=0.95):
        """Create a summary of bootstrap results.

        Args:
            ci_method (str): Method of choice for confidence interval computation.
                The default is "percentile".
            ci_level (float): Confidence level for the calculation of confidence
                intervals. The default is 0.95.

        Returns:
            pd.DataFrame: The estimation summary as a DataFrame containing information
                on the mean, standard errors, as well as the confidence intervals.
                Soon this will be a pytree.
        """
        raise NotImplementedError("summary is not implemented yet.")
