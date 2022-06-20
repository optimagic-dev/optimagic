import functools
from dataclasses import dataclass
from functools import cached_property
from typing import Any

import numpy as np
import pandas as pd
from estimagic.batch_evaluators import joblib_batch_evaluator
from estimagic.inference.bootstrap_ci import calculate_ci
from estimagic.inference.bootstrap_helpers import check_inputs
from estimagic.inference.bootstrap_outcomes import get_bootstrap_outcomes
from estimagic.parameters.block_trees import matrix_to_block_tree
from estimagic.parameters.tree_registry import get_registry
from pybaum import leaf_names
from pybaum import tree_flatten
from pybaum import tree_just_flatten
from pybaum import tree_unflatten


def bootstrap(
    outcome,
    *,
    data=None,
    existing_outcomes=None,
    outcome_kwargs=None,
    n_draws=1_000,
    cluster_by=None,
    seed=None,
    n_cores=1,
    error_handling="continue",
    batch_evaluator=joblib_batch_evaluator,
):
    """Use the bootstrap to calculate inference quantities.

    Args:
        outcome (Union[callable, Any]): Either a function that computes the statistic
            of interest, or an evaluation of that function. If it is an evaluation,
            existing_outcomes must be passed, and the evaluation step is skipped.
        data (pd.DataFrame): Dataset. Default None.
        existing_outcomes (List[Any]): Evaluations of the outcome function. If None,
            then outcome must be callable and new outcomes are generated. If not None
            and outcome is callable, the new evaluations are appended to the existing
            ones. Default None.
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
    if existing_outcomes is None:
        existing_outcomes = []
    elif not isinstance(existing_outcomes, list):
        raise ValueError("existing_outcomes must be a list.")

    if callable(outcome):

        check_inputs(data=data, cluster_by=cluster_by)

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
        existing_outcomes += bootstrap_outcomes
        base_outcome = outcome(data)
    else:
        base_outcome = outcome

    # ==================================================================================
    # Process results
    # ==================================================================================

    registry = get_registry(extended=True)
    flat_outcomes = [tree_just_flatten(e, registry=registry) for e in existing_outcomes]
    internal_outcomes = np.array(flat_outcomes)

    result = BootstrapResult(
        _base_outcome=base_outcome,
        _internal_outcomes=internal_outcomes,
    )
    return result


@dataclass
class BootstrapResult:
    _base_outcome: Any
    _internal_outcomes: np.ndarray

    @cached_property
    def _se(self):
        return self.se()

    @cached_property
    def _cov(self):
        return self.cov()

    @cached_property
    def _ci(self):
        return self.ci()

    @cached_property
    def _p_values(self):
        return self.p_values()

    @cached_property
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

    @cached_property
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
            Any: The standard errors of the estimated parameters as a block-pytree,
                numpy.ndarray, or pandas.DataFrame.
        """
        cov = np.cov(self._internal_outcomes, rowvar=False)
        se = np.sqrt(np.diagonal(cov))

        registry = get_registry(extended=True)
        _, treedef = tree_flatten(self._base_outcome, registry=registry)

        se = tree_unflatten(treedef, se, registry=registry)
        return se

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
                numpy.ndarray, or pandas.DataFrame.
        """
        cov = np.cov(self._internal_outcomes, rowvar=False)

        if return_type == "dataframe":
            registry = get_registry(extended=True)
            names = np.array(leaf_names(self._base_outcome, registry=registry))
            cov = pd.DataFrame(cov, columns=names, index=names)
        elif return_type == "pytree":
            cov = matrix_to_block_tree(cov, self._base_outcome, self._base_outcome)
        elif return_type != "array":
            raise ValueError(
                "return_type must be one of pytree, array, or dataframe, "
                f"not {return_type}."
            )
        return cov

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

        lower_flat, upper_flat = calculate_ci(
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
        msg = (
            "Bootstrap p-values are not implemented yet, due to missing p-values. You"
            " can still view the confidence interval through the method `ci()`."
        )
        raise NotImplementedError(msg)

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
        msg = (
            "Bootstrap summary is not implemented yet, due to missing p-values. You"
            " can still view the confidence interval through the method `ci()`."
        )
        raise NotImplementedError(msg)
