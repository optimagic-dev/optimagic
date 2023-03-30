from functools import partial

import numpy as np

from estimagic.optimization.tranquilo.models import ScalarModel


def get_aggregator(aggregator):
    """Get a function that aggregates a VectorModel into a ScalarModel.

    Args:
        aggregator (str): Name of an aggregator.

    Returns:
        callable: The partialled aggregator that only depends on vector_model.

    """
    built_in_aggregators = {
        "identity": aggregator_identity,
        "sum": aggregator_sum,
        "information_equality_linear": aggregator_information_equality_linear,
        "least_squares_linear": aggregator_least_squares_linear,
    }

    if aggregator in built_in_aggregators:
        _aggregator = built_in_aggregators[aggregator]
    else:
        raise ValueError(
            f"Invalid aggregator: {aggregator}. Must be one of "
            f"{list(built_in_aggregators)} or a callable."
        )

    out = partial(_aggregate_models_template, aggregator=_aggregator)
    return out


def _aggregate_models_template(vector_model, aggregator):
    """Aggregate a VectorModel into a ScalarModel.

    Args:
        vector_model (VectorModel): The VectorModel to aggregate.
        aggregator (callable): The function that does the actual aggregation.

    Returns:
        ScalarModel: The aggregated model

    """
    intercept, linear_terms, square_terms = aggregator(vector_model)
    scalar_model = ScalarModel(
        intercept=intercept,
        linear_terms=linear_terms,
        square_terms=square_terms,
        shift=vector_model.shift,
        scale=vector_model.scale,
    )
    return scalar_model


def aggregator_identity(vector_model):
    """Aggregate quadratic VectorModel using identity function.

    This aggregation is useful if the underlying maximization problem is a scalar
    problem. To get a second-order main model vector_model must be second-order model.

    Assumptions
    -----------
    1. functype: scalar
    2. model_type: quadratic

    """
    n_params = vector_model.linear_terms.size
    intercept = float(vector_model.intercepts)
    linear_terms = vector_model.linear_terms.flatten()
    if vector_model.square_terms is None:
        square_terms = np.zeros((n_params, n_params))
    else:
        square_terms = vector_model.square_terms.reshape(n_params, n_params)
    return intercept, linear_terms, square_terms


def aggregator_sum(vector_model):
    """Aggregate quadratic VectorModel using sum function.

    This aggregation is useful if the underlying maximization problem is a likelihood
    problem. That is, the criterion is the sum of residuals, which allows us to sum
    up the coefficients of the residual model to get the main model. The main model will
    only be a second-order model if the residual model is a second-order model.

    Assumptions
    -----------
    1. functype: likelihood
    2. model_type: quadratic

    """
    vm_intercepts = vector_model.intercepts
    intercept = vm_intercepts.sum(axis=0)
    linear_terms = vector_model.linear_terms.sum(axis=0)
    square_terms = vector_model.square_terms.sum(axis=0)
    return intercept, linear_terms, square_terms


def aggregator_least_squares_linear(vector_model):
    """Aggregate linear VectorModel assuming a least_squares functype.

    This aggregation is useful if the underlying maximization problem is a least-squares
    problem. We can then simply plug-in a linear model for the residuals into the
    least-squares formulae to get a second-order main model.

    Assumptions
    -----------
    1. functype: least_squares
    2. model_type: linear

    References
    ----------
    See section 2.1 of :cite:`Cartis2018` for further information.

    """
    vm_linear_terms = vector_model.linear_terms
    vm_intercepts = vector_model.intercepts

    intercept = vm_intercepts @ vm_intercepts
    linear_terms = 2 * np.sum(vm_linear_terms * vm_intercepts.reshape(-1, 1), axis=0)
    square_terms = 2 * vm_linear_terms.T @ vm_linear_terms

    return intercept, linear_terms, square_terms


def aggregator_information_equality_linear(vector_model):
    """Aggregate linear VectorModel using the Fisher information equality.

    This aggregation is useful if the underlying maximization problem is a likelihood
    problem. Given a linear model for the likelihood contributions we get an estimate of
    the scores. Using the Fisher-Information-Equality we estimate the average Hessian
    using the scores.

    Assumptions
    -----------
    1. functype: likelihood
    2. model_type: linear

    """
    vm_linear_terms = vector_model.linear_terms
    vm_intercepts = vector_model.intercepts

    fisher_information = vm_linear_terms.T @ vm_linear_terms

    intercept = vm_intercepts.sum(axis=0)
    linear_terms = vm_linear_terms.sum(axis=0)
    square_terms = -fisher_information / 2

    return intercept, linear_terms, square_terms
