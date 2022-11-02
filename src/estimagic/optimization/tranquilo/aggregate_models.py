from functools import partial

import numpy as np
from estimagic.optimization.tranquilo.models import ScalarModel


def get_aggregator(aggregator, functype, model_info):
    """Get a function that aggregates a VectorModel into a ScalarModel.

    Args:
        aggregator (str or callable): Name of an aggregator or aggregator function.
            The function must take as argument:
            - vector_model (VectorModel): A fitted vector model.
        functype (str): One of "scalar", "least_squares" and "likelihood".
        model_info (ModelInfo): Information that describes the functional form of
            the model.

    Returns:
        callable: The partialled aggregator that only depends on vector_model.

    """
    built_in_aggregators = {
        "identity": aggregator_identity,
        "sum": aggregator_sum,
        "information_equality_linear": aggregator_information_equality_linear,
        "least_squares_linear": aggregator_least_squares_linear,
    }

    if isinstance(aggregator, str) and aggregator in built_in_aggregators:
        _aggregator = built_in_aggregators[aggregator]
        _aggregator_name = aggregator
        _using_built_in_aggregator = True
    elif callable(aggregator):
        _aggregator = aggregator
        _aggregator_name = getattr(aggregator, "__name__", "your aggregator")
        _using_built_in_aggregator = False
    else:
        raise ValueError(
            "Invalid aggregator: {aggregator}. Must be one of "
            f"{list(built_in_aggregators)} or a callable."
        )

    # determine if aggregator is compatible with functype and model_info
    aggregator_compatible_with_functype = {
        "scalar": ("identity", "sum"),
        "least_squares": ("least_squares_linear",),
        "likelihood": (
            "sum",
            "information_equality_linear",
        ),
    }

    aggregator_compatible_with_model_info = {
        # keys are names of aggregators and values are functions of model_info that
        # return False in case of incompatibility
        "identity": lambda x: True,
        "sum": _is_second_order_model,
        "information_equality_linear": lambda model_info: not _is_second_order_model(
            model_info
        ),
        "least_squares_linear": lambda model_info: not _is_second_order_model(
            model_info
        ),
    }

    if _using_built_in_aggregator:
        # compatibility errors
        if _aggregator_name not in aggregator_compatible_with_functype[functype]:
            raise ValueError(
                f"Aggregator {_aggregator_name} is not compatible with functype "
                f"{functype}. It would not produce a quadratic main model."
            )
        if not aggregator_compatible_with_model_info[_aggregator_name](model_info):
            raise ValueError(
                f"ModelInfo {model_info} is not compatible with aggregator "
                f"{_aggregator_name}. Depending on the aggregator this may be because "
                "it would not produce a quadratic main model or that the aggregator "
                "requires a different residual model for theoretical reasons."
            )

    # create aggregator
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
        intercept=intercept, linear_terms=linear_terms, square_terms=square_terms
    )
    return scalar_model


def aggregator_identity(vector_model):
    """Aggregate quadratic VectorModel using identity function.

    This aggregation is useful if the underlying maximization problem is a scalar
    problem. To get a second-order main model vector_model must be second-order model.

    Assumptions
    -----------
    1. functype: scalar
    2. ModelInfo: has squares or interactions

    """
    intercept = float(vector_model.intercepts)
    linear_terms = np.squeeze(vector_model.linear_terms)
    if vector_model.square_terms is None:
        square_terms = np.zeros((len(linear_terms), len(linear_terms)))
    else:
        square_terms = np.squeeze(vector_model.square_terms)
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
    2. ModelInfo: has squares or interactions

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
    2. ModelInfo: has intercept but no squares and no interaction

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
    2. ModelInfo: has no squares and no interaction

    """
    vm_linear_terms = vector_model.linear_terms
    vm_intercepts = vector_model.intercepts

    fisher_information = vm_linear_terms.T @ vm_linear_terms

    intercept = vm_intercepts.sum(axis=0)
    linear_terms = vm_linear_terms.sum(axis=0)
    square_terms = -fisher_information / 2

    return intercept, linear_terms, square_terms


def _is_second_order_model(model_info):
    return model_info.has_squares or model_info.has_interactions
