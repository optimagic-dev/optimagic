import warnings
from functools import partial

import numpy as np
from estimagic.optimization.tranquilo.models import ScalarModel


def get_aggregator(aggregator, functype, model_info):
    """Get a function that aggregates a VectorModel into a ScalarModel.

    Args:
        aggregator (str or callable): Name of an aggregator or aggregator function.
            The function must take as arguments (in that order):
            - vector_model (VectorModel): A fitted vector model.
            - fvec_center (np.ndarray): A 1d array of the residuals at the center of the
            trust-region. In the noisy case, this may be an average.
            - model_info (ModelInfo): The model information.
        functype (str): One of "scalar", "least_squares" and "likelihood".
        model_info (ModelInfo): Information that describes the functional form of
            the model.

    Returns:
        callable: The partialled aggregator that only depends on vector_model and
            fvec_center.

    """
    built_in_aggregators = {
        "identity": aggregator_identity,
        "sum": aggregator_sum,
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
        "least_squares": (
            "least_squares_linear",
            "least_squares_linear_taylor",
            "least_squares_quadratic",
        ),
        "likelihood": (
            "sum",
            "sum_taylor",
            "information_equality_linear",
            "information_equality_quadratic",
        ),
    }

    aggregator_compatible_with_model_info = {
        # keys are names of aggregators and values are functions of model_info that
        # return False in case of incompatibility
        "identity": _is_second_order_model,
        "sum": _is_second_order_model,
        "sum_taylor": _is_second_order_model,
        "information_equality_linear": lambda model_info: True,  # all models allowed
        "information_equality_quadratic": _is_second_order_model,
        "least_squares_linear": lambda model_info: True,  # all models allowed
        "least_squares_linear_taylor": lambda model_info: True,  # all models allowed
        "least_squares_quadratic": _is_second_order_model,
    }

    if _using_built_in_aggregator:
        # compatibility errors
        if _aggregator_name not in aggregator_compatible_with_functype[functype]:
            ValueError(
                f"Aggregator {_aggregator_name} is not compatible with functype "
                f"{functype}. It would not produce a quadratic main model."
            )
        if functype == "scalar" and not _is_second_order_model(model_info):
            ValueError(
                f"ModelInfo {model_info} is not compatible with functype scalar. "
                "It would not produce a quadratic main model."
            )
        if not aggregator_compatible_with_model_info[_aggregator_name](model_info):
            ValueError(
                f"ModelInfo {model_info} is not compatible with aggregator "
                f"{_aggregator_name}. It would not produce a quadratic main model."
            )
        # inefficiency warnings
        if _is_second_order_model(model_info) and "linear" in _aggregator_name:
            suggestions = [
                aggregator_name
                for aggregator_name in aggregator_compatible_with_functype[functype]
                if "quadratic" in aggregator_name
            ]
            warnings.warn(
                "The residual model is calculating second-order information, "
                f"but the aggregator {_aggregator_name} is not using it. Consider "
                f"switching to {suggestions}."
            )

    # create aggregator
    out = partial(
        _aggregate_models_template, aggregator=_aggregator, model_info=model_info
    )
    return out


def _aggregate_models_template(vector_model, fvec_center, aggregator, model_info):
    """Aggregate a VectorModel into a ScalarModel.

    Note on fvec_center:
    --------------------
    Let x0 be the x-value at which the x-sample is centered. If there is little noise
    and the criterion function f is evaluated at x0, then fvec_center = f(x0). If,
    however, the criterion function is very noisy or only evaluated in a neighborhood
    around x0, then fvec_center is constructed as an average over evaluations of f
    with x close to x0.

    Args:
        vector_model (VectorModel): The VectorModel to aggregate.
        fvec_center (np.ndarray): A 1d array of the residuals at the center of the
            trust-region. In the noisy case, this may be an average.
        aggregator (callable): The function that does the actual aggregation.
        model_info (ModelInfo): Information that describes the functional form of
            the model.

    Returns:
        ScalarModel: The aggregated model

    """
    intercept, linear_terms, square_terms = aggregator(
        vector_model, fvec_center, model_info
    )
    scalar_model = ScalarModel(
        intercept=intercept, linear_terms=linear_terms, square_terms=square_terms
    )
    return scalar_model


def aggregator_identity(vector_model, fvec_center, model_info):
    """Aggregate quadratic VectorModel using identity function.

    This aggregation is useful if the underlying maximization problem is a scalar
    problem.

    """
    intercept = float(fvec_center)
    linear_terms = np.squeeze(vector_model.linear_terms)
    if _is_second_order_model(model_info):
        square_terms = np.squeeze(vector_model.square_terms)
    else:
        square_terms = None
    return intercept, linear_terms, square_terms


def aggregator_sum(vector_model, fvec_center, model_info):
    """Aggregate quadratic VectorModel using sum function.

    This aggregation is useful if the underlying maximization problem is a likelihood
    problem. That is, the criterion is tje sum of residuals, which allows us to sum
    up the coefficients of the residual model to get a main model. The main model will
    only be a second-order model if the residual model is a second-order model.

    """
    intercept = fvec_center.sum()
    linear_terms = vector_model.linear_terms.sum(axis=0)
    if _is_second_order_model(model_info):
        square_terms = vector_model.square_terms.sum(axis=0)
    else:
        square_terms = None
    return intercept, linear_terms, square_terms


def _is_second_order_model(model_info):
    return model_info.has_squares or model_info.has_interactions
