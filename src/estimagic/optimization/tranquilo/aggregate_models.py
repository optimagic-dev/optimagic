import warnings
from functools import partial

import numpy as np
from estimagic.optimization.tranquilo.models import ScalarModel


def get_aggregator(aggregator, functype, model_info):
    """Get a function that aggregates a VectorModel into a ScalarModel.

    Args:
        aggregator (str or callable): Name of an aggregator or aggregator function.
            The function must take vector_model, residuals.
        functype (str): One of "scalar", "least_squares" and "likelihood".
        model_info (ModelInfo): Information that describes the functional form of
            the model.

    Returns:
        callable: The partialled aggregator that only depends on vector_model and
            residuals

    """
    built_in_aggregators = {
        "identity": aggregator_identity,
        "sum": aggregator_sum,
        "information_equality_linear": aggregator_information_equality_linear,
        "information_equality_quadratic": aggregator_information_equality_quadratic,
        "least_squares_linear": aggregator_least_squares_linear,
        "least_squares_quadratic": aggregator_least_squares_quadratic,
    }

    if isinstance(aggregator, str) and aggregator in built_in_aggregators:
        _aggregator = built_in_aggregators[aggregator]
        _aggregator_name = aggregator
        _using_built_in_aggregator = True
    elif callable(aggregator):
        _aggregator = aggregator
        _aggregator_name = getattr(aggregator, "__name__", "your aggregator")
    else:
        raise ValueError(
            "Invalid aggregator: {aggregator}. Must be one of "
            f"{list(built_in_aggregators)} or a callable."
        )

    # determine if aggregator is compatible with functype and model_info
    aggregator_compatible_with_functype = {
        "scalar": ("identity", "sum"),
        "least_squares": ("least_squares_linear", "least_squares_quadratic"),
        "likelihood": (
            "sum",
            "information_equality_linear",
            "information_equality_quadratic",
        ),
    }

    def _has_squares_or_interactions(model_info):
        return model_info.has_squares or model_info.has_interactions

    aggregator_compatible_with_model_info = {
        "identity": _has_squares_or_interactions,
        "sum": _has_squares_or_interactions,
        "information_equality_linear": lambda model_info: True,  # all models allowed
        "information_equality_quadratic": _has_squares_or_interactions,
        "least_squares_linear": lambda model_info: True,  # all models allowed
        "least_squares_quadratic": _has_squares_or_interactions,
    }

    if _using_built_in_aggregator:
        # compatibility errors
        if _aggregator_name not in aggregator_compatible_with_functype[functype]:
            ValueError(
                f"Aggregator {_aggregator_name} is not compatible with functype "
                f"{functype}. It would not produce a quadratic main model."
            )
        if functype == "scalar" and (
            not model_info.has_squares and not model_info.has_interactions
        ):
            ValueError(
                f"ModelInfo {model_info} is not compatible with functype {functype}. "
                "It would not produce a quadratic main model."
            )
        if not aggregator_compatible_with_model_info[_aggregator_name](model_info):
            ValueError(
                f"ModelInfo {model_info} is not compatible with aggregator "
                f"{_aggregator_name}. It would not produce a quadratic main model."
            )
        # inefficiency warnings
        if _has_squares_or_interactions(model_info) and "linear" in _aggregator_name:
            suggestions = [
                aggr
                for aggr in aggregator_compatible_with_functype[functype]
                if "quadratic" in aggr
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


def _aggregate_models_template(vector_model, residuals, aggregator, model_info):
    """Aggregate a VectorModel into a ScalarModel.

    Let x0 be the x-value at which the x-sample is centered. Then the residuals are
    defined by r(x0) = [r_1(x0), ..., r_m(x0)].

    Args:
        vector_model (VectorModel): The VectorModel to aggregate.
        residuals (np.ndarray): The residuals on which the vector model was fit. A 1d
            array of length n_residuals.
        aggregator (callable): The function that does the actual aggregation.
        model_info (ModelInfo): Information that describes the functional form of
            the model.

    Returns:
        ScalarModel: The aggregated model

    """
    intercept, linear_terms, square_terms = aggregator(
        vector_model, residuals, model_info
    )
    scalar_model = ScalarModel(intercept, linear_terms, square_terms)
    return scalar_model


def aggregator_identity(vector_model, residuals, model_info):
    """Aggregate quadratic VectorModel using identity function.

    Here we assume that vector_model is actually a ScalarModel with an unused dimension.

    """
    coefficients = (
        np.squeeze(coef) if coef is not None else None for coef in vector_model
    )
    return coefficients


def aggregator_sum(vector_model, residuals, model_info):
    """Aggregate quadratic VectorModel using sum function.

    Here we assume that vector_model is a quadratic model, which allows us to simply
    sum the coefficients over the residuals.

    """
    coefficients = (
        np.sum(coef, axis=0) if coef is not None else None for coef in vector_model
    )
    return coefficients


def aggregator_information_equality_linear(vector_model, residuals, model_info):
    """Aggregate linear VectorModel using the Fisher information equality.

    Here we assume that vector_model is a linear model. This implies that the estimated
    linear_terms correspond to a gradient estimate of the full model. By the information
    equality we can estimate the Hessian of the full model using the gradient. To get
    the coefficients for the quadratic main model we transform the gradient and Hessian.

    """
    if model_info.has_intercept:
        intercept = np.sum(vector_model.intercepts, axis=0)
    else:
        intercept = None

    vm_linear_terms = vector_model.linear_terms

    gradient = np.sum(vm_linear_terms, axis=0)
    hessian = -(vm_linear_terms.T @ vm_linear_terms) / len(vm_linear_terms)

    square_terms = hessian / 2
    linear_terms = gradient - hessian @ residuals

    return intercept, linear_terms, square_terms


def aggregator_information_equality_quadratic(vector_model, residuals, model_info):
    """Aggregate quadratic VectorModel using the Fisher information equality.

    Here we assume that vector_model is a quadratic model, that is, both square and
    interaction terms were fitted. To utilize the Fisher information equality we
    transform the coefficients to gradient and Hessian. We then get a second Hessian
    estimate using the Fisher inequality. The final Hessian estimate is an average
    of the two. Lastly we retransform gradient and Hessian to coefficients for the
    quadratic main model.

    """
    if model_info.has_intercept:
        intercept = np.sum(vector_model.intercepts, axis=0)
    else:
        intercept = None

    vm_linear_terms = vector_model.linear_terms
    vm_square_terms = vector_model.square_terms

    residual_gradients = (
        vm_linear_terms
        + (vm_square_terms + vm_square_terms.tranpose(0, 2, 1)) @ residuals
    )

    gradient = np.sum(vm_linear_terms, axis=0)

    hessian_one = -(residual_gradients.T @ residual_gradients) / len(vm_linear_terms)
    hessian_two = np.sum(vm_square_terms + vm_square_terms.transpose(0, 2, 1), axis=0)

    hessian = (hessian_one + hessian_two) / 2

    # correct averaging if terms in hessian_two are missing
    diag_mask = np.eye(len(hessian), dtype=bool)
    if not model_info.has_squares:
        hessian[diag_mask] *= 2
    if not model_info.has_interactions:
        hessian[~diag_mask] *= 2

    square_terms = hessian / 2
    linear_terms = gradient - hessian @ residuals

    return intercept, linear_terms, square_terms


def aggregator_least_squares_linear(vector_model, residuals, model_info):
    """Aggregate linear VectorModel assuming a least_squares functype.

    Here we assume that vector_model is a linear model. We further assume that the
    underlying functype is least_squares. This allows us to build a quadratic main model
    by simply pluggin the linear model into the main equation.

    """
    vm_linear_terms = vector_model.linear_terms

    if model_info.has_intercept:
        intercept = vector_model.intercept @ vector_model.intercept
        linear_terms = 2 * np.sum(intercept * vm_linear_terms, axis=0)
    else:
        intercept = None
        linear_terms = np.zeros(vm_linear_terms.shape[1])

    square_terms = vm_linear_terms.T @ vm_linear_terms

    return intercept, linear_terms, square_terms


def aggregator_least_squares_quadratic(vector_model, residuals, model_info):
    """Aggregate quadratic VectorModel using a Taylor approximation.

    We assume that vector_model is a quadratic model. We further assume that the
    underlying functype is least_squares. This allows us to build a quadratic main model
    using a second-degree Taylor approximation.

    """
    vm_linear_terms = vector_model.linear_terms
    vm_square_terms = vector_model.square_terms

    intercept = residuals @ residuals

    residual_hessians = vm_square_terms + vm_square_terms.tranpose(0, 2, 1)
    residual_gradients = vm_linear_terms + residual_hessians @ residuals

    linear_terms = residual_gradients @ residuals
    square_terms = (
        residual_gradients.T @ residual_gradients + residual_hessians @ residuals
    )

    return intercept, linear_terms, square_terms
