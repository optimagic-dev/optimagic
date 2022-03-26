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
        "sum_taylor": aggregator_sum_taylor,
        "information_equality_linear": aggregator_information_equality_linear,
        "information_equality_quadratic": aggregator_information_equality_quadratic,
        "least_squares_linear": aggregator_least_squares_linear,
        "least_squares_linear_taylor": aggregator_least_squares_linear_taylor,
        "least_squares_quadratic": aggregator_least_squares_quadratic,
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
        if _aggregator_name not in ("idenitity", "sum"):
            raise NotImplementedError

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


def aggregator_sum_taylor(vector_model, fvec_center, model_info):
    """Aggregate quadratic VectorModel using Taylor approximation.

    Here we assume that vector_model is a quadratic model, which allows us to simply
    sum the coefficients over the residuals. If vector_model is linear the resulting
    main model will be linear.

    """
    vm_linear_terms = vector_model.linear_terms
    vm_square_terms = vector_model.square_terms

    residual_hessians = vm_square_terms + vm_square_terms.transpose(0, 2, 1)
    residual_gradients = (
        vm_linear_terms + residual_hessians.transpose(2, 1, 0) @ fvec_center
    )

    intercept = fvec_center.sum()
    linear_terms = residual_gradients.sum(axis=0)
    square_terms = residual_hessians.sum(axis=0)

    return intercept, linear_terms, square_terms


def aggregator_information_equality_linear(vector_model, fvec_center, model_info):
    """Aggregate linear VectorModel using the Fisher information equality.

    Here we assume that vector_model is a linear model. This implies that the estimated
    linear_terms correspond to a gradient estimate of the full model. By the information
    equality we can estimate the Hessian of the full model using the gradient. To get
    the coefficients for the quadratic main model we transform the gradient and Hessian.

    """
    intercept = fvec_center.sum(axis=0)

    vm_linear_terms = vector_model.linear_terms

    gradient = np.sum(vm_linear_terms, axis=0)
    hessian = -(vm_linear_terms.T @ vm_linear_terms) / len(vm_linear_terms)

    square_terms = hessian / 2
    linear_terms = gradient - hessian @ fvec_center

    return intercept, linear_terms, square_terms


def aggregator_information_equality_quadratic(vector_model, fvec_center, model_info):
    """Aggregate quadratic VectorModel using the Fisher information equality.

    Here we assume that vector_model is a quadratic model, that is, both square and
    interaction terms were fitted. To utilize the Fisher information equality we
    transform the coefficients to gradient and Hessian. We then get a second Hessian
    estimate using the Fisher inequality. The final Hessian estimate is an average
    of the two. Lastly we retransform gradient and Hessian to coefficients for the
    quadratic main model.

    """
    vm_linear_terms = vector_model.linear_terms
    vm_square_terms = vector_model.square_terms

    residual_gradients = (
        vm_linear_terms
        + (vm_square_terms + vm_square_terms.tranpose(0, 2, 1)) @ fvec_center
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

    intercept = fvec_center.sum(axis=0)
    square_terms = hessian / 2
    linear_terms = gradient - hessian @ fvec_center

    return intercept, linear_terms, square_terms


def aggregator_least_squares_linear(vector_model, fvec_center, model_info):
    """Aggregate linear VectorModel assuming a least_squares functype.

    Here we assume that vector_model is a linear model. We further assume that the
    underlying functype is least_squares. This allows us to build a quadratic main model
    by simply pluggin the linear model into the main equation.

    """
    vm_linear_terms = vector_model.linear_terms

    if model_info.has_intercepts:
        vm_intercepts = vector_model.intercepts.flatten()
        intercept = vm_intercepts @ vm_intercepts
        linear_terms = 2 * np.sum(intercept * vm_linear_terms, axis=0)
    else:
        intercept = None
        linear_terms = np.zeros(vm_linear_terms.shape[1])

    square_terms = vm_linear_terms.T @ vm_linear_terms

    return intercept, linear_terms, square_terms


def aggregator_least_squares_linear_taylor(vector_model, fvec_center, model_info):
    """Aggregate linear VectorModel assuming a least_squares functype using Taylor.

    Here we assume that vector_model is a linear model. We further assume that the
    underlying functype is least_squares. The main model is build using a second-degree
    Taylor approximation, where the gradient and Hessian are derived from the gradients
    of the residual models.

    """
    vm_linear_terms = vector_model.linear_terms

    intercept = fvec_center @ fvec_center
    linear_terms = 2 * vm_linear_terms.T @ fvec_center
    square_terms = 2 * vm_linear_terms.T @ vm_linear_terms

    return intercept, linear_terms, square_terms


def aggregator_least_squares_quadratic(vector_model, fvec_center, model_info):
    """Aggregate quadratic VectorModel using a Taylor approximation.

    We assume that vector_model is a quadratic model. We further assume that the
    underlying functype is least_squares. This allows us to build a quadratic main model
    using a second-degree Taylor approximation.

    """
    vm_linear_terms = vector_model.linear_terms
    vm_square_terms = vector_model.square_terms

    intercept = fvec_center @ fvec_center

    residual_hessians = vm_square_terms + vm_square_terms.tranpose(0, 2, 1)
    residual_gradients = vm_linear_terms + residual_hessians @ fvec_center

    linear_terms = residual_gradients @ fvec_center
    square_terms = (
        residual_gradients.T @ residual_gradients + residual_hessians @ fvec_center
    )

    return intercept, linear_terms, square_terms


def _is_second_order_model(model_info):
    return model_info.has_squares or model_info.has_interactions
