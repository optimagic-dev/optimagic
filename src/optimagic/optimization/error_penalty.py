from functools import partial

import numpy as np

from optimagic.config import CRITERION_PENALTY_CONSTANT, CRITERION_PENALTY_SLOPE
from optimagic.optimization.fun_value import (
    LeastSquaresFunctionValue,
    LikelihoodFunctionValue,
    ScalarFunctionValue,
)
from optimagic.typing import AggregationLevel, ErrorHandling


def _penalty_value(x, constant, slope, x0, dim_out=None):  # noqa: ARG001
    value = constant + slope * np.linalg.norm(x - x0)
    return ScalarFunctionValue(value=value)


def _penalty_loglikes(x, constant, slope, x0, dim_out):
    contrib = (constant + slope * np.linalg.norm(x - x0)) / dim_out
    contrib = np.ones(dim_out) * contrib
    return LikelihoodFunctionValue(value=contrib)


def _penalty_residuals(x, constant, slope, x0, dim_out):
    contrib = np.sqrt((constant + slope * np.linalg.norm(x - x0)) / dim_out)
    contrib = np.ones(dim_out) * contrib
    return LeastSquaresFunctionValue(value=contrib)


def _penalty_value_derivative(x, constant, slope, x0, dim_out=None):  # noqa: ARG001
    return slope * (x - x0) / np.linalg.norm(x - x0)


def _penalty_loglikes_derivative(x, constant, slope, x0, dim_out):  # noqa: ARG001
    row = slope * (x - x0) / (dim_out * np.linalg.norm(x - x0))
    return np.full((dim_out, len(x)), row)


def _penalty_residuals_derivative(x, constant, slope, x0, dim_out):
    inner_deriv = slope * (x - x0) / np.linalg.norm(x - x0)
    outer_deriv = 0.5 / np.sqrt(_penalty_value(x, constant, slope, x0).value * dim_out)
    row = outer_deriv * inner_deriv
    return np.full((dim_out, len(x)), row)


def get_error_penalty_function(
    error_handling,
    start_x,
    start_criterion,
    error_penalty,
    solver_type,
    direction,
):
    # TODO: We only ever need the task "criterion_and_derivative"
    if error_handling == ErrorHandling.RAISE:
        return None

    error_penalty = {} if error_penalty is None else error_penalty

    first_value = start_criterion.internal_value(solver_type)

    constant, slope = _process_error_penalty(
        error_penalty=error_penalty,
        first_value=first_value,
        direction=direction,
    )

    dim_out = (
        1
        if solver_type == AggregationLevel.SCALAR
        else len(start_criterion.internal_value(solver_type))
    )

    kwargs = {"constant": constant, "slope": slope, "x0": start_x, "dim_out": dim_out}

    if solver_type == AggregationLevel.SCALAR:
        _penalty = partial(_penalty_value, **kwargs)
        _derivative = partial(_penalty_value_derivative, **kwargs)
    elif solver_type == AggregationLevel.LIKELIHOOD:
        _penalty = partial(_penalty_loglikes, **kwargs)
        _derivative = partial(
            _penalty_loglikes_derivative,
            **kwargs,
        )
    elif solver_type == AggregationLevel.LEAST_SQUARES:
        _penalty = partial(
            _penalty_residuals,
            **kwargs,
        )
        _derivative = partial(_penalty_residuals_derivative, **kwargs)

    else:
        raise ValueError()

    def penalty(x, task="criterion_and_derivative"):
        if task == "criterion":
            out = _penalty(x)
        elif task == "derivative":
            out = _derivative(x)
        elif task == "criterion_and_derivative":
            out = (_penalty(x), _derivative(x))

        return out

    return penalty


def _process_error_penalty(error_penalty, first_value, direction):
    """Add default options to error_penalty options."""
    error_penalty = error_penalty.copy()

    if direction == "minimize":
        default_constant = (
            first_value + np.abs(first_value) + CRITERION_PENALTY_CONSTANT
        )
        default_slope = CRITERION_PENALTY_SLOPE
    else:
        default_constant = (
            first_value - np.abs(first_value) - CRITERION_PENALTY_CONSTANT
        )
        default_slope = -CRITERION_PENALTY_SLOPE

    constant = error_penalty.get("constant", default_constant)
    slope = error_penalty.get("slope", default_slope)

    return constant, slope
