from typing import Callable

import numpy as np
from numpy.typing import NDArray

from optimagic.config import CRITERION_PENALTY_CONSTANT, CRITERION_PENALTY_SLOPE
from optimagic.optimization.fun_value import (
    LeastSquaresFunctionValue,
    LikelihoodFunctionValue,
    ScalarFunctionValue,
    SpecificFunctionValue,
)
from optimagic.typing import AggregationLevel, Direction


def _scalar_penalty(
    x: NDArray[np.float64],
    constant: float | NDArray[np.float64],
    slope: float | NDArray[np.float64],
    x0: NDArray[np.float64],
    dim_out: int | None = None,
) -> tuple[ScalarFunctionValue, NDArray[np.float64]]:  # noqa: ARG001
    value = constant + slope * np.linalg.norm(x - x0)
    jac = slope * (x - x0) / np.linalg.norm(x - x0)
    return ScalarFunctionValue(value=value), jac


def _likelihood_penalty(
    x: NDArray[np.float64],
    constant: float | NDArray[np.float64],
    slope: float | NDArray[np.float64],
    x0: NDArray[np.float64],
    dim_out: int,
) -> tuple[LikelihoodFunctionValue, NDArray[np.float64]]:  # noqa: ARG001
    factor = (constant + slope * np.linalg.norm(x - x0)) / dim_out
    contrib = np.ones(dim_out) * factor
    row = slope * (x - x0) / (dim_out * np.linalg.norm(x - x0))
    jac = np.full((dim_out, len(x)), row)
    return LikelihoodFunctionValue(value=contrib), jac


def _penalty_residuals(
    x: NDArray[np.float64],
    constant: float | NDArray[np.float64],
    slope: float | NDArray[np.float64],
    x0: NDArray[np.float64],
    dim_out: int,
) -> tuple[LeastSquaresFunctionValue, NDArray[np.float64]]:
    factor = np.sqrt((constant + slope * np.linalg.norm(x - x0)) / dim_out)
    contrib = np.ones(dim_out) * factor

    scalar_penalty, _ = _scalar_penalty(x, constant, slope, x0)
    inner_deriv = slope * (x - x0) / np.linalg.norm(x - x0)
    outer_deriv = 0.5 / np.sqrt(scalar_penalty.value * dim_out)
    row = outer_deriv * inner_deriv
    jac = np.full((dim_out, len(x)), row)

    return LeastSquaresFunctionValue(value=contrib), jac


def get_error_penalty_function(
    start_x: NDArray[np.float64],
    start_criterion: SpecificFunctionValue,
    error_penalty: dict[str, float] | None,
    solver_type: AggregationLevel,
    direction: Direction,
) -> Callable[[NDArray[np.float64]], tuple[SpecificFunctionValue, NDArray[np.float64]]]:
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
        else len(start_criterion.internal_value(solver_type))  # type: ignore
    )

    _penalty: Callable[
        [
            NDArray[np.float64],
            float | NDArray[np.float64],
            float | NDArray[np.float64],
            NDArray[np.float64],
            int,
        ],
        tuple[SpecificFunctionValue, NDArray[np.float64]],
    ]
    if solver_type == AggregationLevel.SCALAR:
        _penalty = _scalar_penalty
    elif solver_type == AggregationLevel.LIKELIHOOD:
        _penalty = _likelihood_penalty
    elif solver_type == AggregationLevel.LEAST_SQUARES:
        _penalty = _penalty_residuals

    def penalty(
        x: NDArray[np.float64],
    ) -> tuple[SpecificFunctionValue, NDArray[np.float64]]:
        out = _penalty(
            x=x,
            constant=constant,
            slope=slope,
            x0=start_x,
            dim_out=dim_out,
        )
        return out

    return penalty


def _process_error_penalty(
    error_penalty: dict[str, float] | None,
    first_value: float | NDArray[np.float64],
    direction: Direction,
) -> tuple[float | NDArray[np.float64], float | NDArray[np.float64]]:
    """Add default options to error_penalty options."""
    if error_penalty is not None:
        error_penalty = error_penalty.copy()
    else:
        error_penalty = {}

    if direction == Direction.MINIMIZE:
        default_constant = (
            first_value + np.abs(first_value) + CRITERION_PENALTY_CONSTANT
        )
        default_slope = CRITERION_PENALTY_SLOPE
    elif direction == Direction.MAXIMIZE:
        default_constant = (
            first_value - np.abs(first_value) - CRITERION_PENALTY_CONSTANT
        )
        default_slope = -CRITERION_PENALTY_SLOPE
    else:
        raise ValueError()

    constant = error_penalty.get("constant", default_constant)
    slope = error_penalty.get("slope", default_slope)

    return constant, slope
