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
    constant: float,
    slope: float,
    x0: NDArray[np.float64],
    dim_out: int | None = None,
) -> tuple[ScalarFunctionValue, NDArray[np.float64]]:  # noqa: ARG001
    r"""Evaluate the scalar error penalty and its gradient.

    The penalty is :math:`P(x) = c + s \lVert x - x_0 \rVert` with gradient
    :math:`\nabla P(x) = s (x - x_0) / \lVert x - x_0 \rVert`.

    Args:
        x: Internal parameter vector of shape (n_params,).
        constant: The constant :math:`c`.
        slope: The slope :math:`s`.
        x0: Internal start parameters of shape (n_params,).
        dim_out: Unused; only present for a uniform signature across penalties.

    Returns:
        The penalty value and its gradient of shape (n_params,).

    """
    value = constant + slope * np.linalg.norm(x - x0)
    jac = slope * (x - x0) / np.linalg.norm(x - x0)
    return ScalarFunctionValue(value=value), jac


def _likelihood_penalty(
    x: NDArray[np.float64],
    constant: float,
    slope: float,
    x0: NDArray[np.float64],
    dim_out: int,
) -> tuple[LikelihoodFunctionValue, NDArray[np.float64]]:  # noqa: ARG001
    r"""Evaluate the likelihood error penalty and its Jacobian.

    The scalar penalty :math:`P(x)` (see :func:`_scalar_penalty`) is split into
    :math:`m` identical contributions :math:`P(x) / m`, so they sum to
    :math:`P(x)`. Each row of the Jacobian is :math:`\nabla P(x) / m`.

    Args:
        x: Internal parameter vector of shape (n_params,).
        constant: The constant :math:`c` of the scalar penalty.
        slope: The slope :math:`s` of the scalar penalty.
        x0: Internal start parameters of shape (n_params,).
        dim_out: Number of contributions :math:`m`.

    Returns:
        The contributions of shape (dim_out,) and the Jacobian of shape
        (dim_out, n_params).

    """
    factor = (constant + slope * np.linalg.norm(x - x0)) / dim_out
    contrib = np.ones(dim_out) * factor
    row = slope * (x - x0) / (dim_out * np.linalg.norm(x - x0))
    jac = np.full((dim_out, len(x)), row)
    return LikelihoodFunctionValue(value=contrib), jac


def _penalty_residuals(
    x: NDArray[np.float64],
    constant: float,
    slope: float,
    x0: NDArray[np.float64],
    dim_out: int,
) -> tuple[LeastSquaresFunctionValue, NDArray[np.float64]]:
    r"""Evaluate the least-squares error penalty and its Jacobian.

    The scalar penalty :math:`P(x)` (see :func:`_scalar_penalty`) is split into
    :math:`m` identical residuals :math:`\sqrt{P(x) / m}`, so their squares sum to
    :math:`P(x)`. By the chain rule, each row of the Jacobian is
    :math:`\nabla P(x) / (2 \sqrt{m P(x)})`.

    Args:
        x: Internal parameter vector of shape (n_params,).
        constant: The constant :math:`c` of the scalar penalty.
        slope: The slope :math:`s` of the scalar penalty.
        x0: Internal start parameters of shape (n_params,).
        dim_out: Number of residuals :math:`m`.

    Returns:
        The residuals of shape (dim_out,) and the Jacobian of shape
        (dim_out, n_params).

    """
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
    """Create the function that replaces the objective after an evaluation error.

    The penalty is worse than the start value and its slope points back towards
    ``start_x``. Its aggregated scalar value is the same for every aggregation
    level, so it does not depend on the solver type.

    Args:
        start_x: Internal start parameters of shape (n_params,).
        start_criterion: Objective function value at ``start_x``.
        error_penalty: Optional dict with the keys "constant" and "slope". Missing
            entries are filled with defaults derived from the scalar start value.
        solver_type: Aggregation level the optimizer expects.
        direction: Whether the problem is minimized or maximized.

    Returns:
        Function that maps an internal parameter vector of shape (n_params,) to the
        penalty value and its derivative, both at the aggregation level of
        ``solver_type``.

    """
    error_penalty = {} if error_penalty is None else error_penalty

    first_value = float(start_criterion.internal_value(AggregationLevel.SCALAR))

    constant, slope = _process_error_penalty(
        error_penalty=error_penalty,
        first_value=first_value,
        direction=direction,
    )

    dim_out = (
        1
        if solver_type == AggregationLevel.SCALAR
        else len(start_criterion.internal_value(solver_type))  # ty:ignore[invalid-argument-type]
    )

    _penalty: Callable[
        [
            NDArray[np.float64],
            float,
            float,
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
    first_value: float,
    direction: Direction,
) -> tuple[float, float]:
    r"""Fill in the default constant and slope of the error penalty.

    With scalar start value :math:`f_0`, the defaults are
    :math:`c = f_0 + |f_0| + C` and :math:`s = S` when minimizing, and
    :math:`c = f_0 - |f_0| - C` and :math:`s = -S` when maximizing, where
    :math:`C` and :math:`S` are ``CRITERION_PENALTY_CONSTANT`` and
    ``CRITERION_PENALTY_SLOPE``. The constant is therefore always worse than
    :math:`f_0`.

    Args:
        error_penalty: Optional dict with the keys "constant" and "slope". Given
            entries take precedence over the defaults.
        first_value: Scalar objective value :math:`f_0` at the start parameters.
        direction: Whether the problem is minimized or maximized.

    Returns:
        The constant :math:`c` and the slope :math:`s`.

    Raises:
        ValueError: If ``direction`` is not a valid Direction.

    """
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
