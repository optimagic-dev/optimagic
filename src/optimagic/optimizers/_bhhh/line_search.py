"""Line search of the BHHH algorithm.

The line search backtracks with safeguarded quadratic and cubic interpolation until the
Armijo condition holds (:cite:`Dennis1996`, Algorithm A6.3.1). If the first trial step
is accepted, it can be doubled as long as the objective keeps improving, because the OPG
often overestimates the curvature far from the optimum or under misspecification
(:cite:`Train2009`, section 8.3). Only the objective is evaluated at trial points. Trial
points at which the objective is not finite or cannot be evaluated are rejected.

"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Annotated, Callable, Literal

import numpy as np
import pydantic
from annotated_types import Ge, Gt, Lt
from numpy.typing import NDArray

from optimagic.exceptions import InvalidAlgoOptionError, UserFunctionRuntimeError
from optimagic.optimization.algo_options import MAX_LINE_SEARCH_STEPS
from optimagic.typing import DEFAULT_PYDANTIC_CONFIG, PositiveInt, validated_dataclass

MIN_SHRINK_FACTOR = 0.1
"""Lower safeguard for interpolated step lengths and shrink factor at invalid points."""

MAX_SHRINK_FACTOR = 0.5
"""Upper safeguard for interpolated step lengths."""


def _line_search_error(e: pydantic.ValidationError) -> Exception:
    msg = f"The following options of LineSearch are invalid:\n\n{e}"
    return InvalidAlgoOptionError(msg)


@validated_dataclass(config=DEFAULT_PYDANTIC_CONFIG, make_error=_line_search_error)
@dataclass(frozen=True)
class LineSearch:
    """Backtracking line search with the Armijo condition and optional step doubling.

    Trial points are only accepted if they improve the objective sufficiently. Steps
    that are rejected are shortened by safeguarded quadratic and cubic interpolation
    (:cite:`Dennis1996`, Algorithm A6.3.1). Trial points at which the objective is not
    finite or raises an error are rejected and the step is shortened by a factor of 10.

    """

    armijo_constant: Annotated[float, Gt(0), Lt(0.5)] = 1e-4
    r"""Constant c of the Armijo condition.

    A step length t is accepted if :math:`F(x + t p) \leq F(x) + c \, t \, g'p`, where
    F is the objective, g its gradient and p the BHHH step. The default is the value
    recommended by :cite:`Dennis1996`.

    """

    max_backtracks: PositiveInt = MAX_LINE_SEARCH_STEPS
    """Maximum number of step reductions per iteration.

    If no sufficient improvement is found, the optimization stops and reports a failure.

    """

    expansion: Literal["doubling", "none"] = "doubling"
    """How to lengthen a BHHH step that is accepted without backtracking.

    "doubling" tries step lengths 2, 4, 8, ... as long as the objective keeps improving
    (:cite:`Train2009`, section 8.3). This makes BHHH much faster when the OPG
    overestimates the curvature of the objective, which is common far from the optimum
    and in misspecified models. It costs one additional evaluation of the objective
    per iteration in which doubling does not pay off. "none" gives textbook BHHH with an
    Armijo line search.

    """

    max_step_factor: Annotated[float, Ge(1)] = 1024.0
    """Largest step length that expansion can reach, in multiples of the BHHH step.

    The default allows up to 10 doublings. Far from the optimum, the best step can be
    several orders of magnitude longer than the BHHH step. With bounds, expansion also
    stops at the first bound the step would cross.

    """


@dataclass(frozen=True)
class LineSearchResult:
    status: Literal["success", "max_backtracks", "max_fun_evals"]
    """Whether a step was accepted, or which limit stopped the search."""

    step_length: float
    """Accepted step length t, or the last tried one if no step was accepted."""

    x: NDArray[np.float64]
    """Accepted parameters (or the start parameters if no step was accepted)."""

    fun_value: NDArray[np.float64] | None
    """Contributions at the accepted parameters; None if no step was accepted."""

    criterion: float
    """Objective at the accepted parameters (or at the start parameters)."""

    n_fun_evals: int
    """Number of evaluations of the objective."""

    n_backtracks: int
    """Number of step reductions."""

    n_expansions: int
    """Number of accepted step doublings."""

    n_failed_fun_evals: int
    """Number of evaluations of the objective that raised an error."""

    last_error: UserFunctionRuntimeError | None
    """The last error raised by an evaluation of the objective, if any."""


def line_search(
    fun: Callable[[NDArray[np.float64]], NDArray[np.float64]],
    x: NDArray[np.float64],
    criterion: float,
    direction: NDArray[np.float64],
    slope: float,
    lower: NDArray[np.float64],
    upper: NDArray[np.float64],
    options: LineSearch,
    initial_step_length: float,
    max_step_length: float,
    expand: bool,
    max_backtracks: int,
    max_fun_evals: int,
) -> LineSearchResult:
    """Search for a step length along a descent direction.

    Args:
        fun: Function that maps parameters of shape (k,) to contributions of shape
            (n,) whose sum is minimized. It may raise a UserFunctionRuntimeError.
        x: Current parameters, shape (k,).
        criterion: Objective at x, i.e. the sum of the contributions.
        direction: Descent direction, shape (k,).
        slope: Directional derivative of the objective along direction (negative).
        lower: Lower bounds on the parameters, shape (k,). Can contain -inf.
        upper: Upper bounds on the parameters, shape (k,). Can contain inf.
        options: Options of the line search.
        initial_step_length: First step length to try.
        max_step_length: Largest step length that expansion may reach. Must keep
            x + max_step_length * direction within the bounds.
        expand: Whether to expand a first step that is accepted.
        max_backtracks: Maximum number of step reductions.
        max_fun_evals: Maximum number of evaluations of fun.

    Returns:
        The result of the line search.

    """
    n_fun_evals = 0
    n_backtracks = 0
    n_failed_fun_evals = 0
    last_error: UserFunctionRuntimeError | None = None

    def evaluate(t: float) -> tuple[NDArray[np.float64] | None, float]:
        nonlocal n_fun_evals, n_failed_fun_evals, last_error
        n_fun_evals += 1
        try:
            values = fun(np.clip(x + t * direction, lower, upper))
        except UserFunctionRuntimeError as e:
            n_failed_fun_evals += 1
            last_error = e
            return None, np.nan
        total = float(np.sum(values))
        return values, (total if np.isfinite(total) else np.nan)

    def is_acceptable(t: float, value: float) -> bool:
        return bool(value <= criterion + options.armijo_constant * t * slope)

    def failure(
        status: Literal["max_backtracks", "max_fun_evals"], t: float
    ) -> LineSearchResult:
        return LineSearchResult(
            status=status,
            step_length=t,
            x=x,
            fun_value=None,
            criterion=criterion,
            n_fun_evals=n_fun_evals,
            n_backtracks=n_backtracks,
            n_expansions=0,
            n_failed_fun_evals=n_failed_fun_evals,
            last_error=last_error,
        )

    t = initial_step_length
    if max_fun_evals < 1:
        return failure("max_fun_evals", t)
    values, value = evaluate(t)
    previous: tuple[float, float] | None = None
    while not is_acceptable(t, value):
        if n_backtracks >= max_backtracks:
            return failure("max_backtracks", t)
        if n_fun_evals >= max_fun_evals:
            return failure("max_fun_evals", t)
        t_new = next_step_length(t, value, previous, criterion, slope)
        if np.isfinite(value):
            previous = (t, value)
        t = t_new
        values, value = evaluate(t)
        n_backtracks += 1

    n_expansions = 0
    if expand and n_backtracks == 0 and options.expansion == "doubling":
        while 2 * t <= max_step_length and n_fun_evals < max_fun_evals:
            candidate_values, candidate = evaluate(2 * t)
            if not candidate < value:
                break
            t, values, value = 2 * t, candidate_values, candidate
            n_expansions += 1

    return LineSearchResult(
        status="success",
        step_length=t,
        x=np.clip(x + t * direction, lower, upper),
        fun_value=values,
        criterion=value,
        n_fun_evals=n_fun_evals,
        n_backtracks=n_backtracks,
        n_expansions=n_expansions,
        n_failed_fun_evals=n_failed_fun_evals,
        last_error=last_error,
    )


def next_step_length(
    t: float,
    value: float,
    previous: tuple[float, float] | None,
    criterion: float,
    slope: float,
) -> float:
    """Compute the next trial step length after a rejected step.

    The first reduction minimizes the quadratic through the objective and its slope at
    zero and the objective at t. Later reductions minimize the cubic that also passes
    through the previous trial point (:cite:`Dennis1996`, Algorithm A6.3.1). The result
    is safeguarded to lie in [0.1t, 0.5t]. If the objective is not finite at t, the
    step length is reduced by a factor of 10.

    Args:
        t: Rejected step length.
        value: Objective at t; nan if it was not finite.
        previous: The previous rejected step length and its objective value, if any
            had a finite objective.
        criterion: Objective at step length zero.
        slope: Directional derivative of the objective at step length zero.

    Returns:
        The next step length.

    """
    if not np.isfinite(value):
        return MIN_SHRINK_FACTOR * t

    # numpy floats return inf or nan on overflow, where Python floats would raise
    t, value, criterion, slope = map(np.float64, (t, value, criterion, slope))
    with np.errstate(all="ignore"):
        excess = value - criterion - slope * t
        if previous is None:
            candidate = -slope * t**2 / (2 * excess)
        else:
            t_prev, value_prev = map(np.float64, previous)
            excess_prev = value_prev - criterion - slope * t_prev
            a = (excess / t**2 - excess_prev / t_prev**2) / (t - t_prev)
            b = (-t_prev * excess / t**2 + t * excess_prev / t_prev**2) / (t - t_prev)
            discriminant = b**2 - 3 * a * slope
            if a == 0:
                candidate = -slope / (2 * b) if b > 0 else np.nan
            elif discriminant >= 0:
                candidate = (-b + np.sqrt(discriminant)) / (3 * a)
            else:
                candidate = np.nan

    if not np.isfinite(candidate):
        candidate = MAX_SHRINK_FACTOR * t
    return float(np.clip(candidate, MIN_SHRINK_FACTOR * t, MAX_SHRINK_FACTOR * t))
