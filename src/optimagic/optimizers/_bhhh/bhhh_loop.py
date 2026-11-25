"""Main loop of the BHHH algorithm."""

import warnings
from typing import Literal, cast

import numpy as np
from numpy.typing import NDArray

from optimagic.exceptions import UserFunctionRuntimeError
from optimagic.optimization.algorithm import InternalOptimizeResult
from optimagic.optimization.internal_optimization_problem import (
    InternalOptimizationProblem,
)
from optimagic.optimizers._bhhh.linalg import (
    OPGFactorization,
    condition_number,
    factorize_opg,
    rank,
    weakly_identified_directions,
)
from optimagic.optimizers._bhhh.line_search import (
    MIN_SHRINK_FACTOR,
    LineSearch,
    LineSearchResult,
    line_search,
)
from optimagic.optimizers._bhhh.subproblem import (
    compute_step,
    max_feasible_step_length,
)

StopReason = Literal[
    "convergence_decrement_tol",
    "convergence_ftol_abs",
    "convergence_xtol_rel",
    "stopping_maxiter",
    "stopping_maxfun",
    "line_search_failure",
]

CONVERGENCE_REASONS = (
    "convergence_decrement_tol",
    "convergence_ftol_abs",
    "convergence_xtol_rel",
)


def bhhh_internal(
    problem: InternalOptimizationProblem,
    x0: NDArray[np.float64],
    line_search_options: LineSearch,
    decrement_tol: float,
    ftol_abs: float,
    xtol_rel: float,
    maxiter: int,
    maxfun: int,
) -> InternalOptimizeResult:
    """Minimize a sum of contributions with the BHHH algorithm.

    Args:
        problem: The internal optimization problem. Its fun returns the contributions,
            whose sum is minimized, and its jac their Jacobian.
        x0: Start parameters, shape (k,).
        line_search_options: Options of the line search.
        decrement_tol: Stop if the decrement is at most this.
        ftol_abs: Stop if an accepted step improves the objective by at most this.
            Disabled if zero.
        xtol_rel: Stop if the relative BHHH step is at most this. Disabled if zero.
        maxiter: Maximum number of accepted steps.
        maxfun: Maximum number of evaluations of the objective.

    Returns:
        The optimization result. Success means that a convergence criterion is met.

    """
    lower, upper = _get_bounds(problem, len(x0))

    raw_fun_value, jac = problem.fun_and_jac(x0)
    fun_value = cast(NDArray[np.float64], raw_fun_value)
    x = x0
    criterion = float(np.sum(fun_value))

    zero_columns = np.flatnonzero(np.all(jac == 0, axis=0))
    if zero_columns.size > 0:
        warnings.warn(_zero_columns_message(zero_columns))

    n_fun_evals = 1
    n_jac_evals = 1
    n_iterations = 0
    n_backtracks = 0
    n_expansions = 0
    n_failed_fun_evals = 0
    n_failed_jac_evals = 0
    last_error: str | None = None
    stop_reason: StopReason | None = None

    while True:
        factorization = factorize_opg(jac)
        step, decrement = compute_step(factorization, lower - x, upper - x)
        relative_step = np.max(np.abs(step) / np.maximum(1, np.abs(x)))

        if decrement <= decrement_tol:
            stop_reason = "convergence_decrement_tol"
        elif stop_reason is not None:
            pass
        elif xtol_rel > 0 and relative_step <= xtol_rel:
            stop_reason = "convergence_xtol_rel"
        elif n_iterations >= maxiter:
            stop_reason = "stopping_maxiter"
        elif n_fun_evals >= maxfun:
            stop_reason = "stopping_maxfun"
        if stop_reason is not None:
            break

        # The step is a descent direction whenever the decrement is positive; this only
        # fails through rounding errors if the decrement tolerance is tiny.
        slope = float(factorization.gradient @ step)
        if not slope < 0:
            stop_reason = "line_search_failure"
            break

        max_step_length = min(
            line_search_options.max_step_factor,
            max_feasible_step_length(x, step, lower, upper),
        )
        initial_step_length = 1.0
        backtracks_left = line_search_options.max_backtracks
        is_retry = False
        while True:
            search = line_search(
                fun=problem.fun,  # ty: ignore[invalid-argument-type]
                x=x,
                criterion=criterion,
                direction=step,
                slope=slope,
                lower=lower,
                upper=upper,
                options=line_search_options,
                initial_step_length=initial_step_length,
                max_step_length=max_step_length,
                expand=not is_retry,
                max_backtracks=backtracks_left,
                max_fun_evals=maxfun - n_fun_evals,
            )
            n_fun_evals += search.n_fun_evals
            n_backtracks += search.n_backtracks
            n_expansions += search.n_expansions
            n_failed_fun_evals += search.n_failed_fun_evals
            if search.last_error is not None:
                last_error = _error_text(search.last_error)
            if search.status != "success":
                break
            # Derivatives that cannot be evaluated or are not finite at an accepted
            # point, e.g. numerical derivatives close to a region where the objective
            # is undefined, reject the point.
            n_jac_evals += 1
            try:
                new_jac = problem.jac(search.x)
            except UserFunctionRuntimeError as e:
                n_failed_jac_evals += 1
                last_error = _error_text(e)
                backtracks_left -= search.n_backtracks + 1
                if backtracks_left < 0:
                    search = _as_failure(search)
                    break
                initial_step_length = MIN_SHRINK_FACTOR * search.step_length
                is_retry = True
                continue
            break

        if search.status == "max_fun_evals":
            stop_reason = "stopping_maxfun"
            break
        if search.status == "max_backtracks":
            stop_reason = "line_search_failure"
            break

        n_iterations += 1
        if ftol_abs > 0 and criterion - search.criterion <= ftol_abs:
            stop_reason = "convergence_ftol_abs"
        x = search.x
        fun_value = cast(NDArray[np.float64], search.fun_value)
        criterion = search.criterion
        jac = new_jac

    weak_directions = weakly_identified_directions(factorization)
    if weak_directions or jac.shape[0] < jac.shape[1]:
        warnings.warn(_identification_message(weak_directions, *jac.shape))

    info = {
        "stop_reason": stop_reason,
        "decrement": decrement,
        "rank": rank(factorization),
        "condition_number": condition_number(factorization),
        "active_lower_bounds": np.flatnonzero(x <= lower).tolist(),
        "active_upper_bounds": np.flatnonzero(x >= upper).tolist(),
        "n_backtracks": n_backtracks,
        "n_expansions": n_expansions,
        "n_failed_fun_evals": n_failed_fun_evals,
        "n_failed_jac_evals": n_failed_jac_evals,
    }

    return InternalOptimizeResult(
        x=x,
        fun=fun_value,
        success=stop_reason in CONVERGENCE_REASONS,
        message=_message(
            stop_reason=stop_reason,
            decrement=decrement,
            decrement_tol=decrement_tol,
            factorization=factorization,
            n_failed_evals=n_failed_fun_evals + n_failed_jac_evals,
            last_error=last_error,
        ),
        n_fun_evals=n_fun_evals,
        n_jac_evals=n_jac_evals,
        n_iterations=n_iterations,
        jac=factorization.gradient,
        info=info,
    )


def _get_bounds(
    problem: InternalOptimizationProblem, n_params: int
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    lower = problem.bounds.lower
    upper = problem.bounds.upper
    lower = np.full(n_params, -np.inf) if lower is None else lower
    upper = np.full(n_params, np.inf) if upper is None else upper
    return lower, upper


def _as_failure(search: LineSearchResult) -> LineSearchResult:
    return LineSearchResult(
        status="max_backtracks",
        step_length=search.step_length,
        x=search.x,
        fun_value=None,
        criterion=search.criterion,
        n_fun_evals=search.n_fun_evals,
        n_backtracks=search.n_backtracks,
        n_expansions=search.n_expansions,
        n_failed_fun_evals=search.n_failed_fun_evals,
        last_error=search.last_error,
    )


def _error_text(error: UserFunctionRuntimeError) -> str:
    """Describe an error together with the user error that caused it."""
    cause = error.__cause__
    if cause is None:
        return str(error)
    return f"{error} {type(cause).__name__}: {cause}"


def _message(
    stop_reason: StopReason,
    decrement: float,
    decrement_tol: float,
    factorization: OPGFactorization,
    n_failed_evals: int,
    last_error: str | None,
) -> str:
    reasons = {
        "convergence_decrement_tol": (
            "The decrement is below convergence_decrement_tol."
        ),
        "convergence_ftol_abs": (
            "The last step improved the objective by less than convergence_ftol_abs."
        ),
        "convergence_xtol_rel": "The relative step is below convergence_xtol_rel.",
        "stopping_maxiter": "The maximum number of iterations was reached.",
        "stopping_maxfun": ("The maximum number of function evaluations was reached."),
        "line_search_failure": (
            "The line search found no point with a sufficiently better objective. "
            "Possible reasons are inaccurate derivatives, an objective that is not "
            "smooth, or a stopping tolerance below the attainable accuracy."
        ),
    }
    msg = (
        f"{reasons[stop_reason]} The decrement g'(J'J)^-1 g is {decrement:.3g}, "
        f"convergence_decrement_tol is {decrement_tol:.3g}. For a likelihood, the "
        f"estimate is about {np.sqrt(decrement):.2g} robust standard errors away from "
        "a stationary point."
    )
    if stop_reason != "convergence_decrement_tol" and rank(factorization) < len(
        factorization.gradient
    ):
        msg += " The outer product of gradients is singular; see the warning."
    if n_failed_evals > 0:
        msg += (
            f" At {n_failed_evals} trial points, the objective or its derivative "
            "raised an error or was not finite. These points were rejected. The last "
            f"error was: {last_error}"
        )
    return msg


def _zero_columns_message(positions: NDArray[np.int64]) -> str:
    return (
        "The derivatives of all likelihood contributions with respect to the internal "
        f"parameters at positions {positions.tolist()} are zero at the start values. "
        "These parameters do not affect the objective there, and BHHH does not move "
        "them as long as this is the case. Check whether they are identified and "
        "whether the start values are at a point where the derivative vanishes for "
        "all observations, e.g. a standard deviation of a random effect of zero. "
        + _POSITIONS_NOTE
    )


def _identification_message(
    weak_directions: list[NDArray[np.int64]], n_obs: int, n_params: int
) -> str:
    msg = (
        "The outer product of gradients (OPG) is singular or nearly singular at the "
        "final parameters, so not all parameters are identified from the data. "
    )
    if n_obs < n_params:
        msg += (
            f"There are fewer likelihood contributions ({n_obs}) than parameters "
            f"({n_params}). "
        )
    if weak_directions:
        groups = "; ".join(str(d.tolist()) for d in weak_directions)
        msg += (
            "The following groups of internal parameters (by position) are involved "
            f"in nearly singular directions: {groups}. "
        )
    msg += (
        "This usually indicates collinear variables, perfect prediction or "
        "parameters that do not affect the likelihood. Standard errors of the "
        "involved parameters are not reliable. " + _POSITIONS_NOTE
    )
    return msg


_POSITIONS_NOTE = (
    "Positions refer to the internal parameter vector, which equals the flattened "
    "params if no constraints are used."
)
