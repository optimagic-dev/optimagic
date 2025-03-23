from dataclasses import dataclass
from typing import Optional

import numpy as np
from iminuit import Minuit  # type: ignore
from numpy.typing import NDArray

from optimagic import mark
from optimagic.optimization.algo_options import (
    STOPPING_MAXFUN,
    STOPPING_MAXITER,
)
from optimagic.optimization.algorithm import Algorithm, InternalOptimizeResult
from optimagic.optimization.internal_optimization_problem import (
    InternalOptimizationProblem,
)
from optimagic.typing import AggregationLevel


@mark.minimizer(
    name="iminuit_migrad",
    solver_type=AggregationLevel.SCALAR,
    is_available=True,
    is_global=False,
    needs_jac=True,
    needs_hess=False,
    supports_parallelism=False,
    supports_bounds=True,
    supports_linear_constraints=False,
    supports_nonlinear_constraints=False,
    disable_history=False,
)
@dataclass(frozen=True)
class IminuitMigrad(Algorithm):
    stopping_maxfun: int = STOPPING_MAXFUN
    stopping_maxiter: int = STOPPING_MAXITER

    def _solve_internal_problem(
        self, problem: InternalOptimizationProblem, params: NDArray[np.float64]
    ) -> InternalOptimizeResult:
        def wrapped_objective(x: NDArray[np.float64]) -> float:
            return float(problem.fun(x))

        m = Minuit(wrapped_objective, params, grad=problem.jac)

        bounds = _convert_bounds_to_minuit_limits(
            problem.bounds.lower, problem.bounds.upper
        )
        _set_minuit_limits(m, bounds)

        m.migrad(
            ncall=self.stopping_maxfun,
            iterate=self.stopping_maxiter,  # review
        )

        res = _process_minuit_result(m)
        return res


def _process_minuit_result(minuit_result: Minuit) -> InternalOptimizeResult:
    """Convert iminuit result to Optimagic's internal result format."""

    res = InternalOptimizeResult(
        x=np.array(minuit_result.values),
        fun=minuit_result.fval,
        success=minuit_result.valid,
        message=repr(minuit_result.fmin),
        n_fun_evals=minuit_result.nfcn,
        n_jac_evals=minuit_result.ngrad,
        n_hess_evals=None,
        n_iterations=minuit_result.nfcn,
        status=None,
        jac=None,
        hess=None,
        hess_inv=np.array(minuit_result.covariance),
        max_constraint_violation=None,
        info=None,
        history=None,
    )
    return res


def _convert_bounds_to_minuit_limits(
    lower_bounds: Optional[NDArray[np.float64]],
    upper_bounds: Optional[NDArray[np.float64]],
) -> list[tuple[Optional[float], Optional[float]]]:
    """Convert optimization bounds to Minuit-compatible limit format.

    Transforms numpy arrays of bounds into List of tuples as expected by iminuit.
    Handles special values like np.inf, -np.inf, and np.nan by converting
    them to None where appropriate, as required by Minuit's limits API.

    Parameters
    ----------
    lower_bounds : Optional[NDArray[np.float64]]
        Array of lower bounds for parameters.
    upper_bounds : Optional[NDArray[np.float64]]
        Array of upper bounds for parameters.

    Returns
    -------
    list[tuple[Optional[float], Optional[float]]]
        List of (lower, upper) limit tuples in Minuit format, where:
        - None indicates unbounded (equivalent to infinity)
        - Float values represent actual bounds

    Notes
    -----
    Minuit expects bounds as tuples of (lower, upper) where:
    - `None` indicates no bound (equivalent to -inf or +inf)
    - A finite float value indicates a specific bound
    - Bounds can be asymmetric (e.g., one side bounded, one side not)

    """
    if lower_bounds is None or upper_bounds is None:
        return []

    return [
        (
            None if np.isneginf(lower) or np.isnan(lower) else float(lower),
            None if np.isposinf(upper) or np.isnan(upper) else float(upper),
        )
        for lower, upper in zip(lower_bounds, upper_bounds, strict=True)
    ]


def _set_minuit_limits(
    m: Minuit, bounds: list[tuple[Optional[float], Optional[float]]]
) -> None:
    """Set parameter limits on a Minuit minimizer instance.

    Applies the converted bounds to an iminuit.Minuit object. Minuit expects
    parameter limits as tuples of (lower, upper) for each parameter, where
    None indicates an unbounded direction.

    Parameters
    ----------
    m : Minuit
        The iminuit minimizer instance to configure.
    bounds : list[tuple[Optional[float], Optional[float]]]
        List of parameter bounds as (lower, upper) tuples in Minuit format.
        For each tuple:
        - (None, None): Fully unbounded parameter
        - (value, None): Lower bound only
        - (None, value): Upper bound only
        - (min, max): Two-sided constraint

    """
    for i, (lower, upper) in enumerate(bounds):
        if lower is not None or upper is not None:
            m.limits[i] = (lower, upper)
