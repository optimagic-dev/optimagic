"""Implement the MIGRAD algorithm from iminuit."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

import numpy as np
from numpy.typing import NDArray

from optimagic import mark
from optimagic.config import IS_IMINUIT_INSTALLED
from optimagic.exceptions import NotInstalledError
from optimagic.optimization.algo_options import (
    N_RESTARTS,
    STOPPING_MAXFUN,
)
from optimagic.optimization.algorithm import Algorithm, InternalOptimizeResult
from optimagic.optimization.internal_optimization_problem import (
    InternalOptimizationProblem,
)
from optimagic.typing import AggregationLevel

if TYPE_CHECKING:
    from iminuit import Minuit


@mark.minimizer(
    name="iminuit_migrad",
    solver_type=AggregationLevel.SCALAR,
    is_available=IS_IMINUIT_INSTALLED,
    is_global=False,
    needs_jac=True,
    needs_hess=False,
    needs_bounds=False,
    supports_parallelism=False,
    supports_bounds=True,
    supports_infinite_bounds=True,
    supports_linear_constraints=False,
    supports_nonlinear_constraints=False,
    disable_history=False,
)
@dataclass(frozen=True)
class IminuitMigrad(Algorithm):
    """Minimize a scalar differentiable function using the MIGRAD algorithm from
    iminuit.

    This optimizer wraps the MIGRAD algorithm from the iminuit package, which provides a
    Python interface to the Minuit2 C++ library developed and maintained by CERN.

    MIGRAD is a local optimization method in the quasi-Newton family. It iteratively
    builds an approximation of the inverse Hessian matrix using the DFP variable-metric
    method to efficiently navigate optimization landscapes.

    At each iteration, the algorithm attempts a Newton step, using gradient and Hessian
    approximations to move toward the function’s minimum. If this step fails to reduce
    the objective function, MIGRAD conducts a line search along the gradient direction
    to maintain progress. This continues until the convergence criteria, such as the
    Estimated Distance to Minimum (EDM) are met, that is, they fall below preset
    thresholds.

    MIGRAD is designed for statistical optimization problems where accurate parameter
    uncertainty estimates are essential. It excels at maximum-likelihood and least-
    squares fits common in scientific computing, and is best suited for smooth,
    differentiable cost functions.

    For best performance, supply analytical gradients. Convergence and solution will
    depend on your starting values. Bound constraints (limits) supported.

    """

    stopping_maxfun: int = STOPPING_MAXFUN
    """Maximum number of function evaluations."""

    n_restarts: int = N_RESTARTS
    """Number of times to restart the optimizer if convergence is not reached.

    A value of 1 (the default) indicates that the optimizer will only run once,
    disabling the restart feature. Values greater than 1 specify the maximum number of
    restart attempts.

    """

    def _solve_internal_problem(
        self, problem: InternalOptimizationProblem, params: NDArray[np.float64]
    ) -> InternalOptimizeResult:
        if not IS_IMINUIT_INSTALLED:
            raise NotInstalledError(  # pragma: no cover
                "To use the 'iminuit_migrad` optimizer you need to install iminuit. "
                "Use 'pip install iminuit' or 'conda install -c conda-forge iminuit'. "
                "Check the iminuit documentation for more details: "
                "https://scikit-hep.org/iminuit/install.html"
            )
        from iminuit import Minuit

        def wrapped_objective(x: NDArray[np.float64]) -> float:
            return float(problem.fun(x))

        m = Minuit(wrapped_objective, params, grad=problem.jac)

        bounds = _convert_bounds_to_minuit_limits(
            problem.bounds.lower, problem.bounds.upper
        )

        for i, (lower, upper) in enumerate(bounds):
            if lower is not None or upper is not None:
                m.limits[i] = (lower, upper)

        m.migrad(
            ncall=self.stopping_maxfun,
            iterate=self.n_restarts,
        )

        res = _process_minuit_result(m)
        return res


def _process_minuit_result(minuit_result: Minuit) -> InternalOptimizeResult:
    """Convert iminuit result to optimagic's internal result format."""

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
