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
    needs_jac=False,
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
    errordef: Optional[float] = None

    def _solve_internal_problem(
        self, problem: InternalOptimizationProblem, params: NDArray[np.float64]
    ) -> InternalOptimizeResult:
        def wrapped_objective(x: NDArray[np.float64]) -> float:
            return float(problem.fun(x))

        m = Minuit(
            wrapped_objective,
            params,
        )

        if problem.bounds:
            lower_bounds = problem.bounds.lower
            upper_bounds = problem.bounds.upper

            if lower_bounds is not None and upper_bounds is not None:
                for i, (lower, upper) in enumerate(
                    zip(lower_bounds, upper_bounds, strict=False)
                ):
                    if lower is not None or upper is not None:
                        m.limits[i] = (lower, upper)

        m.migrad(
            ncall=self.stopping_maxfun,
            iterate=self.stopping_maxiter,
        )
        print(m.params)

        res = process_minuit_result(m)
        return res


def process_minuit_result(minuit_result: Minuit) -> InternalOptimizeResult:
    x = np.array(minuit_result.values)
    fun = minuit_result.fval
    success = minuit_result.valid
    message = repr(minuit_result.fmin)
    jac: Optional[NDArray[np.float64]] = None
    if hasattr(minuit_result, "gradient"):
        jac = np.array(minuit_result.gradient)

    return InternalOptimizeResult(
        x=x,
        fun=fun,
        success=success,
        message=message,
        jac=jac,
    )
