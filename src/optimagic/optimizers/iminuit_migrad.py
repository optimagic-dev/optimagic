from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np
from iminuit import Minuit  # type: ignore
from numpy.typing import NDArray

from optimagic import mark
from optimagic.optimization.algo_options import (
    STOPPING_MAXFUN,
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

    def _solve_internal_problem(
        self, problem: InternalOptimizationProblem, params: NDArray[np.float64]
    ) -> InternalOptimizeResult:
        def wrapped_objective(x: NDArray[np.float64]) -> float:
            return float(problem.fun(x))

        wrapped_gradient = None
        jac_func = None
        if problem.jac is not None:

            def wrapped_gradient(x: NDArray[np.float64]) -> NDArray[np.float64]:
                return problem.jac(x)

            jac_func = problem.jac
        elif problem.fun_and_jac is not None:

            def wrapped_gradient(x: NDArray[np.float64]) -> NDArray[np.float64]:
                _, jac = problem.fun_and_jac(x)
                return jac

            jac_func = lambda x: problem.fun_and_jac(x)[1]

        m = Minuit(wrapped_objective, params, grad=wrapped_gradient)

        if problem.bounds:
            lower_bounds = problem.bounds.lower
            upper_bounds = problem.bounds.upper

            if lower_bounds is not None and upper_bounds is not None:
                for i, (lower, upper) in enumerate(
                    zip(lower_bounds, upper_bounds, strict=False)
                ):
                    if lower is not None or upper is not None:
                        m.limits[i] = (lower, upper)

        m.migrad(ncall=self.stopping_maxfun)

        res = process_minuit_result(m, jac_func)
        return res


def process_minuit_result(
    minuit_result: Minuit,
    jac_fun: Optional[Callable[[NDArray[np.float64]], NDArray[np.float64]]] = None,
) -> InternalOptimizeResult:
    jac = None
    if jac_fun is not None:
        jac = jac_fun(np.array(minuit_result.values))

    res = InternalOptimizeResult(
        x=np.array(minuit_result.values),
        fun=minuit_result.fval,
        success=minuit_result.valid,
        message=repr(minuit_result.fmin),
        n_fun_evals=minuit_result.nfcn,
        n_jac_evals=minuit_result.ngrad,
        n_hess_evals=None,
        n_iterations=None,
        status=None,
        jac=jac,
        hess=np.array(minuit_result.hesse()),
        hess_inv=np.array(minuit_result.covariance),
        max_constraint_violation=None,
        info={"minos": minuit_result.minos()},
        history=None,
    )
    return res
