"""Implement ensmallen optimizers."""

from dataclasses import dataclass

import numpy as np
import pyensmallen as pye
from numpy.typing import NDArray

from optimagic import mark
from optimagic.optimization.algo_options import (
    MAX_LINE_SEARCH_STEPS,
    STOPPING_MAXITER,
)
from optimagic.optimization.algorithm import Algorithm, InternalOptimizeResult
from optimagic.optimization.internal_optimization_problem import (
    InternalOptimizationProblem,
)
from optimagic.typing import (
    AggregationLevel,
)


@mark.minimizer(
    name="ensmallen_lbfgs",
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
class EnsmallenLBFGS(Algorithm):
    stopping_maxiter = STOPPING_MAXITER
    max_step_for_line_search = MAX_LINE_SEARCH_STEPS
    # min_step_for_line_search = MIN_LINE_SEARCH_STEPS

    def _solve_internal_problem(
        self, problem: InternalOptimizationProblem, x0: NDArray[np.float64]
    ) -> InternalOptimizeResult:
        optimizer = pye.L_BFGS()

        print(type(problem.fun(x0)))

        def objective_function(
            x: NDArray[np.float64], grad: NDArray[np.float64]
        ) -> np.float64:
            grad[:] = problem.jac(x)
            return np.float64(problem.fun(x))

        raw_res = optimizer.optimize(objective_function, x0)

        res = InternalOptimizeResult(
            x=raw_res,
            fun=problem.fun(raw_res),
        )

        return res
