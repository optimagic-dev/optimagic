from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

try:
    import nevergrad as ng
except ImportError:
    pass

from optimagic import mark
from optimagic.config import IS_NEVERGRAD_INSTALLED
from optimagic.exceptions import NotInstalledError
from optimagic.optimization.algo_options import (
    CONVERGENCE_FTOL_ABS,
    STOPPING_MAXITER,
)
from optimagic.optimization.algorithm import Algorithm, InternalOptimizeResult
from optimagic.optimization.internal_optimization_problem import (
    InternalOptimizationProblem,
)
from optimagic.typing import (
    AggregationLevel,
    PositiveInt,
)


@mark.minimizer(
    name="nevergrad_oneplusone",
    solver_type=AggregationLevel.SCALAR,
    is_available=IS_NEVERGRAD_INSTALLED,
    is_global=False,
    needs_jac=False,
    needs_hess=False,
    supports_parallelism=True,
    supports_bounds=True,
    supports_linear_constraints=False,
    supports_nonlinear_constraints=True,
    disable_history=False,
)
@dataclass(frozen=True)
class NevergradOnePlusOne(Algorithm):
    stopping_maxiter: PositiveInt = STOPPING_MAXITER
    n_cores: int = 1
    convergence_ftol_abs: float = CONVERGENCE_FTOL_ABS
    # time limit in seconds
    max_time: float = np.inf

    def _solve_internal_problem(
        self, problem: InternalOptimizationProblem, x0: NDArray[np.float64]
    ) -> InternalOptimizeResult:
        if not IS_NEVERGRAD_INSTALLED:
            raise NotInstalledError(
                "Nevergrad is not installed. Please install it to use this algorithm."
            )
        instrum = ng.p.Instrumentation(
            ng.p.Array(shape=x0.shape).set_bounds(
                problem.bounds.lower, problem.bounds.upper
            )
        )

        optimizer = ng.optimizers.OnePlusOne(
            parametrization=instrum,
            budget=self.stopping_maxiter,
            num_workers=self.n_cores,
        )
        early_stopping = ng.callbacks.EarlyStopping.no_improvement_stopper(
            tolerance_window=self.convergence_ftol_abs
        )
        optimizer.register_callback("ask", early_stopping)

        early_stopping = ng.callbacks.EarlyStopping.timer(max_duration=self.max_time)
        optimizer.register_callback("ask", early_stopping)

        recommendation = optimizer.minimize(problem.fun)
        best_x = (recommendation.value)[0][0]
        best_fun = recommendation.loss
        n_fun_evals = optimizer.num_ask
        out = InternalOptimizeResult(
            x=best_x,
            fun=best_fun,
            success=True,
            n_fun_evals=n_fun_evals,
        )
        return out
