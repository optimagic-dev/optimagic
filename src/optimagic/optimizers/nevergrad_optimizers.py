from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

try:
    import nevergrad as ng

    IS_NEVERGRAD_INSTALLED = True
except ImportError:
    IS_NEVERGRAD_INSTALLED = False

from optimagic import mark
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
    is_available=True,
    is_global=False,
    needs_jac=False,
    needs_hess=False,
    supports_parallelism=True,
    supports_bounds=True,
    supports_linear_constraints=True,
    supports_nonlinear_constraints=True,
    disable_history=False,
)
@dataclass(frozen=True)
class NevergradOnePlusOne(Algorithm):
    stopping_maxfun: PositiveInt = 100
    n_cores: int = 1

    def _solve_internal_problem(
        self, problem: InternalOptimizationProblem, x0: NDArray[np.float64]
    ) -> InternalOptimizeResult:
        print(problem.bounds)

        instrum = ng.p.Instrumentation(ng.p.Array(shape=x0.shape))

        optimizer = ng.optimizers.OnePlusOne(
            parametrization=instrum,
            budget=self.stopping_maxfun,
            num_workers=self.n_cores,
        )
        recommendation = optimizer.minimize(problem.fun)
        best_x = (recommendation.value)[0][0]
        best_fun = recommendation.loss
        n_fun_evals = optimizer.num_ask
        out = InternalOptimizeResult(
            x=best_x,
            fun=best_fun,
            n_fun_evals=n_fun_evals,
        )
        return out


# TESTING.........

# from optimagic.optimization.internal_optimization_problem import (
#     SphereExampleInternalOptimizationProblem,
# )

# problem = SphereExampleInternalOptimizationProblem()

# ngopo = NevergradOnePlusOne()

# result = ngopo._solve_internal_problem(problem, x0=np.array([1.0, 1.0]))

# print(result.fun)
# print(result.x)
# print(result.n_fun_evals)
