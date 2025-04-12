from dataclasses import dataclass
from typing import Literal

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
from optimagic.typing import AggregationLevel, NonNegativeFloat, PositiveInt


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
    noise_handling: Literal["random", "optimistic"] | None = None
    noise_handling_value: NonNegativeFloat = 0.005
    mutation: Literal[
        "gaussian",
        "cauchy",
        "discrete",
        "fastga",
        "rls",
        "doublefastga",
        "adaptive",
        "coordinatewise_adaptive",
        "portfolio",
        "discreteBSO",
        "lengler",
        "lengler2",
        "lengler3",
        "lenglerhalf",
        "lenglerfourth",
        "doerr",
        "lognormal",
        "xlognormal",
        "xsmalllognormal",
        "tinylognormal",
        "lognormal",
        "smalllognormal",
        "biglognormal",
        "hugelognormal",
    ] = "gaussian"
    annealing: Literal[
        "none", "Exp0.9", "Exp0.99", "Exp0.9Auto", "Lin100.0", "Lin1.0", "LinAuto"
    ] = "none"
    sparse: bool | int = False
    super_radii: bool = False
    smoother: bool = False
    roulette_size: int = 2
    antismooth: int = 55
    crossover: bool = False
    crossover_type: Literal["none", "rand", "max", "min", "onepoint", "twopoint"] = (
        "none"
    )
    tabu_length: int = 0
    rotation: bool = False

    stopping_maxiter: PositiveInt = STOPPING_MAXITER
    n_cores: PositiveInt = 1
    convergence_ftol_abs: NonNegativeFloat = CONVERGENCE_FTOL_ABS
    # time limit in seconds
    max_time: NonNegativeFloat = np.inf

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
        optimizer = ng.optimizers.ParametrizedOnePlusOne(
            noise_handling=(self.noise_handling, self.noise_handling_value),
            tabu_length=self.tabu_length,
            mutation=self.mutation,
            crossover=self.crossover,
            rotation=self.rotation,
            annealing=self.annealing,
            sparse=self.sparse,
            smoother=self.smoother,
            super_radii=self.super_radii,
            roulette_size=self.roulette_size,
            antismooth=self.antismooth,
            crossover_type=self.crossover_type,
        )(
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
