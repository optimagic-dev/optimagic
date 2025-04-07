"""Implement nevergrad optimizers."""

import math
from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import NDArray

from optimagic import mark
from optimagic.config import IS_NEVERGRAD_INSTALLED
from optimagic.exceptions import NotInstalledError
from optimagic.optimization.algo_options import STOPPING_MAXFUN_GLOBAL
from optimagic.optimization.algorithm import Algorithm, InternalOptimizeResult
from optimagic.optimization.internal_optimization_problem import (
    InternalOptimizationProblem,
)
from optimagic.typing import AggregationLevel, PositiveInt

if IS_NEVERGRAD_INSTALLED:
    import nevergrad as ng


@mark.minimizer(
    name="nevergrad_pso",
    solver_type=AggregationLevel.SCALAR,
    is_available=IS_NEVERGRAD_INSTALLED,
    is_global=True,
    needs_jac=False,
    needs_hess=False,
    supports_parallelism=True,
    supports_bounds=True,
    supports_linear_constraints=False,
    supports_nonlinear_constraints=False,
    disable_history=True,
)
@dataclass(frozen=True)
class NevergradPSO(Algorithm):
    transform: Literal["arctan", "gaussian", "identity"] = "arctan"
    population_size: int | None = None
    n_cores: int = 1
    stopping_maxfun: PositiveInt = STOPPING_MAXFUN_GLOBAL
    inertia: float = 0.5 / math.log(2.0)
    cognitive: float = 0.5 + math.log(2.0)
    social: float = 0.5 + math.log(2.0)
    quasi_opp_init: bool = False
    speed_quasi_opp_init: bool = False
    special_speed_quasi_opp_init: bool = False

    def _solve_internal_problem(
        self, problem: InternalOptimizationProblem, x0: NDArray[np.float64]
    ) -> InternalOptimizeResult:
        if not IS_NEVERGRAD_INSTALLED:
            raise NotInstalledError(
                "The nevergrad_pso optimizer requires the 'nevergrad' package to be "
                "installed. You can install it with `pip install nevergrad`. "
                "Visit https://facebookresearch.github.io/nevergrad/getting_started.html"
                " for more detailed installation instructions."
            )

        instrum = ng.p.Instrumentation(
            ng.p.Array(
                shape=x0.shape, lower=problem.bounds.lower, upper=problem.bounds.upper
            )
        )

        optimizer = ng.optimizers.ConfPSO(
            transform=self.transform,
            popsize=self.population_size,
            omega=self.inertia,
            phip=self.cognitive,
            phig=self.social,
            qo=self.quasi_opp_init,
            sqo=self.speed_quasi_opp_init,
            so=self.special_speed_quasi_opp_init,
        )(
            parametrization=instrum,
            budget=self.stopping_maxfun,
            num_workers=self.n_cores,
        )

        recommendation = optimizer.minimize(problem.fun)

        result = InternalOptimizeResult(
            x=recommendation.value[0][0],
            fun=recommendation.loss,
            success=True,
            n_fun_evals=optimizer.num_ask,
        )

        return result
