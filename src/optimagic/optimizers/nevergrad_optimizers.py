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
    disable_history=False,
)
@dataclass(frozen=True)
class NevergradPSO(Algorithm):
    transform: Literal["arctan", "gaussian", "identity"] = "arctan"
    population_size: int | None = None
    n_cores: int = 1
    seed: int | None = None
    stopping_maxfun: PositiveInt = STOPPING_MAXFUN_GLOBAL * 2  # imprecise algorithm
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

        if self.seed is not None:
            instrum.random_state.seed(self.seed)

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

        while optimizer.num_ask < self.stopping_maxfun:
            x_list = [
                optimizer.ask()
                for _ in range(
                    min(self.n_cores, self.stopping_maxfun - optimizer.num_ask)
                )
            ]
            losses = problem.batch_fun(
                [x.value[0][0] for x in x_list], n_cores=self.n_cores
            )
            for x, loss in zip(x_list, losses, strict=True):
                optimizer.tell(x, loss)

        recommendation = optimizer.provide_recommendation()

        result = InternalOptimizeResult(
            x=recommendation.value[0][0],
            fun=recommendation.loss,
            success=True,
            n_fun_evals=optimizer.num_ask,
            n_jac_evals=0,
            n_hess_evals=0,
        )

        return result
