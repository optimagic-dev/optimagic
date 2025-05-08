"""Implement nevergrad optimizers."""

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from optimagic import mark
from optimagic.config import IS_NEVERGRAD_INSTALLED
from optimagic.exceptions import NotInstalledError
from optimagic.optimization.algo_options import (
    STOPPING_MAXFUN_GLOBAL,
)
from optimagic.optimization.algorithm import Algorithm, InternalOptimizeResult
from optimagic.optimization.internal_optimization_problem import (
    InternalOptimizationProblem,
)
from optimagic.typing import (
    AggregationLevel,
    Callable,
    NonNegativeFloat,
    PositiveInt,
)

if IS_NEVERGRAD_INSTALLED:
    import nevergrad as ng


@mark.minimizer(
    name="nevergrad_cmaes",
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
class NevergradCMAES(Algorithm):
    scale: NonNegativeFloat = 1.0
    seed: int | None = None
    population_size: int | None = None
    elitist: bool = False
    diagonal: bool = False
    use_fast_implementation: bool = False
    high_speed: bool = False
    n_cores: PositiveInt = 1
    stopping_maxfun: PositiveInt = STOPPING_MAXFUN_GLOBAL
    learning_rate_rank_one_update: NonNegativeFloat = 1.0
    learning_rate_rank_mu_update: NonNegativeFloat = 1.0
    ftol: NonNegativeFloat = 1e-11
    xtol: NonNegativeFloat = 1e-11

    def _solve_internal_problem(
        self, problem: InternalOptimizationProblem, x0: NDArray[np.float64]
    ) -> InternalOptimizeResult:
        cma_options = {
            "tolx": self.xtol,
            "tolfun": self.ftol,
            "CMA_rankmu": self.learning_rate_rank_mu_update,
            "CMA_rankone": self.learning_rate_rank_one_update,
        }

        raw_optimizer = ng.optimizers.ParametrizedCMA(
            scale=self.scale,
            popsize=self.population_size,
            elitist=self.elitist,
            diagonal=self.diagonal,
            high_speed=self.high_speed,
            fcmaes=self.use_fast_implementation,
            inopts=cma_options,
        )

        res = _nevergrad_internal(
            problem=problem,
            x0=x0,
            raw_optimizer=raw_optimizer,
            stopping_maxfun=self.stopping_maxfun,
            n_cores=self.n_cores,
            seed=self.seed,
        )

        return res


""" helper function for nevergrad"""


def _nevergrad_internal(
    problem: InternalOptimizationProblem,
    x0: NDArray[np.float64],
    n_cores: int,
    raw_optimizer: Callable[..., Any],
    stopping_maxfun: int,
    seed: int | None,
) -> InternalOptimizeResult:
    if not IS_NEVERGRAD_INSTALLED:
        raise NotInstalledError(
            "The nevergrad_cmaes optimizer requires the 'nevergrad' package to be "
            "installed. You can install it with `pip install nevergrad`. "
            "Visit https://facebookresearch.github.io/nevergrad/getting_started.html"
            " for more detailed installation instructions."
        )

    param = ng.p.Array(
        init=np.clip(x0, problem.bounds.lower, problem.bounds.upper)
    ).set_bounds(problem.bounds.lower, upper=problem.bounds.upper)

    instrum = ng.p.Instrumentation(param)

    if seed is not None:
        instrum.random_state.seed(seed)

    optimizer = raw_optimizer(
        parametrization=instrum, budget=stopping_maxfun, num_workers=n_cores
    )

    while optimizer.num_ask < stopping_maxfun:
        x_list = [
            optimizer.ask()
            for _ in range(min(n_cores, stopping_maxfun - optimizer.num_ask))
        ]
        losses = problem.batch_fun([x.value[0][0] for x in x_list], n_cores=n_cores)
        for x, loss in zip(x_list, losses, strict=True):
            optimizer.tell(x, loss)

    recommendation = optimizer.provide_recommendation()
    best_x = recommendation.value[0][0]
    loss = recommendation.loss
    if loss is None:
        loss = problem.fun(best_x)

    result = InternalOptimizeResult(
        x=best_x,
        fun=loss,
        success=True,
        n_fun_evals=optimizer.num_ask,
        n_jac_evals=0,
        n_hess_evals=0,
    )

    return result
