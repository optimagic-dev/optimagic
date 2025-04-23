"""Implement nevergrad optimizers."""

from concurrent.futures import Future
from dataclasses import dataclass
from typing import Any, List

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
    random_init: bool = False

    def _solve_internal_problem(
        self, problem: InternalOptimizationProblem, x0: NDArray[np.float64]
    ) -> InternalOptimizeResult:
        cma_options = {
            "tolx": self.xtol,
            "tolfun": self.ftol,
            "CMA_rankmu": self.learning_rate_rank_mu_update,
            "CMA_rankone": self.learning_rate_rank_one_update,
        }

        optimizer = ng.optimizers.ParametrizedCMA(
            scale=self.scale,
            popsize=self.population_size,
            elitist=self.elitist,
            diagonal=self.diagonal,
            high_speed=self.high_speed,
            fcmaes=self.use_fast_implementation,
            random_init=self.random_init,
            inopts=cma_options,
        )

        res = nevergrad_internal(
            problem=problem,
            x0=x0,
            optimizer=optimizer,
            stopping_maxfun=self.stopping_maxfun,
            n_cores=self.n_cores,
        )

        return res


""" helper function for nevergrad"""


def nevergrad_internal(
    problem: InternalOptimizationProblem,
    x0: NDArray[np.float64],
    n_cores: int,
    optimizer: Callable[..., Any],
    stopping_maxfun: int,
) -> InternalOptimizeResult:
    if not IS_NEVERGRAD_INSTALLED:
        raise NotInstalledError("Nevergrad is not installed.")

    param = ng.p.Array(
        init=np.clip(x0, problem.bounds.lower, problem.bounds.upper)
    ).set_bounds(problem.bounds.lower, upper=problem.bounds.upper)
    instrum = ng.p.Instrumentation(param)

    parametrized_optimizer = optimizer(
        parametrization=instrum, budget=stopping_maxfun, num_workers=n_cores
    )

    executor = NevergradExecutor(
        batch_fun=problem.batch_fun, n_cores=n_cores, budget=stopping_maxfun
    )
    recommendation = parametrized_optimizer.minimize(
        problem.fun,
        executor=executor,
    )
    best_x = recommendation.value[0][0]

    result = InternalOptimizeResult(
        x=best_x,
        fun=problem.fun(best_x),
        success=True,
        n_fun_evals=parametrized_optimizer.num_ask,
        n_jac_evals=0,
    )

    return result


""" Custom Executor class that uses batch_fun inside """


class NevergradExecutor:
    def __init__(
        self,
        batch_fun: Callable[..., list[float | NDArray[np.float64]]],
        n_cores: int,
        budget: int,
    ):
        self._batch_fun: Callable[..., list[float | NDArray[np.float64]]] = batch_fun
        self._n_cores: int = n_cores
        self._budget: int = budget
        self._batch_inputs: List[NDArray[np.float64]] = []
        self._futures: List[Future[Any]] = []
        self._submitted_count: int = 0

    def submit(
        self, func: Callable[..., Any], *args: Any, **kwargs: Any
    ) -> Future[Any]:
        future: Future[Any] = Future()
        x = args[0]
        self._batch_inputs.append(x)
        self._futures.append(future)
        self._submitted_count += 1

        if len(self._batch_inputs) == self._n_cores:
            self._run_batch()

        if self._submitted_count == self._budget:
            self._flush()

        return future

    def _run_batch(self) -> None:
        results = self._batch_fun(
            x_list=self._batch_inputs, n_cores=len(self._batch_inputs)
        )

        for future, result in zip(self._futures, results, strict=False):
            future.set_result(result)

        self._batch_inputs.clear()
        self._futures.clear()

    def _flush(self) -> None:
        if self._batch_inputs:
            self._run_batch()
