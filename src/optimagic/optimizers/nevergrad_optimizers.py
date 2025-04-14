from dataclasses import dataclass
from typing import Literal, Tuple

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
    STOPPING_MAXFUN_GLOBAL,
)
from optimagic.optimization.algorithm import Algorithm, InternalOptimizeResult
from optimagic.optimization.internal_optimization_problem import (
    InternalOptimizationProblem,
)
from optimagic.typing import AggregationLevel, PositiveInt


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
    supports_nonlinear_constraints=True,  # TBA
    disable_history=False,
)
@dataclass(frozen=True)
class NevergradOnePlusOne(Algorithm):
    noise_handling: str | Tuple[str, float] | None = None
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

    stopping_maxfun: PositiveInt = STOPPING_MAXFUN_GLOBAL
    n_cores: PositiveInt = 1

    def _solve_internal_problem(
        self, problem: InternalOptimizationProblem, x0: NDArray[np.float64]
    ) -> InternalOptimizeResult:
        if not IS_NEVERGRAD_INSTALLED:
            raise NotInstalledError(
                "Nevergrad is not installed. Please install it to use this algorithm."
            )
        param = ng.p.Array(
            init=np.clip(x0, problem.bounds.lower, problem.bounds.upper)
        ).set_bounds(problem.bounds.lower, upper=problem.bounds.upper)
        instrum = ng.p.Instrumentation(param)

        optimizer = ng.optimizers.ParametrizedOnePlusOne(
            noise_handling=self.noise_handling,
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
            budget=self.stopping_maxfun,
            num_workers=self.n_cores,
        )

        while optimizer.num_ask < optimizer.budget:
            x_list = [
                optimizer.ask()
                for _ in range(
                    min(optimizer.num_workers, optimizer.budget - optimizer.num_ask)
                )
            ]
            losses = problem.batch_fun(
                [x.value[0][0] for x in x_list],
                n_cores=self.n_cores,
            )
            for x, loss in zip(x_list, losses, strict=True):
                optimizer.tell(x, loss, problem.nonlinear_constraints)

        recommendation = optimizer.provide_recommendation()

        result = InternalOptimizeResult(
            x=recommendation.value[0][0],
            fun=recommendation.loss,
            success=True,
            n_fun_evals=optimizer.num_ask,
        )

        return result
