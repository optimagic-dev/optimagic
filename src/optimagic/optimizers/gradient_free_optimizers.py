from dataclasses import dataclass
from functools import partial
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
from numpy.typing import NDArray

from optimagic import mark

# from optimagic.config import
from optimagic.optimization.algo_options import (
    CONVERGENCE_FTOL_ABS,
    CONVERGENCE_FTOL_REL,
)
from optimagic.optimization.algorithm import Algorithm, InternalOptimizeResult
from optimagic.optimization.internal_optimization_problem import (
    InternalBounds,
    InternalOptimizationProblem,
)
from optimagic.typing import (
    AggregationLevel,
    NonNegativeFloat,
    PositiveInt,
)

try:
    import gradient_free_optimizers as gfo  # todo
except ImportError:
    pass

if TYPE_CHECKING:
    from gradient_free_optimizers.optimizers.base_optimizer import BaseOptimizer


@mark.minimizer(
    name="gfo_hillclimbing",  # todo
    solver_type=AggregationLevel.SCALAR,
    is_available=True,  # todo
    is_global=False,
    needs_jac=False,
    needs_hess=False,
    needs_bounds=True,
    supports_parallelism=False,  # todo
    supports_bounds=True,
    supports_infinite_bounds=False,
    supports_linear_constraints=False,
    supports_nonlinear_constraints=True,
    disable_history=False,
)
@dataclass(frozen=True)
class HillClimbing(Algorithm):
    step_size = 0.03  # todo
    sampling: Literal["normal", "laplace", "logistic", "gumbel"] = "normal"
    n_neighbours: PositiveInt = 3  # todo name
    n_grid_points: PositiveInt = 200  # todo
    stopping_maxiter: PositiveInt = 100  # todo what to set
    stopping_maxtime: NonNegativeFloat | None = None  # todo check type
    stopping_funval: float | None = 0  # todo name
    convergence_iter_noimprove: PositiveInt = 1000  # todo name
    convergence_ftol_abs: NonNegativeFloat = CONVERGENCE_FTOL_ABS
    convergence_ftol_rel: NonNegativeFloat = CONVERGENCE_FTOL_REL  # todo
    memory: bool = True  # todo name
    verbosity: Literal["progress_bar", "print_results", "print_times"] | bool = (
        False  # todo
    )
    seed: int | None = None

    def _solve_internal_problem(
        self, problem: InternalOptimizationProblem, x0: NDArray[np.float64]
    ) -> InternalOptimizeResult:
        opt = gfo.HillClimbingOptimizer
        optimizer = partial(
            opt,
            epsilon=self.step_size,
            distribution=self.sampling,
            n_neighbours=self.n_neighbours,
        )

        res = _gfo_internal(
            problem=problem,
            x0=x0,
            optimizer=optimizer,
            n_grid_points=self.n_grid_points,
            stopping_maxiter=self.stopping_maxiter,
            stopping_maxtime=self.stopping_maxtime,
            stopping_funval=self.stopping_funval,
            convergence_iter_noimprove=self.convergence_iter_noimprove,
            convergence_ftol_abs=self.convergence_ftol_abs,
            convergence_ftol_rel=self.convergence_ftol_rel,
            memory=self.memory,
            verbosity=self.verbosity,
            seed=self.seed,
        )

        return res


def _gfo_internal(
    problem: InternalOptimizationProblem,
    x0: NDArray[np.float64],
    optimizer: "BaseOptimizer",
    n_grid_points: PositiveInt,
    stopping_maxiter: PositiveInt,  # todo what to set,
    stopping_maxtime: NonNegativeFloat | None,  # todo check type,
    stopping_funval: float | None,  # todo name,
    convergence_iter_noimprove: PositiveInt | None,  # todo name,
    convergence_ftol_abs: NonNegativeFloat,
    convergence_ftol_rel: NonNegativeFloat,  # todo
    memory: bool,  # todo name
    verbosity: Literal["progress_bar", "print_results", "print_times"] | bool,  # todo
    seed: int | None,
) -> InternalOptimizeResult:
    """Internal helper function."""

    # set early stopping criterion
    early_stopping = {
        "n_iter_no_change": convergence_iter_noimprove,
        "tol_abs": convergence_ftol_abs,
        "tol_rel": convergence_ftol_rel,
    }

    # define search space, initial params, population, constraints
    opt = optimizer(
        search_space=_get_search_space_gfo(problem.bounds, n_grid_points),
        initialize=_get_initialize(x0),
        constraints=_get_gfo_constraints(),
        random_state=seed,
    )

    # define objective function, negate to convert minimize to maximize
    def objective_function(para: dict[str, float]) -> float | NDArray[np.float64]:
        x = np.array(opt.conv.para2value(para))
        return -problem.fun(x)

    # run optimization
    opt.search(
        objective_function=objective_function,
        n_iter=stopping_maxiter,
        max_time=stopping_maxtime,
        max_score=stopping_funval,
        early_stopping=early_stopping,
        memory=memory,  # todo
        memory_warm_start=None,  # todo
        verbosity=verbosity,
        # optimum="minimum"
    )

    return _process_result_gfo(opt)


def _get_search_space_gfo(
    bounds: InternalBounds, n_grid_points: PositiveInt
) -> dict[str, NDArray[np.float64]]:
    """Create search space."""
    search_space = {}
    for i, (lower, upper) in enumerate(zip(bounds.lower, bounds.upper, strict=False)):  # type:ignore
        step = (upper - lower) / n_grid_points
        search_space[f"x{i}"] = np.arange(lower, upper, step)
    return search_space  # type:ignore


def _process_result_gfo(opt: "BaseOptimizer") -> InternalOptimizeResult:
    """Process result."""

    res = InternalOptimizeResult(
        x=np.array(opt.best_value),
        fun=opt.best_score,
        success=True,
        n_fun_evals=len(opt.eval_times),
        n_jac_evals=0,
        n_hess_evals=0,
        n_iterations=opt.n_iter_search,
    )

    return res


def _get_gfo_constraints() -> list[Any]:
    return []


def _get_initialize(x0: NDArray[np.float64]) -> dict[str, Any]:
    """Set initial params x0 or population."""
    init = _value2para(x0)
    initialize = {"warm_start": [init]}
    return initialize


def _value2para(x: NDArray[np.float64]) -> dict[str, float]:
    para = {}
    for i in range(len(x)):
        para[f"x{i}"] = x[i]
    return para
