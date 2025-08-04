from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
from numpy.typing import NDArray

from optimagic import mark
from optimagic.config import IS_GRADIENT_FREE_OPTIMIZERS_INSTALLED
from optimagic.optimization.algo_options import (
    CONVERGENCE_FTOL_ABS,
    CONVERGENCE_FTOL_REL,
    STOPPING_MAXFUN_GLOBAL,
    get_population_size,
)
from optimagic.optimization.algorithm import Algorithm, InternalOptimizeResult
from optimagic.optimization.internal_optimization_problem import (
    InternalBounds,
    InternalOptimizationProblem,
)
from optimagic.parameters.conversion import Converter
from optimagic.typing import (
    AggregationLevel,
    NonNegativeFloat,
    PositiveFloat,
    PositiveInt,
    PyTree,
)

if TYPE_CHECKING:
    from gradient_free_optimizers.optimizers.base_optimizer import BaseOptimizer


@dataclass(frozen=True)
class GFOCommonOptions:
    """Common options for all optimizers from GFO."""

    n_grid_points: PositiveInt | PyTree = 200
    """Number of grid points in each dimension."""

    n_init: PositiveInt = 10
    """N positions will be randmoly initialized in the search plane."""

    stopping_maxiter: PositiveInt = STOPPING_MAXFUN_GLOBAL  # todo maybe maxfun global
    """Maximum number of iterations."""

    stopping_maxtime: NonNegativeFloat | None = None  # todo check type
    """Maximum time in seconds before termination."""

    stopping_funval: float | None = None  # todo name defn switch signs
    """"Stop the optimization if the objective function is more than this value."""

    convergence_iter_noimprove: PositiveInt = (
        50  # default is 10 , need to increase for pso
    )
    """Number of iterations without improvement before termination."""

    convergence_ftol_abs: NonNegativeFloat = CONVERGENCE_FTOL_ABS
    """Converge if the absolute change in the objective function is less than this
    value."""

    convergence_ftol_rel: NonNegativeFloat = CONVERGENCE_FTOL_REL
    """Converge if the relative change in the objective function is less than this
    value."""

    caching: bool = True
    """Whether to cache evaluated param and function values in a dictionary for
    lookup."""

    warm_start: list[PyTree] | None = None  # todo
    """List of additional start points for the optimization run."""

    verbosity: Literal["progress_bar", "print_results", "print_times"] | bool = (
        False  # todo
    )
    """Determines what part of the optimization information will be printed."""

    seed: int | None = None
    """Random seed for reproducibility."""


@mark.minimizer(
    name="gfo_hillclimbing",  # todo
    solver_type=AggregationLevel.SCALAR,
    is_available=IS_GRADIENT_FREE_OPTIMIZERS_INSTALLED,
    is_global=False,
    needs_jac=False,
    needs_hess=False,
    needs_bounds=True,
    supports_parallelism=False,  # todo
    supports_bounds=True,
    supports_infinite_bounds=False,
    supports_linear_constraints=False,
    supports_nonlinear_constraints=False,
    disable_history=False,
)
@dataclass(frozen=True)
class GFOHillClimbing(Algorithm, GFOCommonOptions):
    """Minimize a scalar function using the HillClimbing algorithm.

    This algorithm is a Python implementation of the HillClimbing algorithm through the
    gradient_free_optimizers package.

    Hill climbing is a local search algorithm suited for exploring combinatorial search
    spaces.

    It starts at an initial point, which is often chosen randomly and continues to move
    to positions within its neighbourhood with a better solution. It has no method
    against getting stuck in local optima.

    """

    epsilon: PositiveFloat = 0.03
    """The step-size of the hill climbing algorithm.If step_size is too large the newly
    selected positions will be at the edge of the search space.

    If its value is very low it might not find new positions.

    """

    distribution: Literal["normal", "laplace", "logistic", "gumbel"] = "normal"
    """Sampling method the algorithm samples from."""

    n_neighbours: PositiveInt = 3
    """The number of positions the algorithm explores from its current postion before
    setting its current position to the best of those neighbour positions.

    If the value of n_neighbours is large the hill-climbing-based algorithm will take a
    lot of time to choose the next position to move to, but the choice will probably be
    a good one. It might be a prudent approach to increase n_neighbours of the search-
    space has a lot of dimensions, because there are more possible directions to move
    to.

    """

    def _solve_internal_problem(
        self, problem: InternalOptimizationProblem, x0: NDArray[np.float64]
    ) -> InternalOptimizeResult:
        import gradient_free_optimizers as gfo

        opt = gfo.HillClimbingOptimizer
        optimizer = partial(
            opt,
            epsilon=self.epsilon,
            distribution=self.distribution,
            n_neighbours=self.n_neighbours,
        )
        res = _gfo_internal(
            problem=problem,
            x0=x0,
            optimizer=optimizer,
            warm_start=self.warm_start,
            n_init=self.n_init,
            n_grid_points=self.n_grid_points,
            stopping_maxiter=self.stopping_maxiter,
            stopping_maxtime=self.stopping_maxtime,
            stopping_funval=self.stopping_funval,
            convergence_iter_noimprove=self.convergence_iter_noimprove,
            convergence_ftol_abs=self.convergence_ftol_abs,
            convergence_ftol_rel=self.convergence_ftol_rel,
            caching=self.caching,
            verbosity=self.verbosity,
            seed=self.seed,
        )

        return res


@mark.minimizer(
    name="gfo_pso",  # todo
    solver_type=AggregationLevel.SCALAR,
    is_available=IS_GRADIENT_FREE_OPTIMIZERS_INSTALLED,
    is_global=False,
    needs_jac=False,
    needs_hess=False,
    needs_bounds=True,
    supports_parallelism=False,  # todo
    supports_bounds=True,
    supports_infinite_bounds=False,
    supports_linear_constraints=False,
    supports_nonlinear_constraints=False,
    disable_history=False,
)
@dataclass(frozen=True)
class GFOParticleSwarmOptimization(Algorithm, GFOCommonOptions):
    """Minimize a scalar function using the Particle Swarm Optimization algorithm.

    This algorithm is a Python implementation of the HillClimbing algorithm through the
    gradient_free_optimizers package.

    Hill climbing is a local search algorithm suited for exploring combinatorial search
    spaces.

    It starts at an initial point, which is often chosen randomly and continues to move
    to positions within its neighbourhood with a better solution. It has no method
    against getting stuck in local optima.

    """

    population_size: PositiveInt = 10
    initial_population: list[PyTree] | None = None
    inertia: NonNegativeFloat = 0.5
    cognitive_weight: NonNegativeFloat = 0.5
    social_weight: NonNegativeFloat = 0.5
    rand_rest_p: NonNegativeFloat = 0

    def _solve_internal_problem(
        self, problem: InternalOptimizationProblem, x0: NDArray[np.float64]
    ) -> InternalOptimizeResult:
        import gradient_free_optimizers as gfo

        population_size = get_population_size(
            population_size=self.population_size, x=x0, lower_bound=20
        )

        opt = gfo.ParticleSwarmOptimizer
        optimizer = partial(
            opt,
            population=population_size,
            inertia=self.inertia,
            cognitive_weight=self.cognitive_weight,
            social_weight=self.social_weight,
            rand_rest_p=self.rand_rest_p,
        )

        res = _gfo_internal(
            problem=problem,
            x0=x0,
            optimizer=optimizer,
            warm_start=self.initial_population,
            n_init=self.n_init,
            n_grid_points=self.n_grid_points,
            stopping_maxiter=self.stopping_maxiter,
            stopping_maxtime=self.stopping_maxtime,
            stopping_funval=self.stopping_funval,
            convergence_iter_noimprove=self.convergence_iter_noimprove,
            convergence_ftol_abs=self.convergence_ftol_abs,
            convergence_ftol_rel=self.convergence_ftol_rel,
            caching=self.caching,
            verbosity=self.verbosity,
            seed=self.seed,
        )

        return res


def _gfo_internal(
    problem: InternalOptimizationProblem,
    x0: NDArray[np.float64],
    optimizer: BaseOptimizer,
    warm_start: list[PyTree] | None,
    n_init: PositiveInt,
    n_grid_points: PositiveInt | PyTree,
    stopping_maxiter: PositiveInt,
    stopping_maxtime: NonNegativeFloat | None,  # todo check type,
    stopping_funval: float | None,
    convergence_iter_noimprove: PositiveInt | None,
    convergence_ftol_abs: NonNegativeFloat,
    convergence_ftol_rel: NonNegativeFloat,
    caching: bool,
    verbosity: Literal["progress_bar", "print_results", "print_times"] | bool,  # todo
    seed: int | None,
) -> InternalOptimizeResult:
    """Internal helper function.

    Define the search space and inital params, define the objective function and run
    optimization.

    """

    # set early stopping criterion
    early_stopping = {
        "n_iter_no_change": convergence_iter_noimprove,
        "tol_abs": convergence_ftol_abs,
        "tol_rel": convergence_ftol_rel,
    }

    # define search space, initial params, population, constraints
    opt = optimizer(
        search_space=_get_search_space_gfo(
            problem.bounds, n_grid_points, problem.converter
        ),
        initialize=_get_initialize(x0, n_init, warm_start, problem.converter),
        constraints=_get_gfo_constraints(),
        random_state=seed,
    )

    # define objective function,
    def objective_function(para: dict[str, float]) -> float | NDArray[np.float64]:
        x = np.array(opt.conv.para2value(para))
        return problem.fun(x)

    # run optimization
    opt.search(
        objective_function=objective_function,
        n_iter=stopping_maxiter,
        max_time=stopping_maxtime,
        max_score=stopping_funval,
        early_stopping=early_stopping,
        memory=caching,
        memory_warm_start=None,  # todo
        verbosity=verbosity,
        optimum="minimum",
    )

    return _process_result_gfo(opt)


def _get_search_space_gfo(
    bounds: InternalBounds, n_grid_points: PositiveInt | PyTree, converter: Converter
) -> dict[str, NDArray[np.float64]]:
    """Create search space.

    Args:
        bounds: Internal Bounds
        n_grid_points: number of grid points in each dimension
    Returns:
    dict: search_space dictionary

    """
    search_space = {}
    if bounds.lower is not None and bounds.upper is not None:
        dim = len(bounds.lower)
        upper = bounds.upper
        lower = bounds.lower

    if isinstance(n_grid_points, int):
        n_grid_points = [n_grid_points] * dim
    else:
        n_grid_points = converter.params_to_internal(n_grid_points)

    for i in range(dim):
        step = (upper[i] - lower[i]) / n_grid_points[i]
        search_space[f"x{i}"] = np.arange(lower[i], upper[i], step)

    return search_space


def _process_result_gfo(opt: "BaseOptimizer") -> InternalOptimizeResult:
    """Process result.

    Args:
        opt: Optimizer instance after optimization run is complete

    Returns:
        InternalOptimizeResult: Internal optimization result.

    """

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
    """Process constraints."""
    return []


def _get_initialize(
    x0: NDArray[np.float64],
    n_init: PositiveInt,
    warm_start: PyTree | None,
    converter: Converter,
) -> dict[str, Any]:
    """Set initial params x0, additional start points for the
        optimization run or the initial_population.
    Args:
    x0: initial param

    Returns:
    dict: initialize dictionary with initial parameters set
    """
    init = _value2para(x0)
    # dim = len(x0)
    initialize = {"warm_start": [init], "vertices": n_init}
    if warm_start is not None:
        internal_values = [converter.params_to_internal(value) for value in warm_start]
        warm_start = [_value2para(value) for value in internal_values]
        initialize["warm_start"] += warm_start

    return initialize


def _value2para(x: NDArray[np.float64]) -> dict[str, float]:
    """
    Convert values to dict
    Args:
        x: Array of parameter values

    Returns:
        dict: Dictionary of parameter values with key-value pair as { x{i} : x[i]}
    """
    para = {}
    for i in range(len(x)):
        para[f"x{i}"] = x[i]
    return para
