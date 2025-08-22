from __future__ import annotations

import math
from dataclasses import dataclass
from functools import partial
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
from numpy.typing import NDArray

from optimagic import mark
from optimagic.config import IS_GRADIENT_FREE_OPTIMIZERS_INSTALLED
from optimagic.optimization.algo_options import (
    CONVERGENCE_FTOL_ABS,
    STOPPING_MAXFUN_GLOBAL,
    STOPPING_MAXITER,
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
    PositiveInt,
    PyTree,
)
from optimagic.typing import UnitIntervalFloat as ProbabilityFloat

if TYPE_CHECKING:
    import pandas as pd
    from gradient_free_optimizers.optimizers.base_optimizer import BaseOptimizer


@dataclass(frozen=True)
class GFOCommonOptions:
    """Common options for all optimizers from GFO."""

    n_grid_points: PositiveInt | PyTree = 201
    """Number of grid points per dimension.

    If an integer is provided, it will be used for all dimensions.

    """

    n_init: PositiveInt = 20
    """Number of initialization steps to run.

    Accordingly, N//2 positions will be initialized in a grid like pattern and remaining
    initialized at the vertices and randomly in the search space.

    """

    stopping_maxiter: PositiveInt = STOPPING_MAXITER
    """Maximum number of iterations."""

    stopping_maxtime: NonNegativeFloat | None = None
    """Maximum time in seconds before termination."""

    stopping_funval: float | None = None
    """"Stop the optimization if the objective function is less than this value."""

    convergence_iter_noimprove: PositiveInt = 1000000  # do not want to trigger this
    """Number of iterations without improvement before termination."""

    convergence_ftol_abs: NonNegativeFloat | None = (
        CONVERGENCE_FTOL_ABS  # set to zero, so disabled
    )
    """Converge if the absolute change in the objective function is less than this
    value."""

    convergence_ftol_rel: NonNegativeFloat | None = None
    """Converge if the relative change in the objective function is less than this
    value."""

    caching: bool = True
    """Whether to cache evaluated param and function values in a dictionary for
    lookup."""

    extra_start_points: list[PyTree] | None = None
    """List of additional start points for the optimization run."""

    warm_start: pd.DataFrame | None = None
    """Pandas dataframe that contains score and paramter information that will be
    automatically loaded into the memory.

    example:

    score       x1      x2      x...
    0.756       0.1     0.2     ...
    0.823       0.3     0.1     ...
    ... ...     ...     ...
    ... ...     ...     ...

    """

    verbosity: Literal["progress_bar", "print_results", "print_times"] | bool = False
    """Determines what part of the optimization information will be printed."""

    seed: int | None = None
    """Random seed for reproducibility."""


# ==================================================================================
# Population Based
# ==================================================================================


@mark.minimizer(
    name="gfo_pso",
    solver_type=AggregationLevel.SCALAR,
    is_available=IS_GRADIENT_FREE_OPTIMIZERS_INSTALLED,
    is_global=True,
    needs_jac=False,
    needs_hess=False,
    needs_bounds=True,
    supports_parallelism=False,
    supports_bounds=True,
    supports_infinite_bounds=False,
    supports_linear_constraints=False,
    supports_nonlinear_constraints=False,
    disable_history=False,
)
@dataclass(frozen=True)
class GFOParticleSwarmOptimization(Algorithm, GFOCommonOptions):
    """Minimize a scalar function using the Particle Swarm Optimization algorithm.

    This algorithm is a Python implementation of the Particle Swarm Optimization
    algorithm through the gradient_free_optimizers package.

    Particle Swarm Optimization is a global population based algorithm.
    The algorithm simulates a swarm of particles across the search space.
    Each particle adjusts its position based on its own experience (cognitive weight)
    and the experiences of its neighbors or the swarm (social weight), using
    velocity updates.
    The algorithm iteratively guides the swarm toward promising regions of the
    search space. The velocity of a particle is calculated by the following
    equation:

    .. math::
        v_{n+1} = \\omega \\cdot v_n + c_k \\cdot r_1 \\cdot (p_{best}-p_n)
        + c_s \\cdot r_2 \\cdot (g_{best} - p_n)

    """

    population_size: PositiveInt = 10
    """Size of the population."""

    initial_population: list[PyTree] | None = None
    """The user-provided inital population."""

    inertia: NonNegativeFloat = 0.5 / math.log(2.0)
    """The inertia of the movement of the individual particles in the population."""

    cognitive_weight: NonNegativeFloat = 0.5 + math.log(2.0)
    """A factor of the movement towards the personal best position of the individual
    particles in the population."""

    social_weight: NonNegativeFloat = 0.5 + math.log(2.0)
    """A factor of the movement towards the global best position of the individual
    particles in the population."""

    rand_rest_p: NonNegativeFloat = 0.01
    """Probability for the optimization algorithm to jump to a random position in an
    iteration step."""

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
            common_options=self,
            problem=problem,
            x0=x0,
            optimizer=optimizer,
        )

        return res


@mark.minimizer(
    name="gfo_parallel_tempering",
    solver_type=AggregationLevel.SCALAR,
    is_available=IS_GRADIENT_FREE_OPTIMIZERS_INSTALLED,
    is_global=True,
    needs_jac=False,
    needs_hess=False,
    needs_bounds=True,
    supports_parallelism=False,
    supports_bounds=True,
    supports_infinite_bounds=False,
    supports_linear_constraints=False,
    supports_nonlinear_constraints=False,
    disable_history=False,
)
@dataclass(frozen=True)
class GFOParallelTempering(Algorithm, GFOCommonOptions):
    """Minimize a scalar function using the Parallel Tempering algorithm.

    This algorithm is a Python implementation of the Parallel Tempering
    algorithm through the gradient_free_optimizers package.

    Parallel Tempering is a global optimization algorithm that is inspired by
    metallurgical annealing. It runs multiple optimization chains at different
    "temperatures" in parallel. Periodically, swaps between these chains are
    attempted. Swaps between chains at different temperatures allow the optimizer
    to overcome local optima.

    The acceptance probability of a new position :math:`p_{new}` over an old one
    :math:`p_{old}` is given by:

    .. math::
        AP = e^{-\\frac{f(p_{new}) - f(p_{old})}{T}}

    where :math:`T` is the current temperature.

    """

    population_size: PositiveInt = 10
    """Size of the population, i.e., number of parallel chains."""

    initial_population: list[PyTree] | None = None
    """The user-provided inital population."""

    n_iter_swap: PositiveInt = 10
    """The number of iterations the algorithm performs before switching temperatures of
    the individual optimizers in the population."""

    rand_rest_p: NonNegativeFloat = 0
    """Probability for the optimization algorithm to jump to a random position in an
    iteration step."""

    def _solve_internal_problem(
        self, problem: InternalOptimizationProblem, x0: NDArray[np.float64]
    ) -> InternalOptimizeResult:
        import gradient_free_optimizers as gfo

        population_size = get_population_size(
            population_size=self.population_size, x=x0, lower_bound=20
        )

        opt = gfo.ParallelTemperingOptimizer
        optimizer = partial(
            opt,
            population=population_size,
            n_iter_swap=self.n_iter_swap,
            rand_rest_p=self.rand_rest_p,
        )

        res = _gfo_internal(
            common_options=self,
            problem=problem,
            x0=x0,
            optimizer=optimizer,
        )

        return res


@mark.minimizer(
    name="gfo_spiral_optimization",
    solver_type=AggregationLevel.SCALAR,
    is_available=IS_GRADIENT_FREE_OPTIMIZERS_INSTALLED,
    is_global=True,
    needs_jac=False,
    needs_hess=False,
    needs_bounds=True,
    supports_parallelism=False,
    supports_bounds=True,
    supports_infinite_bounds=False,
    supports_linear_constraints=False,
    supports_nonlinear_constraints=False,
    disable_history=False,
)
@dataclass(frozen=True)
class GFOSpiralOptimization(Algorithm, GFOCommonOptions):
    """Minimize a scalar function using the Spiral Optimization algorithm.

    This algorithm is a Python implementation of the Spiral Optimization
    algorithm through the gradient_free_optimizers package.

    Spiral Optimization is a global optimization algorithm inspired by the dynamics
    of spiral phenomena. It uses a multi-point search strategy that moves towards
    the current best solution in a logarithmic spiral trajectory.

    The spiral model for a two-dimensional search space is defined as:

    .. math::
        x_{i}(k+1) = S_n(r, \\theta) x_{i}(k) - (S_n(r, \\theta) - I_n) x^{*}

    where :math:`x^{*}` is the current center of the spiral (best solution),
    :math:`S_n(r, \\theta)` is a spiral rotation-scaling matrix, :math:`r` is the
    convergence rate, and :math:`\\theta` is the angle of rotation.

    """

    population_size: PositiveInt = 10
    """Size of the population for population-based optimization algorithms.

    Each
    member of the population is a separate optimizer. All population based
    optimizers in this package calculate the new positions one member at a time.
    So if the optimizer performs 10 iterations and has a population size of 10,
    then each member of the population would move once to a new position.
    Typical range: 4 to 25.

    """

    initial_population: list[PyTree] | None = None
    """The user-provided inital population."""

    decay_rate: NonNegativeFloat = 0.99
    """The `r` is called in the spiral-optimization equation and is usually referred to
    as a step-size, but behaves more like a modification factor of the radius of the
    spiral movement of the particles in this implementation.

    Lower values accelerate the convergence of the particles to the best known position,
    while values above 1 eventually lead to a movement where the particles spiral away
    from each other. Typical range: 0.85 to 1.15.

    """

    def _solve_internal_problem(
        self, problem: InternalOptimizationProblem, x0: NDArray[np.float64]
    ) -> InternalOptimizeResult:
        import gradient_free_optimizers as gfo

        population_size = get_population_size(
            population_size=self.population_size, x=x0, lower_bound=20
        )

        opt = gfo.SpiralOptimization
        optimizer = partial(
            opt,
            population=population_size,
            decay_rate=self.decay_rate,
        )

        res = _gfo_internal(
            common_options=self,
            problem=problem,
            x0=x0,
            optimizer=optimizer,
        )

        return res


@mark.minimizer(
    name="gfo_genetic_algorithm",
    solver_type=AggregationLevel.SCALAR,
    is_available=IS_GRADIENT_FREE_OPTIMIZERS_INSTALLED,
    is_global=True,
    needs_jac=False,
    needs_hess=False,
    needs_bounds=True,
    supports_parallelism=False,
    supports_bounds=True,
    supports_infinite_bounds=False,
    supports_linear_constraints=False,
    supports_nonlinear_constraints=False,
    disable_history=False,
)
@dataclass(frozen=True)
class GFOGeneticAlgorithm(Algorithm, GFOCommonOptions):
    """Minimize a scalar function using the Genetic Algorithm.

    This algorithm is a Python implementation of the Genetic Algorithm through the
    gradient_free_optimizers package.

    The Genetic Algorithm is an evolutionary algorithm inspired by the process of
    natural selection. It evolves a population of candidate solutions over generations
    using mechanisms like selection, crossover, and mutation to find the best solution.

    """

    population_size: PositiveInt = 10
    """Size of the population."""

    initial_population: list[PyTree] | None = None
    """The user-provided inital population."""

    mutation_rate: ProbabilityFloat = 0.5
    """Probability of a mutation event occurring in an individual."""

    crossover_rate: ProbabilityFloat = 0.5
    """Probability of a crossover event occurring between two parents."""

    n_parents: PositiveInt = 2

    offspring: PositiveInt = 10

    def _solve_internal_problem(
        self, problem: InternalOptimizationProblem, x0: NDArray[np.float64]
    ) -> InternalOptimizeResult:
        import gradient_free_optimizers as gfo

        population_size = get_population_size(
            population_size=self.population_size, x=x0, lower_bound=20
        )

        opt = gfo.GeneticAlgorithmOptimizer
        optimizer = partial(
            opt,
            population=population_size,
            mutation_rate=self.mutation_rate,
            crossover_rate=self.crossover_rate,
            n_parents=self.n_parents,
            offspring=self.offspring,
        )

        res = _gfo_internal(
            common_options=self,
            problem=problem,
            x0=x0,
            optimizer=optimizer,
        )

        return res


@mark.minimizer(
    name="gfo_evolution_strategy",
    solver_type=AggregationLevel.SCALAR,
    is_available=IS_GRADIENT_FREE_OPTIMIZERS_INSTALLED,
    is_global=True,
    needs_jac=False,
    needs_hess=False,
    needs_bounds=True,
    supports_parallelism=False,
    supports_bounds=True,
    supports_infinite_bounds=False,
    supports_linear_constraints=False,
    supports_nonlinear_constraints=False,
    disable_history=False,
)
@dataclass(frozen=True)
class GFOEvolutionStrategy(Algorithm, GFOCommonOptions):
    """Minimize a scalar function using the Evolution Strategy algorithm.

    This algorithm is a Python implementation of the Evolution Strategy algorithm
    through the gradient_free_optimizers package.

    Evolution Strategy is another type of evolutionary algorithm. It primarily relies on
    mutation to explore the search space. A population of parents generates offspring,
    and the fittest individuals from both parents and offspring are selected to form the
    next generation.

    """

    population_size: PositiveInt = 10
    """Size of the population."""

    stopping_maxiter: PositiveInt = STOPPING_MAXFUN_GLOBAL
    """Maximum number of iterations."""

    initial_population: list[PyTree] | None = None
    """The user-provided inital population."""

    mutation_rate: ProbabilityFloat = 0.7
    """Probability of a mutation event occurring in an individual."""

    crossover_rate: ProbabilityFloat = 0.3
    """Probability of a crossover event occurring between two parents."""

    rand_rest_p: NonNegativeFloat = 0
    """Probability for the optimization algorithm to jump to a random position in an
    iteration step."""

    def _solve_internal_problem(
        self, problem: InternalOptimizationProblem, x0: NDArray[np.float64]
    ) -> InternalOptimizeResult:
        import gradient_free_optimizers as gfo

        population_size = get_population_size(
            population_size=self.population_size, x=x0, lower_bound=20
        )

        opt = gfo.EvolutionStrategyOptimizer
        optimizer = partial(
            opt,
            population=population_size,
            mutation_rate=self.mutation_rate,
            crossover_rate=self.crossover_rate,
            rand_rest_p=self.rand_rest_p,
        )

        res = _gfo_internal(
            common_options=self,
            problem=problem,
            x0=x0,
            optimizer=optimizer,
        )

        return res


@mark.minimizer(
    name="gfo_differential_evolution",
    solver_type=AggregationLevel.SCALAR,
    is_available=IS_GRADIENT_FREE_OPTIMIZERS_INSTALLED,
    is_global=True,
    needs_jac=False,
    needs_hess=False,
    needs_bounds=True,
    supports_parallelism=False,
    supports_bounds=True,
    supports_infinite_bounds=False,
    supports_linear_constraints=False,
    supports_nonlinear_constraints=False,
    disable_history=False,
)
@dataclass(frozen=True)
class GFODifferentialEvolution(Algorithm, GFOCommonOptions):
    """Minimize a scalar function using the Differential Evolution algorithm.

    This algorithm is a Python implementation of the Differential Evolution
    algorithm through the gradient_free_optimizers package.

    Differential Evolution is a population-based optimization algorithm that
    creates new candidate solutions by combining existing ones. It creates new
    positions in the search space by adding the weighted difference between
    two population members to a third member.

    A new trial vector is generated according to:

    .. math::
        x_{trial} = x_{r1} + F \\cdot (x_{r2} - x_{r3})

    where :math:`r1, r2, r3` are random individuals from the population, and
    :math:`F` is the differential weight.

    """

    population_size: PositiveInt = 10
    """Size of the population."""

    initial_population: list[PyTree] | None = None
    """The user-provided inital population."""

    mutation_rate: ProbabilityFloat = 0.9
    """Probability of a mutation event occurring in an individual."""

    crossover_rate: ProbabilityFloat = 0.9
    """Probability of a crossover event occurring between two parents."""

    def _solve_internal_problem(
        self, problem: InternalOptimizationProblem, x0: NDArray[np.float64]
    ) -> InternalOptimizeResult:
        import gradient_free_optimizers as gfo

        population_size = get_population_size(
            population_size=self.population_size, x=x0, lower_bound=4
        )

        opt = gfo.DifferentialEvolutionOptimizer
        optimizer = partial(
            opt,
            population=population_size,
            mutation_rate=self.mutation_rate,
            crossover_rate=self.crossover_rate,
        )

        res = _gfo_internal(
            common_options=self,
            problem=problem,
            x0=x0,
            optimizer=optimizer,
        )

        return res


# ==================================================================================
# Helper functions
# ==================================================================================


def _gfo_internal(
    common_options: GFOCommonOptions,
    problem: InternalOptimizationProblem,
    x0: NDArray[np.float64],
    optimizer: BaseOptimizer,
) -> InternalOptimizeResult:
    """Internal helper function.

    Define the search space and inital params, define the objective function and run
    optimization.

    """
    # Use common options from GFOCommonOptions
    common = common_options

    # set early stopping criterion
    early_stopping = {
        "n_iter_no_change": common.convergence_iter_noimprove,
        "tol_abs": common.convergence_ftol_abs,
        "tol_rel": common.convergence_ftol_rel,
    }

    # define search space, initial params, population, constraints
    opt = optimizer(
        search_space=_get_search_space_gfo(
            problem.bounds,
            common.n_grid_points,
            problem.converter,
        ),
        initialize=_get_initialize_gfo(
            x0, common.n_init, common.extra_start_points, problem.converter
        ),
        constraints=_get_gfo_constraints(),
        random_state=common.seed,
    )

    # define objective function, negate to perform minimize
    def objective_function(para: dict[str, float]) -> float | NDArray[np.float64]:
        x = np.array(opt.conv.para2value(para))
        return -problem.fun(x)

    # negate in case of minimize
    stopping_funval = (
        -1 * common.stopping_funval if common.stopping_funval is not None else None
    )

    # run optimization
    opt.search(
        objective_function=objective_function,
        n_iter=common.stopping_maxiter,
        max_time=common.stopping_maxtime,
        max_score=stopping_funval,
        early_stopping=early_stopping,
        memory=common.caching,
        memory_warm_start=common.warm_start,
        verbosity=common.verbosity,
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
        n_grid_points = list(map(int, converter.params_to_internal(n_grid_points)))

    for i in range(dim):
        search_space[f"x{i}"] = np.linspace(lower[i], upper[i], n_grid_points[i])

    return search_space


def _get_gfo_constraints() -> list[Any]:
    """Process constraints."""
    return []


def _get_initialize_gfo(
    x0: NDArray[np.float64],
    n_init: PositiveInt,
    extra_start_points: list[PyTree] | None,
    converter: Converter,
) -> dict[str, Any]:
    """Set initial params x0, additional start points for the optimization run or the
    initial_population. Here, warm_start is actually extra_start_points.

    Args:
    x0: initial param

    Returns:
    dict: initialize dictionary with initial parameters set

    """
    init = _value2para(x0)
    x_list = [init]
    if extra_start_points is not None:
        internal_values = [converter.params_to_internal(x) for x in extra_start_points]
        extra_start_points = [_value2para(x) for x in internal_values]
        x_list += extra_start_points

    initialize = {
        "warm_start": x_list,
        "vertices": n_init // 2,
        "grid": n_init // 2,
    }
    return initialize


def _process_result_gfo(opt: "BaseOptimizer") -> InternalOptimizeResult:
    """Process result.

    Args:
        opt: Optimizer instance after optimization run is complete

    Returns:
        InternalOptimizeResult: Internal optimization result.

    """
    res = InternalOptimizeResult(
        x=np.array(opt.best_value),
        fun=-opt.best_score,  # negate once again
        success=True,
        n_fun_evals=len(opt.eval_times),
        n_jac_evals=0,
        n_hess_evals=0,
        n_iterations=opt.n_iter_search,
    )

    return res


def _value2para(x: NDArray[np.float64]) -> dict[str, float]:
    """Convert values to dict.

    Args:
        x: Array of parameter values

    Returns:
        dict: Dictionary of parameter values with key-value pair as { x{i} : x[i]}

    """
    para = {}
    for i in range(len(x)):
        para[f"x{i}"] = x[i]
    return para
