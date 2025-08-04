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
    """Number of grid points per dimension.

    If an integer is provided, it will be used for all dimensions.

    """

    n_init: PositiveInt = 10
    """Number of initialization steps to run.

    Accordingly, N positions will be initialized at the vertices and remaining
    initialized randmoly in the search space.

    """

    stopping_maxiter: PositiveInt = STOPPING_MAXFUN_GLOBAL
    """Maximum number of iterations."""

    stopping_maxtime: NonNegativeFloat | None = None
    """Maximum time in seconds before termination."""

    stopping_funval: float | None = None
    """"Stop the optimization if the objective function is less than this value."""

    convergence_iter_noimprove: PositiveInt = 50
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

    warm_start: list[PyTree] | None = None
    """List of additional start points for the optimization run."""

    verbosity: Literal["progress_bar", "print_results", "print_times"] | bool = False
    """Determines what part of the optimization information will be printed."""

    seed: int | None = None
    """Random seed for reproducibility."""


@mark.minimizer(
    name="gfo_hillclimbing",
    solver_type=AggregationLevel.SCALAR,
    is_available=IS_GRADIENT_FREE_OPTIMIZERS_INSTALLED,
    is_global=False,
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
    """The step-size of the hill climbing algorithm. If step_size is too large the newly
    selected positions will be at the edge of the search space.

    If its value is very low it might not find new positions.

    """

    distribution: Literal["normal", "laplace", "logistic", "gumbel"] = "normal"
    """The mathematical distribution the algorithm draws samples from.

    All available distributions are taken from the numpy-package.

    """

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
    name="gfo_stochastichillclimbing",
    solver_type=AggregationLevel.SCALAR,
    is_available=IS_GRADIENT_FREE_OPTIMIZERS_INSTALLED,
    is_global=False,
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
class GFOStochasticHillClimbing(Algorithm, GFOCommonOptions):
    """Minimize a scalar function using the Stochastic Hill Climbing algorithm.

    This algorithm is a Python implementation of the StochasticHillClimbing algorithm
    through the gradient_free_optimizers package.

    Stochastic hill climbing extends the normal hill climbing by a simple method against
    getting stuck in local optima.

    """

    epsilon: PositiveFloat = 0.03
    """The step-size of the hill climbing algorithm.If step_size is too large the newly
    selected positions will be at the edge of the search space.

    If its value is very low it might not find new positions.

    """

    distribution: Literal["normal", "laplace", "logistic", "gumbel"] = "normal"
    """The mathematical distribution the algorithm draws samples from.

    All available distributions are taken from the numpy-package.

    """

    n_neighbours: PositiveInt = 3
    """The number of positions the algorithm explores from its current postion before
    setting its current position to the best of those neighbour positions.

    If the value of n_neighbours is large the hill-climbing-based algorithm will take a
    lot of time to choose the next position to move to, but the choice will probably be
    a good one. It might be a prudent approach to increase n_neighbours of the search-
    space has a lot of dimensions, because there are more possible directions to move
    to.

    """

    p_accept: NonNegativeFloat = 0.1
    """The probability factor used in the equation to calculate if a worse position is
    accepted as the new position.

    If the new score is not better than the previous one the algorithm accepts worse
    positions with probability p_accept.

    .. math::
        score_{normalized} = norm * \\frac{score_{current} - score_{new}}
        {score_{current} + score_{new}}
    .. math::
        p = \\exp^{-score_{normalized}}

    If p is less than p_accept the new position gets accepted anyways.

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
            p_accept=self.p_accept,
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
    name="gfo_repulsinghillclimbing",
    solver_type=AggregationLevel.SCALAR,
    is_available=IS_GRADIENT_FREE_OPTIMIZERS_INSTALLED,
    is_global=False,
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
class GFORepulsingHillClimbing(Algorithm, GFOCommonOptions):
    """Minimize a scalar function using the Repulsing Hill Climbing algorithm.

    This algorithm is a Python implementation of the Repulsing Hill Climbing algorithm
    through the gradient_free_optimizers package.

    The algorithm inherits from the Hill climbing which is a local search algorithm but
    always activates its methods to espace local optima.

    """

    epsilon: PositiveFloat = 0.03
    """The step-size of the hill climbing algorithm. If step_size is too large the newly
    selected positions will be at the edge of the search space.

    If its value is very low it might not find new positions.

    """

    distribution: Literal["normal", "laplace", "logistic", "gumbel"] = "normal"
    """The mathematical distribution the algorithm draws samples from.

    All available distributions are taken from the numpy-package.

    """

    n_neighbours: PositiveInt = 3
    """The number of positions the algorithm explores from its current position before
    setting its current position to the best of those neighbour positions."""

    repulsion_factor: PositiveFloat = 5
    """The algorithm increases the step size by multiplying it with the repulsion_factor
    for the next iteration. This way the algorithm escapes the region that does not
    offer better positions.

    .. math::
        \\epsilon = \\epsilon * {repulsion factor}

    """

    def _solve_internal_problem(
        self, problem: InternalOptimizationProblem, x0: NDArray[np.float64]
    ) -> InternalOptimizeResult:
        import gradient_free_optimizers as gfo

        opt = gfo.RepulsingHillClimbingOptimizer
        optimizer = partial(
            opt,
            epsilon=self.epsilon,
            distribution=self.distribution,
            n_neighbours=self.n_neighbours,
            repulsion_factor=self.repulsion_factor,
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
    name="gfo_randomrestarthillclimbing",
    solver_type=AggregationLevel.SCALAR,
    is_available=IS_GRADIENT_FREE_OPTIMIZERS_INSTALLED,
    is_global=False,
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
class GFORandomRestartHillClimbing(Algorithm, GFOCommonOptions):
    """Minimize a scalar function using the Random Restart Hill Climbing algorithm.

    This algorithm is a Python implementation of the Random Restart Hill Climbing
    algorithm through the gradient_free_optimizers package.

    The random restart hill climbing works by starting a hill climbing search and
    jumping to a random new position after n_iter_restart iterations. Those restarts
    should prevent the algorithm getting stuck in local optima.

    """

    epsilon: PositiveFloat = 0.03
    """The step-size of the hill climbing algorithm.If step_size is too large the newly
    selected positions will be at the edge of the search space.

    If its value is very low it might not find new positions.

    """

    distribution: Literal["normal", "laplace", "logistic", "gumbel"] = "normal"
    """The mathematical distribution the algorithm draws samples from.

    All available distributions are taken from the numpy-package.

    """

    n_neighbours: PositiveInt = 3
    """The number of positions the algorithm explores from its current postion before
    setting its current position to the best of those neighbour positions.

    If the value of n_neighbours is large the hill-climbing-based algorithm will take a
    lot of time to choose the next position to move to, but the choice will probably be
    a good one. It might be a prudent approach to increase n_neighbours of the search-
    space has a lot of dimensions, because there are more possible directions to move
    to.

    """

    n_iter_restart: PositiveInt = 10
    """The number of iterations the algorithm performs before jumping to a random
    position."""

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
            n_iter_restart=self.n_iter_restart,
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
    name="gfo_simulatedannealing",
    solver_type=AggregationLevel.SCALAR,
    is_available=IS_GRADIENT_FREE_OPTIMIZERS_INSTALLED,
    is_global=False,
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
class GFOSimulatedAnnealing(Algorithm, GFOCommonOptions):
    """Minimize a scalar function using the Simulated Annealing algorithm.

    This algorithm is a Python implementation of Simulated Annealing through the
    gradient_free_optimizers package.

    Simulated annealing chooses its next possible position similar to hill climbing, but
    it accepts worse results with a probability that decreases with time. It simulates a
    temperature that decreases with each iteration, similar to a material cooling down.

    """

    epsilon: PositiveFloat = 0.03
    """The step-size of the algorithm.

    If step_size is too large the newly selected positions will be at the edge of the
    search space. If its value is very low it might not find new positions.

    """

    distribution: Literal["normal", "laplace", "logistic", "gumbel"] = "normal"
    """The mathematical distribution the algorithm draws samples from.

    All available distributions are taken from the numpy-package.

    """

    n_neighbours: PositiveInt = 3
    """The number of positions the algorithm explores from its current position before
    setting its current position to the best of those neighbour positions."""

    start_temp: PositiveFloat = 1
    """The start_temp is a factor for the probability p of accepting a worse position.

    .. math::
        p = \\exp^{-\\frac{score_{normalized}}{temp}}

    """

    annealing_rate: PositiveFloat = 0.97
    """Rate at which the temperatur-value of the algorithm decreases. An annealing rate
    above 1 increases the temperature over time.

    .. math::
        start\\_temp \\leftarrow start\\_temp * annealing\\_rate

    """

    def _solve_internal_problem(
        self, problem: InternalOptimizationProblem, x0: NDArray[np.float64]
    ) -> InternalOptimizeResult:
        import gradient_free_optimizers as gfo

        opt = gfo.SimulatedAnnealingOptimizer
        optimizer = partial(
            opt,
            epsilon=self.epsilon,
            distribution=self.distribution,
            n_neighbours=self.n_neighbours,
            start_temp=self.start_temp,
            annealing_rate=self.annealing_rate,
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
    name="gfo_downhillsimplex",  # nelder_mead
    solver_type=AggregationLevel.SCALAR,
    is_available=IS_GRADIENT_FREE_OPTIMIZERS_INSTALLED,
    is_global=False,
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
class GFODownhillSimplex(Algorithm, GFOCommonOptions):
    """Minimize a scalar function using the Downhill Simplex algorithm.

    This algorithm is a Python implementation of the Downhill Simplex algorithm through
    the gradient_free_optimizers package.

    The Downhill simplex or Nelder mead algorithm works by grouping `number of
    dimensions + 1` positions into a simplex, which can explore the search-space by
    changing shape. The simplex changes shape by reflecting, expanding, contracting or
    shrinking via the alpha, gamma, beta or sigma parameters. It needs at least `number
    of dimensions + 1` initial positions to form a simplex in the search-space and the
    movement of the positions in the simplex are affected by each other.

    """

    alpha: PositiveFloat = 1
    """The reflection parameter of the simplex algorithm."""

    gamma: PositiveFloat = 2
    """The expansion parameter of the simplex algorithm."""

    beta: PositiveFloat = 0.5
    """The contraction parameter of the simplex algorithm."""

    sigma: PositiveFloat = 0.5
    """The shrinking parameter of the simplex algorithm."""

    def _solve_internal_problem(
        self, problem: InternalOptimizationProblem, x0: NDArray[np.float64]
    ) -> InternalOptimizeResult:
        import gradient_free_optimizers as gfo

        opt = gfo.DownhillSimplexOptimizer
        optimizer = partial(
            opt,
            alpha=self.alpha,
            gamma=self.gamma,
            beta=self.beta,
            sigma=self.sigma,
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

    inertia: NonNegativeFloat = 0.5
    """The inertia of the movement of the individual particles in the population."""

    cognitive_weight: NonNegativeFloat = 0.5
    """A factor of the movement towards the personal best position of the individual
    particles in the population."""

    social_weight: NonNegativeFloat = 0.5
    """A factor of the movement towards the global best position of the individual
    particles in the population."""

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
    stopping_maxtime: NonNegativeFloat | None,
    stopping_funval: float | None,
    convergence_iter_noimprove: PositiveInt | None,
    convergence_ftol_abs: NonNegativeFloat,
    convergence_ftol_rel: NonNegativeFloat,
    caching: bool,
    verbosity: Literal["progress_bar", "print_results", "print_times"] | bool,
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

    # define objective function, negate to perform minimize
    def objective_function(para: dict[str, float]) -> float | NDArray[np.float64]:
        x = np.array(opt.conv.para2value(para))
        return -problem.fun(x)

    # negate in case of minimize
    if stopping_funval is not None:
        stopping_funval = -1 * stopping_funval

    # run optimization
    opt.search(
        objective_function=objective_function,
        n_iter=stopping_maxiter,
        max_time=stopping_maxtime,
        max_score=stopping_funval,
        early_stopping=early_stopping,
        memory=caching,
        memory_warm_start=None,
        verbosity=verbosity,
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
    """Set initial params x0, additional start points for the optimization run or the
    initial_population.

    Args:
    x0: initial param

    Returns:
    dict: initialize dictionary with initial parameters set

    """
    init = _value2para(x0)
    initialize = {"warm_start": [init], "vertices": n_init}
    if warm_start is not None:
        internal_values = [converter.params_to_internal(value) for value in warm_start]
        warm_start = [_value2para(value) for value in internal_values]
        initialize["warm_start"] += warm_start

    return initialize


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
