"""Implement PySwarms particle swarm optimization algorithms.

This module provides optimagic-compatible wrappers for PySwarms particle swarm
optimization algorithms including global best, local best, and general PSO variants with
support for different topologies.

"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Literal, Union

import numpy as np
from numpy.typing import NDArray

from optimagic import mark
from optimagic.config import IS_PYSWARMS_INSTALLED
from optimagic.exceptions import NotInstalledError
from optimagic.optimization.algo_options import (
    CONVERGENCE_FTOL_REL,
    STOPPING_MAXITER,
)
from optimagic.optimization.algorithm import Algorithm, InternalOptimizeResult
from optimagic.optimization.internal_optimization_problem import (
    InternalBounds,
    InternalOptimizationProblem,
)
from optimagic.typing import (
    AggregationLevel,
    NonNegativeFloat,
    PositiveFloat,
    PositiveInt,
)

PYSWARMS_NOT_INSTALLED_ERROR = (
    "This optimizer requires the 'pyswarms' package to be installed. "
    "You can install it with `pip install pyswarms`. "
    "Visit https://pyswarms.readthedocs.io/en/latest/installation.html "
    "for more detailed installation instructions."
)


# ======================================================================================
# 1. Topology Dataclasses
# ======================================================================================


@dataclass(frozen=True)
class BaseTopology:
    """Base class for all topology configurations."""


@dataclass(frozen=True)
class StarTopology(BaseTopology):
    """Star topology configuration.

    All particles are connected to the global best.

    """


@dataclass(frozen=True)
class RingTopology(BaseTopology):
    """Ring topology configuration.

    Particles are connected in a ring structure.

    """

    k_neighbors: PositiveInt = 3
    """Number of neighbors for each particle."""

    p_norm: Literal[1, 2] = 2
    """Distance metric for neighbor selection: 1 (Manhattan), 2 (Euclidean)."""

    static: bool = False
    """Whether to use a static or dynamic ring topology.

    When True, the neighborhood structure is fixed throughout optimization. When False,
    neighbors are recomputed at each iteration based on current particle positions.

    """


@dataclass(frozen=True)
class VonNeumannTopology(BaseTopology):
    """Von Neumann topology configuration.

    Particles are arranged on a 2D grid.

    """

    p_norm: Literal[1, 2] = 2
    """Distance metric for neighbor selection: 1 (Manhattan), 2 (Euclidean)."""

    range: PositiveInt = 1
    r"""Range parameter :math:`r` for neighborhood size."""


@dataclass(frozen=True)
class PyramidTopology(BaseTopology):
    """Pyramid topology configuration."""

    static: bool = False
    """Whether to use a static or dynamic pyramid topology.

    When True, the neighborhood structure is fixed throughout optimization. When False,
    neighbors are recomputed at each iteration based on current particle positions.

    """


@dataclass(frozen=True)
class RandomTopology(BaseTopology):
    """Random topology configuration.

    Particles are connected to random neighbors.

    """

    k_neighbors: PositiveInt = 3
    """Number of neighbors for each particle."""

    static: bool = False
    """Whether to use a static or dynamic random topology.

    When True, the neighborhood structure is fixed throughout optimization. When False,
    neighbors are recomputed at each iteration based on current particle positions.

    """


TopologyConfig = Union[
    Literal["star", "ring", "vonneumann", "random", "pyramid"],
    BaseTopology,
]

# ======================================================================================
# 2. PSO Options Classes
# ======================================================================================


@dataclass(frozen=True)
class BasePSOOptions:
    """Common PSO parameters used by all PSO variants."""

    cognitive_parameter: PositiveFloat
    """Cognitive parameter (c1) - attraction to personal best."""

    social_parameter: PositiveFloat
    """Social parameter (c2) - attraction to neighborhood/global best."""

    inertia_weight: PositiveFloat
    """Inertia weight (w) - momentum control."""


@dataclass(frozen=True)
class LocalBestPSOOptions(BasePSOOptions):
    """Local Best PSO specific parameters."""

    k_neighbors: PositiveInt
    """Number of neighbors in local neighborhood."""

    p_norm: Literal[1, 2]
    """Distance metric for neighbor selection (1=Manhattan, 2=Euclidean)."""


@dataclass(frozen=True)
class GeneralPSOOptions(BasePSOOptions):
    """General PSO parameters with topology support."""

    k_neighbors: PositiveInt | None = None
    """Number of neighbors for topologies requiring neighborhoods."""

    p_norm: Literal[1, 2] | None = None
    """Distance metric for neighbor selection (1=Manhattan, 2=Euclidean)."""

    vonneumann_range: PositiveInt | None = None
    """Range parameter for Von Neumann topology."""


@mark.minimizer(
    name="pyswarms_global_best",
    solver_type=AggregationLevel.SCALAR,
    is_available=IS_PYSWARMS_INSTALLED,
    is_global=True,
    needs_jac=False,
    needs_hess=False,
    needs_bounds=False,
    supports_parallelism=True,
    supports_bounds=True,
    supports_infinite_bounds=False,
    supports_linear_constraints=False,
    supports_nonlinear_constraints=False,
    disable_history=False,
)
@dataclass(frozen=True)
class PySwarmsGlobalBestPSO(Algorithm):
    r"""Minimize a scalar function using Global Best Particle Swarm Optimization.

    A population-based stochastic, global optimization optimization algorithm that
    simulates the social behavior of bird flocking or fish schooling. Particles
    (candidate solutions) move through the search space, adjusting their positions
    based on their own experience (cognitive component) and the experience of their
    neighbors or the entire swarm (social component).

    This implementation uses a star topology where all particles are connected to
    each other, making each particle aware of the global best solution found by the
    entire swarm.

    The position update follows:

    .. math::

        x_{i}(t+1) = x_{i}(t) + v_{i}(t+1)

    The velocity update follows:

    .. math::

        v_{ij}(t+1) = w \cdot v_{ij}(t) + c_1 r_{1j}(t)[y_{ij}(t) - x_{ij}(t)]
                      + c_2 r_{2j}(t)[\hat{y}_j(t) - x_{ij}(t)]

    Where:
        - :math:`w`: inertia weight controlling momentum
        - :math:`c_1`: cognitive parameter for attraction to personal best
        - :math:`c_2`: social parameter for attraction to global best
        - :math:`r_{1j}, r_{2j}`: random numbers in [0,1]
        - :math:`y_{ij}(t)`: personal best position of particle i
        - :math:`\hat{y}_j(t)`: global best position

    """

    n_particles: PositiveInt = 50
    """Number of particles in the swarm."""

    cognitive_parameter: PositiveFloat = 0.5
    r"""Cognitive parameter :math:`c_1` controlling attraction to personal best."""

    social_parameter: PositiveFloat = 0.3
    r"""Social parameter :math:`c_2` controlling attraction to global best."""

    inertia_weight: PositiveFloat = 0.9
    r"""Inertia weight :math:`w` controlling momentum."""

    convergence_ftol_rel: NonNegativeFloat = CONVERGENCE_FTOL_REL
    """Stop when relative change in objective function is less than this value."""

    convergence_ftol_iter: PositiveInt = 1
    """Number of iterations to check for convergence."""

    stopping_maxiter: PositiveInt = STOPPING_MAXITER
    """Maximum number of iterations."""

    boundary_strategy: Literal[
        "periodic", "reflective", "shrink", "random", "intermediate"
    ] = "periodic"
    """Strategy for handling out-of-bounds particles."""

    velocity_strategy: Literal["unmodified", "adjust", "invert", "zero"] = "unmodified"
    """Strategy for handling out-of-bounds velocities."""

    velocity_clamp_min: float | None = None
    """Minimum velocity limit for particles."""

    velocity_clamp_max: float | None = None
    """Maximum velocity limit for particles."""

    n_cores: PositiveInt = 1
    """Number of cores for parallel evaluation."""

    center_init: PositiveFloat = 1.0
    """Scaling factor for initial particle positions."""

    verbose: bool = False
    """Enable or disable the logs and progress bar."""

    def _solve_internal_problem(
        self, problem: InternalOptimizationProblem, x0: NDArray[np.float64]
    ) -> InternalOptimizeResult:
        if not IS_PYSWARMS_INSTALLED:
            raise NotInstalledError(PYSWARMS_NOT_INSTALLED_ERROR)

        import pyswarms as ps

        # Build structured options using dataclass
        pso_options = BasePSOOptions(
            cognitive_parameter=self.cognitive_parameter,
            social_parameter=self.social_parameter,
            inertia_weight=self.inertia_weight,
        )
        options = _build_pso_options_dict(pso_options)

        velocity_clamp = _build_velocity_clamp(
            self.velocity_clamp_min, self.velocity_clamp_max
        )

        bounds = _convert_bounds_to_pyswarms(problem.bounds, len(x0))

        init_pos = _create_initial_population(
            x0=x0, n_particles=self.n_particles, bounds=bounds, center=self.center_init
        )

        optimizer = ps.single.GlobalBestPSO(
            n_particles=self.n_particles,
            dimensions=len(x0),
            options=options,
            bounds=bounds,
            bh_strategy=self.boundary_strategy,
            velocity_clamp=velocity_clamp,
            vh_strategy=self.velocity_strategy,
            ftol=self.convergence_ftol_rel,
            ftol_iter=self.convergence_ftol_iter,
            init_pos=init_pos,
        )

        objective_wrapper = _create_batch_objective(problem, self.n_cores)

        result = optimizer.optimize(
            objective_func=objective_wrapper,
            iters=self.stopping_maxiter,
            verbose=self.verbose,
        )

        res = _process_pyswarms_result(
            result=result,
            n_particles=self.n_particles,
            n_iterations_run=self.stopping_maxiter,
        )

        return res


@mark.minimizer(
    name="pyswarms_local_best",
    solver_type=AggregationLevel.SCALAR,
    is_available=IS_PYSWARMS_INSTALLED,
    is_global=True,
    needs_jac=False,
    needs_hess=False,
    needs_bounds=False,
    supports_parallelism=True,
    supports_bounds=True,
    supports_infinite_bounds=False,
    supports_linear_constraints=False,
    supports_nonlinear_constraints=False,
    disable_history=False,
)
@dataclass(frozen=True)
class PySwarmsLocalBestPSO(Algorithm):
    r"""Minimize a scalar function using Local Best Particle Swarm Optimization.

    A variant of PSO that uses local neighborhoods instead of a single global best.
    Each particle is influenced only by the best position found within its local
    neighborhood, which is determined by the k-nearest neighbors using distance metrics.

    This approach uses a ring topology where particles are connected to their local
    neighbors, making each particle aware of only the best solution found within its
    neighborhood.

    The position update follows:

    .. math::

        x_{i}(t+1) = x_{i}(t) + v_{i}(t+1)

    The velocity update follows:

    .. math::

        v_{ij}(t+1) = w \cdot v_{ij}(t) + c_1 r_{1j}(t)[y_{ij}(t) - x_{ij}(t)]
                      + c_2 r_{2j}(t)[\hat{y}_{lj}(t) - x_{ij}(t)]

    Where:
        - :math:`w`: inertia weight controlling momentum
        - :math:`c_1`: cognitive parameter for attraction to personal best
        - :math:`c_2`: social parameter for attraction to local best
        - :math:`r_{1j}, r_{2j}`: random numbers in [0,1]
        - :math:`y_{ij}(t)`: personal best position of particle i
        - :math:`\hat{y}_{lj}(t)`: local best position in particle i's neighborhood

    """

    n_particles: PositiveInt = 50
    """Number of particles in the swarm."""

    cognitive_parameter: PositiveFloat = 0.5
    r"""Cognitive parameter :math:`c_1` controlling attraction to personal best."""

    social_parameter: PositiveFloat = 0.3
    r"""Social parameter :math:`c_2` controlling attraction to local best."""

    inertia_weight: PositiveFloat = 0.9
    r"""Inertia weight :math:`w` controlling momentum."""

    k_neighbors: PositiveInt = 3
    r"""Number of neighbors :math:`k` defining local neighborhood."""

    p_norm: Literal[1, 2] = 2
    """Distance metric: 1 (Manhattan), 2 (Euclidean). """

    convergence_ftol_rel: NonNegativeFloat = CONVERGENCE_FTOL_REL
    """Stop when relative change in objective function is less than this value."""

    convergence_ftol_iter: PositiveInt = 1
    """Number of iterations to check for convergence."""

    stopping_maxiter: PositiveInt = STOPPING_MAXITER
    """Maximum number of iterations."""

    boundary_strategy: Literal[
        "periodic", "reflective", "shrink", "random", "intermediate"
    ] = "periodic"
    """Strategy for handling out-of-bounds particles."""

    velocity_strategy: Literal["unmodified", "adjust", "invert", "zero"] = "unmodified"
    """Strategy for handling out-of-bounds velocities."""

    velocity_clamp_min: float | None = None
    """Minimum velocity limit for particles."""

    velocity_clamp_max: float | None = None
    """Maximum velocity limit for particles."""

    n_cores: PositiveInt = 1
    """Number of cores for parallel evaluation."""

    center_init: PositiveFloat = 1.0
    """Scaling factor for initial particle positions."""

    static_topology: bool = False
    """Whether to use static or dynamic ring topology."""

    verbose: bool = False
    """Enable or disable the logs and progress bar."""

    def _solve_internal_problem(
        self, problem: InternalOptimizationProblem, x0: NDArray[np.float64]
    ) -> InternalOptimizeResult:
        if not IS_PYSWARMS_INSTALLED:
            raise NotInstalledError(PYSWARMS_NOT_INSTALLED_ERROR)

        import pyswarms as ps

        # Build structured options using dataclass
        pso_options = LocalBestPSOOptions(
            cognitive_parameter=self.cognitive_parameter,
            social_parameter=self.social_parameter,
            inertia_weight=self.inertia_weight,
            k_neighbors=self.k_neighbors,
            p_norm=self.p_norm,
        )
        options = _build_pso_options_dict(pso_options)

        velocity_clamp = _build_velocity_clamp(
            self.velocity_clamp_min, self.velocity_clamp_max
        )

        bounds = _convert_bounds_to_pyswarms(problem.bounds, len(x0))

        init_pos = _create_initial_population(
            x0=x0, n_particles=self.n_particles, bounds=bounds, center=self.center_init
        )

        optimizer = ps.single.LocalBestPSO(
            n_particles=self.n_particles,
            dimensions=len(x0),
            options=options,
            bounds=bounds,
            bh_strategy=self.boundary_strategy,
            velocity_clamp=velocity_clamp,
            vh_strategy=self.velocity_strategy,
            ftol=self.convergence_ftol_rel,
            ftol_iter=self.convergence_ftol_iter,
            init_pos=init_pos,
            static=self.static_topology,
        )

        objective_wrapper = _create_batch_objective(problem, self.n_cores)

        result = optimizer.optimize(
            objective_func=objective_wrapper,
            iters=self.stopping_maxiter,
            verbose=self.verbose,
        )

        res = _process_pyswarms_result(
            result=result,
            n_particles=self.n_particles,
            n_iterations_run=self.stopping_maxiter,
        )

        return res


@mark.minimizer(
    name="pyswarms_general",
    solver_type=AggregationLevel.SCALAR,
    is_available=IS_PYSWARMS_INSTALLED,
    is_global=True,
    needs_jac=False,
    needs_hess=False,
    needs_bounds=False,
    supports_parallelism=True,
    supports_bounds=True,
    supports_infinite_bounds=False,
    supports_linear_constraints=False,
    supports_nonlinear_constraints=False,
    disable_history=False,
)
@dataclass(frozen=True)
class PySwarmsGeneralPSO(Algorithm):
    r"""Minimize a scalar function using General Particle Swarm Optimization with custom
    topologies.

    A flexible PSO implementation that allows selection of different neighborhood
    topologies, providing control over the balance between exploration and exploitation.
    The topology determines how particles communicate and share information, directly
    affecting the algorithm's search behavior.

    The position update follows:

    .. math::

        x_{i}(t+1) = x_{i}(t) + v_{i}(t+1)

    The velocity update follows:

    .. math::

        v_{ij}(t+1) = w \cdot v_{ij}(t) + c_1 r_{1j}(t)[y_{ij}(t) - x_{ij}(t)]
                      + c_2 r_{2j}(t)[\hat{y}_{nj}(t) - x_{ij}(t)]

    Where:
        - :math:`w`: inertia weight controlling momentum
        - :math:`c_1`: cognitive parameter for attraction to personal best
        - :math:`c_2`: social parameter for attraction to neighborhood best
        - :math:`r_{1j}, r_{2j}`: random numbers in [0,1]
        - :math:`y_{ij}(t)`: personal best position of particle i
        - :math:`\hat{y}_{nj}(t)`: neighborhood best position

    **Topology Options:**

    - **Star**: All particles connected to global best
    - **Ring**: Ring arrangement with k neighbors
    - **Von Neumann**: 2D grid topology
    - **Random**: Dynamic random connections
    - **Pyramid**: Hierarchical pyramid-like network of connected particles

    """

    n_particles: PositiveInt = 50
    """Number of particles in the swarm."""

    cognitive_parameter: PositiveFloat = 0.5
    r"""Cognitive parameter :math:`c_1` controlling attraction to personal best."""

    social_parameter: PositiveFloat = 0.3
    r"""Social parameter :math:`c_2` controlling attraction to neighborhood best."""

    inertia_weight: PositiveFloat = 0.9
    r"""Inertia weight :math:`w` controlling momentum."""

    topology: TopologyConfig = "star"
    """Topology structure for particle communication.

    Can be a string name or a topology dataclass instance.

    """

    convergence_ftol_rel: NonNegativeFloat = CONVERGENCE_FTOL_REL
    """Stop when relative change in objective function is less than this value."""

    convergence_ftol_iter: PositiveInt = 1
    """Number of iterations to check for convergence."""

    stopping_maxiter: PositiveInt = STOPPING_MAXITER
    """Maximum number of iterations."""

    boundary_strategy: Literal[
        "periodic", "reflective", "shrink", "random", "intermediate"
    ] = "periodic"
    """Strategy for handling out-of-bounds particles."""

    velocity_strategy: Literal["unmodified", "adjust", "invert", "zero"] = "unmodified"
    """Strategy for handling out-of-bounds velocities."""

    velocity_clamp_min: float | None = None
    """Minimum velocity limit for particles."""

    velocity_clamp_max: float | None = None
    """Maximum velocity limit for particles."""

    n_cores: PositiveInt = 1
    """Number of cores for parallel evaluation."""

    center_init: PositiveFloat = 1.0
    """Scaling factor for initial particle positions."""

    verbose: bool = False
    """Enable or disable the logs and progress bar."""

    def _solve_internal_problem(
        self, problem: InternalOptimizationProblem, x0: NDArray[np.float64]
    ) -> InternalOptimizeResult:
        if not IS_PYSWARMS_INSTALLED:
            raise NotInstalledError(PYSWARMS_NOT_INSTALLED_ERROR)

        import pyswarms as ps

        # Resolve topology config to PySwarms topology instance and options
        pyswarms_topology, topology_options = _resolve_topology_config(self.topology)

        base_options = {
            "c1": self.cognitive_parameter,
            "c2": self.social_parameter,
            "w": self.inertia_weight,
        }
        options = {**base_options, **topology_options}

        velocity_clamp = _build_velocity_clamp(
            self.velocity_clamp_min, self.velocity_clamp_max
        )
        bounds = _convert_bounds_to_pyswarms(problem.bounds, len(x0))
        init_pos = _create_initial_population(
            x0=x0, n_particles=self.n_particles, bounds=bounds, center=self.center_init
        )

        optimizer = ps.single.GeneralOptimizerPSO(
            n_particles=self.n_particles,
            dimensions=len(x0),
            options=options,
            topology=pyswarms_topology,
            bounds=bounds,
            bh_strategy=self.boundary_strategy,
            velocity_clamp=velocity_clamp,
            vh_strategy=self.velocity_strategy,
            ftol=self.convergence_ftol_rel,
            ftol_iter=self.convergence_ftol_iter,
            init_pos=init_pos,
        )

        objective_wrapper = _create_batch_objective(problem, self.n_cores)

        result = optimizer.optimize(
            objective_func=objective_wrapper,
            iters=self.stopping_maxiter,
            verbose=self.verbose,
        )

        res = _process_pyswarms_result(
            result=result,
            n_particles=self.n_particles,
            n_iterations_run=self.stopping_maxiter,
        )

        return res


def _resolve_topology_config(
    config: TopologyConfig,
) -> tuple[Any, dict[str, float | int]]:
    """Resolves the topology config into a pyswarms topology instance and options
    dict."""
    from pyswarms.backend.topology import Pyramid, Random, Ring, Star, VonNeumann

    if isinstance(config, str):
        default_topologies = {
            "star": StarTopology(),
            "ring": RingTopology(),
            "vonneumann": VonNeumannTopology(),
            "random": RandomTopology(),
            "pyramid": PyramidTopology(),
        }
        if config not in default_topologies:
            raise ValueError(f"Unknown topology string: '{config}'")
        config = default_topologies[config]

    topology_instance: Any
    options: dict[str, float | int] = {}

    if isinstance(config, StarTopology):
        topology_instance = Star()
    elif isinstance(config, RingTopology):
        topology_instance = Ring(static=config.static)
        options = {"k": config.k_neighbors, "p": config.p_norm}
    elif isinstance(config, VonNeumannTopology):
        topology_instance = VonNeumann()
        options = {"p": config.p_norm, "r": config.range}
    elif isinstance(config, RandomTopology):
        topology_instance = Random(static=config.static)
        options = {"k": config.k_neighbors}
    elif isinstance(config, PyramidTopology):
        topology_instance = Pyramid(static=config.static)
    else:
        raise TypeError(f"Unsupported topology configuration type: {type(config)}")

    return topology_instance, options


def _build_pso_options_dict(options: BasePSOOptions) -> dict[str, float | int]:
    """Convert structured PSO options to PySwarms format."""
    base_options = {
        "c1": options.cognitive_parameter,
        "c2": options.social_parameter,
        "w": options.inertia_weight,
    }

    # Add topology-specific options if present
    if isinstance(options, LocalBestPSOOptions):
        base_options.update(
            {
                "k": options.k_neighbors,
                "p": options.p_norm,
            }
        )
    elif isinstance(options, GeneralPSOOptions):
        if options.k_neighbors is not None:
            base_options["k"] = options.k_neighbors
        if options.p_norm is not None:
            base_options["p"] = options.p_norm
        if options.vonneumann_range is not None:
            base_options["r"] = options.vonneumann_range

    return base_options


def _build_velocity_clamp(
    velocity_clamp_min: float | None, velocity_clamp_max: float | None
) -> tuple[float, float] | None:
    """Build velocity clamp tuple."""
    clamp = None
    if velocity_clamp_min is not None and velocity_clamp_max is not None:
        clamp = (velocity_clamp_min, velocity_clamp_max)
    return clamp


def _process_pyswarms_result(
    result: tuple[float, NDArray[np.float64]], n_particles: int, n_iterations_run: int
) -> InternalOptimizeResult:
    """Convert PySwarms result to optimagic format."""
    best_cost, best_position = result

    return InternalOptimizeResult(
        x=best_position,
        fun=best_cost,
        success=True,
        message="PySwarms optimization completed",
        n_fun_evals=n_particles * n_iterations_run,
        n_jac_evals=0,
        n_hess_evals=0,
        n_iterations=n_iterations_run,
    )


def _create_batch_objective(
    problem: InternalOptimizationProblem,
    n_cores: int,
) -> Callable[[NDArray[np.float64]], NDArray[np.float64]]:
    """Return an batch objective function."""

    def batch_objective(x: NDArray[np.float64]) -> NDArray[np.float64]:
        """Compute objective values for all particles in x.

        Args:
            x: 2D array of shape (n_particles, n_dimensions) with particle positions.

        Returns:
            1D array of shape (n_particles,) with objective values.

        """
        arguments = [xi for xi in x]
        results = problem.batch_fun(arguments, n_cores=n_cores)

        return np.array(results)

    return batch_objective


def _convert_bounds_to_pyswarms(
    bounds: InternalBounds, n_dimensions: int
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Convert optimagic bounds to PySwarms format."""
    lower_bounds_arr = (
        bounds.lower if bounds.lower is not None else np.zeros(n_dimensions)
    )
    upper_bounds_arr = (
        bounds.upper if bounds.upper is not None else np.ones(n_dimensions)
    )
    if not np.all(np.isfinite(lower_bounds_arr)) or not np.all(
        np.isfinite(upper_bounds_arr)
    ):
        raise ValueError("PySwarms does not support infinite bounds.")

    return (lower_bounds_arr, upper_bounds_arr)


def _create_initial_population(
    x0: NDArray[np.float64],
    n_particles: int,
    bounds: tuple[NDArray[np.float64], NDArray[np.float64]],
    center: float = 1.0,
) -> NDArray[np.float64]:
    """Create initial population with x0 as first particle.

    Args:
        x0: Initial parameter vector
        n_particles: Number of particles in the swarm
        bounds: Tuple of (lower_bounds, upper_bounds) arrays
        center: Scaling factor for initial particle positions around bounds

    Returns:
        Initial population array of shape (n_particles, n_dimensions)

    """
    n_dimensions = len(x0)
    lower_bounds, upper_bounds = bounds

    # Generate random initial positions within the bounds, scaled by center
    init_pos = center * np.random.uniform(
        low=lower_bounds, high=upper_bounds, size=(n_particles, n_dimensions)
    )

    init_pos = np.clip(init_pos, lower_bounds, upper_bounds)
    init_pos[0] = np.clip(x0, lower_bounds, upper_bounds)

    return init_pos
