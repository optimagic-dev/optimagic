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
    PyTree,
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
class Topology:
    """Base class for all topology configurations."""


@dataclass(frozen=True)
class StarTopology(Topology):
    """Star topology configuration.

    All particles are connected to the global best.

    """


@dataclass(frozen=True)
class RingTopology(Topology):
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
class VonNeumannTopology(Topology):
    """Von Neumann topology configuration.

    Particles are arranged on a 2D grid.

    """

    p_norm: Literal[1, 2] = 2
    """Distance metric for neighbor selection: 1 (Manhattan), 2 (Euclidean)."""

    range: PositiveInt = 1
    r"""Range parameter :math:`r` for neighborhood size."""


@dataclass(frozen=True)
class PyramidTopology(Topology):
    """Pyramid topology configuration."""

    static: bool = False
    """Whether to use a static or dynamic pyramid topology.

    When True, the neighborhood structure is fixed throughout optimization. When False,
    neighbors are recomputed at each iteration based on current particle positions.

    """


@dataclass(frozen=True)
class RandomTopology(Topology):
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
    Topology,
]

# ======================================================================================
# 2. PSO Options Classes
# ======================================================================================


@dataclass(frozen=True)
class PSOOptions:
    """Common PSO parameters used by all PSO variants."""

    cognitive_parameter: PositiveFloat = 0.5
    """Cognitive parameter (c1) - attraction to personal best."""

    social_parameter: PositiveFloat = 0.3
    """Social parameter (c2) - attraction to neighborhood/global best."""

    inertia_weight: PositiveFloat = 0.9
    """Inertia weight (w) - momentum control."""


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

    This algorithm is an adaptation of the original Particle Swarm Optimization method
    by :cite:`Kennedy1995`

    """

    n_particles: PositiveInt = 50
    """Number of particles in the swarm."""

    options: PSOOptions = PSOOptions()
    """PSO hyperparameters controlling particle behavior."""

    stopping_maxiter: PositiveInt = STOPPING_MAXITER
    """Maximum number of iterations."""

    initial_positions: list[PyTree] | None = None
    """Option to set the initial particle positions.

    If None, positions are generated randomly within the given bounds, or within [0, 1]
    if bounds are not specified.

    """

    oh_strategy: dict[str, str] | None = None
    """Dictionary of strategies for time-varying options."""

    boundary_strategy: Literal[
        "periodic", "reflective", "shrink", "random", "intermediate"
    ] = "periodic"
    """Strategy for handling out-of-bounds particles.

    Available options: periodic (default),
    reflective, shrink, random, intermediate.

    """

    velocity_strategy: Literal["unmodified", "adjust", "invert", "zero"] = "unmodified"
    """Strategy for handling out-of-bounds velocities.

    Available options: unmodified (default),
    adjust, invert, zero.

    """

    velocity_clamp_min: float | None = None
    """Minimum velocity limit for particles."""

    velocity_clamp_max: float | None = None
    """Maximum velocity limit for particles."""

    convergence_ftol_rel: NonNegativeFloat = CONVERGENCE_FTOL_REL
    """Stop when relative change in objective function is less than this value."""

    convergence_ftol_iter: PositiveInt = 1
    """Number of iterations to check for convergence."""

    n_cores: PositiveInt = 1
    """Number of cores for parallel evaluation."""

    center_init: PositiveFloat = 1.0
    """Scaling factor for initial particle positions."""

    verbose: bool = False
    """Enable or disable the logs and progress bar."""

    seed: int | None = None
    """Random seed for reproducibility."""

    def _solve_internal_problem(
        self, problem: InternalOptimizationProblem, x0: NDArray[np.float64]
    ) -> InternalOptimizeResult:
        if not IS_PYSWARMS_INSTALLED:
            raise NotInstalledError(PYSWARMS_NOT_INSTALLED_ERROR)

        import pyswarms as ps

        pso_options_dict = _pso_options_to_dict(self.options)
        optimizer_kwargs = {"options": pso_options_dict}

        res = _pyswarms_internal(
            problem=problem,
            x0=x0,
            optimizer_class=ps.single.GlobalBestPSO,
            optimizer_kwargs=optimizer_kwargs,
            n_particles=self.n_particles,
            stopping_maxiter=self.stopping_maxiter,
            initial_positions=self.initial_positions,
            oh_strategy=self.oh_strategy,
            boundary_strategy=self.boundary_strategy,
            velocity_strategy=self.velocity_strategy,
            velocity_clamp_min=self.velocity_clamp_min,
            velocity_clamp_max=self.velocity_clamp_max,
            convergence_ftol_rel=self.convergence_ftol_rel,
            convergence_ftol_iter=self.convergence_ftol_iter,
            n_cores=self.n_cores,
            center_init=self.center_init,
            verbose=self.verbose,
            seed=self.seed,
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

    The algorithm is based on the original Particle Swarm Optimization method by
    :cite:`Kennedy1995` and the local best concept introduced in
    :cite:`EberhartKennedy1995`.

    """

    n_particles: PositiveInt = 50
    """Number of particles in the swarm."""

    options: PSOOptions = PSOOptions()
    """PSO hyperparameters controlling particle behavior."""

    topology: RingTopology = RingTopology()
    """Configuration for the Ring topology.

    This algorithm uses a fixed ring topology where particles are connected to their
    local neighbors. This parameter allows customization of the number of neighbors,
    distance metric, and whether the topology remains static throughout optimization.

    """

    stopping_maxiter: PositiveInt = STOPPING_MAXITER
    """Maximum number of iterations."""

    initial_positions: list[PyTree] | None = None
    """Option to set the initial particle positions.

    If None, positions are generated randomly within the given bounds, or within [0, 1]
    if bounds are not specified.

    """

    oh_strategy: dict[str, str] | None = None
    """Dictionary of strategies for time-varying options."""

    boundary_strategy: Literal[
        "periodic", "reflective", "shrink", "random", "intermediate"
    ] = "periodic"
    """Strategy for handling out-of-bounds particles.

    Available options: periodic (default),
    reflective, shrink, random, intermediate.

    """

    velocity_strategy: Literal["unmodified", "adjust", "invert", "zero"] = "unmodified"
    """Strategy for handling out-of-bounds velocities.

    Available options: unmodified (default),
    adjust, invert, zero.

    """

    velocity_clamp_min: float | None = None
    """Minimum velocity limit for particles."""

    velocity_clamp_max: float | None = None
    """Maximum velocity limit for particles."""

    convergence_ftol_rel: NonNegativeFloat = CONVERGENCE_FTOL_REL
    """Stop when relative change in objective function is less than this value."""

    convergence_ftol_iter: PositiveInt = 1
    """Number of iterations to check for convergence."""

    n_cores: PositiveInt = 1
    """Number of cores for parallel evaluation."""

    center_init: PositiveFloat = 1.0
    """Scaling factor for initial particle positions."""

    verbose: bool = False
    """Enable or disable the logs and progress bar."""

    seed: int | None = None
    """Random seed for reproducibility."""

    def _solve_internal_problem(
        self, problem: InternalOptimizationProblem, x0: NDArray[np.float64]
    ) -> InternalOptimizeResult:
        if not IS_PYSWARMS_INSTALLED:
            raise NotInstalledError(PYSWARMS_NOT_INSTALLED_ERROR)

        import pyswarms as ps

        base_options = _pso_options_to_dict(self.options)
        topology_options = {
            "k": self.topology.k_neighbors,
            "p": self.topology.p_norm,
        }
        pso_options_dict = {**base_options, **topology_options}

        optimizer_kwargs = {
            "options": pso_options_dict,
            "static": self.topology.static,
        }

        res = _pyswarms_internal(
            problem=problem,
            x0=x0,
            optimizer_class=ps.single.LocalBestPSO,
            optimizer_kwargs=optimizer_kwargs,
            n_particles=self.n_particles,
            stopping_maxiter=self.stopping_maxiter,
            initial_positions=self.initial_positions,
            oh_strategy=self.oh_strategy,
            boundary_strategy=self.boundary_strategy,
            velocity_strategy=self.velocity_strategy,
            velocity_clamp_min=self.velocity_clamp_min,
            velocity_clamp_max=self.velocity_clamp_max,
            convergence_ftol_rel=self.convergence_ftol_rel,
            convergence_ftol_iter=self.convergence_ftol_iter,
            n_cores=self.n_cores,
            center_init=self.center_init,
            verbose=self.verbose,
            seed=self.seed,
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

    This algorithm is based on the original Particle Swarm Optimization method by
    :cite:`Kennedy1995` with configurable topology structures. For topology references,
    see :cite:`Lane2008SpatialPSO, Ni2013`.

    """

    n_particles: PositiveInt = 50
    """Number of particles in the swarm."""

    options: PSOOptions = PSOOptions()
    """PSO hyperparameters controlling particle behavior."""

    topology: TopologyConfig = "star"
    """Topology structure for particle communication.

    The `topology` can be specified in two ways:

    1.  **By name (string):** e.g., ``"star"``, ``"ring"``. This uses the default
        parameters for that topology.
    2.  **By dataclass instance:** e.g., ``RingTopology(k_neighbors=5, p_norm=1)``.
        This allows for detailed configuration of topology-specific parameters.

    Available topologies: ``StarTopology``, ``RingTopology``, ``VonNeumannTopology``,
    ``RandomTopology``, ``PyramidTopology``.

    """

    stopping_maxiter: PositiveInt = STOPPING_MAXITER
    """Maximum number of iterations."""

    initial_positions: list[PyTree] | None = None
    """Option to set the initial particle positions.

    If None, positions are generated randomly within the given bounds, or within [0, 1]
    if bounds are not specified.

    """

    oh_strategy: dict[str, str] | None = None
    """Dictionary of strategies for time-varying options."""

    boundary_strategy: Literal[
        "periodic", "reflective", "shrink", "random", "intermediate"
    ] = "periodic"
    """Strategy for handling out-of-bounds particles.

    Available options: periodic (default),
    reflective, shrink, random, intermediate.

    """

    velocity_strategy: Literal["unmodified", "adjust", "invert", "zero"] = "unmodified"
    """Strategy for handling out-of-bounds velocities.

    Available options: unmodified (default),
    adjust, invert, zero.

    """

    velocity_clamp_min: float | None = None
    """Minimum velocity limit for particles."""

    velocity_clamp_max: float | None = None
    """Maximum velocity limit for particles."""

    convergence_ftol_rel: NonNegativeFloat = CONVERGENCE_FTOL_REL
    """Stop when relative change in objective function is less than this value."""

    convergence_ftol_iter: PositiveInt = 1
    """Number of iterations to check for convergence."""

    n_cores: PositiveInt = 1
    """Number of cores for parallel evaluation."""

    center_init: PositiveFloat = 1.0
    """Scaling factor for initial particle positions."""

    verbose: bool = False
    """Enable or disable the logs and progress bar."""

    seed: int | None = None
    """Random seed for reproducibility."""

    def _solve_internal_problem(
        self, problem: InternalOptimizationProblem, x0: NDArray[np.float64]
    ) -> InternalOptimizeResult:
        if not IS_PYSWARMS_INSTALLED:
            raise NotInstalledError(PYSWARMS_NOT_INSTALLED_ERROR)

        import pyswarms as ps

        pyswarms_topology, topology_options = _resolve_topology_config(self.topology)
        base_options = _pso_options_to_dict(self.options)
        pso_options_dict = {**base_options, **topology_options}

        optimizer_kwargs = {
            "options": pso_options_dict,
            "topology": pyswarms_topology,
        }

        res = _pyswarms_internal(
            problem=problem,
            x0=x0,
            optimizer_class=ps.single.GeneralOptimizerPSO,
            optimizer_kwargs=optimizer_kwargs,
            n_particles=self.n_particles,
            stopping_maxiter=self.stopping_maxiter,
            initial_positions=self.initial_positions,
            oh_strategy=self.oh_strategy,
            boundary_strategy=self.boundary_strategy,
            velocity_strategy=self.velocity_strategy,
            velocity_clamp_min=self.velocity_clamp_min,
            velocity_clamp_max=self.velocity_clamp_max,
            convergence_ftol_rel=self.convergence_ftol_rel,
            convergence_ftol_iter=self.convergence_ftol_iter,
            n_cores=self.n_cores,
            center_init=self.center_init,
            verbose=self.verbose,
            seed=self.seed,
        )

        return res


def _pyswarms_internal(
    problem: InternalOptimizationProblem,
    x0: NDArray[np.float64],
    optimizer_class: Any,
    optimizer_kwargs: dict[str, Any],
    n_particles: int,
    stopping_maxiter: int,
    initial_positions: list[PyTree] | None,
    oh_strategy: dict[str, str] | None,
    boundary_strategy: str,
    velocity_strategy: str,
    velocity_clamp_min: float | None,
    velocity_clamp_max: float | None,
    convergence_ftol_rel: float,
    convergence_ftol_iter: int,
    n_cores: int,
    center_init: float,
    verbose: bool,
    seed: int | None,
) -> InternalOptimizeResult:
    """Internal function for PySwarms optimization.

    Args:
        problem: Internal optimization problem
        x0: Initial parameter vector
        optimizer_class: PySwarms optimizer class to use
        optimizer_kwargs: Arguments for optimizer class
        n_particles: Number of particles in the swarm
        stopping_maxiter: Maximum number of iterations before stopping
        initial_positions: User-provided initial positions for particles
        oh_strategy: Options handling strategy
        boundary_strategy: Strategy for handling boundary constraints
        velocity_strategy: Strategy for velocity updates
        velocity_clamp_min: Minimum velocity bound
        velocity_clamp_max: Maximum velocity bound
        convergence_ftol_rel: Relative tolerance for convergence detection
        convergence_ftol_iter: Number of iterations for convergence check
        n_cores: Number of cores for parallel evaluation
        center_init: Scaling factor for initial particle positions
        verbose: Enable verbose output during optimization
        seed: Random seed for reproducibility

    Returns:
        InternalOptimizeResult: Internal optimization result

    """
    rng = np.random.default_rng(seed)

    velocity_clamp = _build_velocity_clamp(velocity_clamp_min, velocity_clamp_max)
    bounds = _get_pyswarms_bounds(problem.bounds)

    if initial_positions is not None:
        init_pos = np.array(
            [
                problem.converter.params_to_internal(position)
                for position in initial_positions
            ]
        )
    else:
        init_pos = _create_initial_positions(
            x0=x0,
            n_particles=n_particles,
            bounds=bounds,
            center=center_init,
            rng=rng,
        )

    optimizer = optimizer_class(
        n_particles=n_particles,
        dimensions=len(x0),
        bounds=bounds,
        velocity_clamp=velocity_clamp,
        init_pos=init_pos,
        ftol=convergence_ftol_rel,
        ftol_iter=convergence_ftol_iter,
        bh_strategy=boundary_strategy,
        vh_strategy=velocity_strategy,
        oh_strategy=oh_strategy,
        **optimizer_kwargs,
    )

    objective_wrapper = _create_batch_objective(problem, n_cores)

    result = optimizer.optimize(
        objective_func=objective_wrapper,
        iters=stopping_maxiter,
        verbose=verbose,
    )

    res = _process_pyswarms_result(result=result, optimizer=optimizer)

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


def _pso_options_to_dict(options: PSOOptions) -> dict[str, float | int]:
    """Convert option parameters to PySwarms format."""
    pso_options = {
        "c1": options.cognitive_parameter,
        "c2": options.social_parameter,
        "w": options.inertia_weight,
    }

    return pso_options


def _build_velocity_clamp(
    velocity_clamp_min: float | None, velocity_clamp_max: float | None
) -> tuple[float, float] | None:
    """Build velocity clamp tuple."""
    clamp = None
    if velocity_clamp_min is not None and velocity_clamp_max is not None:
        clamp = (velocity_clamp_min, velocity_clamp_max)
    return clamp


def _get_pyswarms_bounds(
    bounds: InternalBounds,
) -> tuple[NDArray[np.float64], NDArray[np.float64]] | None:
    """Convert optimagic bounds to PySwarms format."""
    pyswarms_bounds = None

    if bounds.lower is not None and bounds.upper is not None:
        if not np.all(np.isfinite(bounds.lower)) or not np.all(
            np.isfinite(bounds.upper)
        ):
            raise ValueError("PySwarms does not support infinite bounds.")

        pyswarms_bounds = (bounds.lower, bounds.upper)

    return pyswarms_bounds


def _create_initial_positions(
    x0: NDArray[np.float64],
    n_particles: int,
    bounds: tuple[NDArray[np.float64], NDArray[np.float64]] | None,
    center: float,
    rng: np.random.Generator,
) -> NDArray[np.float64]:
    """Create an initial swarm positions.

    Args:
        x0: Initial parameter vector.
        n_particles: Number of particles in the swarm.
        bounds: Tuple of (lower_bounds, upper_bounds) arrays or None.
        center: Scaling factor for initial particle positions around bounds.
        rng: NumPy random number generator instance.

    Returns:
        Initial positions array of shape (n_particles, n_dimensions)
        where each row represents one particle's starting position.

    """
    n_dimensions = len(x0)
    if bounds is None:
        lower_bounds: NDArray[np.float64] = np.zeros(n_dimensions, dtype=np.float64)
        upper_bounds: NDArray[np.float64] = np.ones(n_dimensions, dtype=np.float64)
    else:
        lower_bounds, upper_bounds = bounds

    # Generate random initial positions within the bounds, scaled by center
    init_pos = center * rng.uniform(
        low=lower_bounds, high=upper_bounds, size=(n_particles, n_dimensions)
    )

    init_pos = np.clip(init_pos, lower_bounds, upper_bounds)
    init_pos[0] = np.clip(x0, lower_bounds, upper_bounds)

    return init_pos


def _create_batch_objective(
    problem: InternalOptimizationProblem,
    n_cores: int,
) -> Callable[[NDArray[np.float64]], NDArray[np.float64]]:
    """Return an batch objective function."""

    def batch_objective(positions: NDArray[np.float64]) -> NDArray[np.float64]:
        """Compute objective values for all particles in positions.

        Args:
            positions: 2D array of shape (n_particles, n_dimensions) with
            particle positions.

        Returns:
            1D array of shape (n_particles,) with objective values.

        """
        arguments = [position for position in positions]
        results = problem.batch_fun(arguments, n_cores=n_cores)

        return np.array(results)

    return batch_objective


def _process_pyswarms_result(
    result: tuple[float, NDArray[np.float64]], optimizer: Any
) -> InternalOptimizeResult:
    """Convert PySwarms result to optimagic format."""
    best_cost, best_position = result
    n_iterations = len(optimizer.cost_history)
    n_particles = optimizer.n_particles

    return InternalOptimizeResult(
        x=best_position,
        fun=best_cost,
        success=True,
        message="PySwarms optimization completed",
        n_fun_evals=n_particles * n_iterations,
        n_jac_evals=0,
        n_hess_evals=0,
        n_iterations=n_iterations,
    )
