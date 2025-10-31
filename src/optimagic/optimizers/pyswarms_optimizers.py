"""Implement PySwarms particle swarm optimization algorithms.

This module provides optimagic-compatible wrappers for PySwarms particle swarm
optimization algorithms including global best, local best, and general PSO variants with
support for different topologies.

"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Any, Callable, Literal

import numpy as np
from numpy.typing import NDArray

from optimagic import mark
from optimagic.config import IS_PYSWARMS_INSTALLED
from optimagic.exceptions import NotInstalledError
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

    range_param: PositiveInt = 1
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


# ======================================================================================
# Common PSO Options
# ======================================================================================


@dataclass(frozen=True)
class PSOCommonOptions:
    """Common options for PySwarms optimizers."""

    n_particles: PositiveInt = 10
    """Number of particles in the swarm."""

    cognitive_parameter: PositiveFloat = 0.5
    """Cognitive parameter (c1) - attraction to personal best."""

    social_parameter: PositiveFloat = 0.3
    """Social parameter (c2) - attraction to neighborhood/global best."""

    inertia_weight: PositiveFloat = 0.9
    """Inertia weight (w) - momentum control."""

    stopping_maxiter: PositiveInt = 1000
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

    convergence_ftol_rel: NonNegativeFloat = 0
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
    """Random seed for initial positions.

    For full reproducibility, set a global seed with `np.random.seed()`.

    """


# ======================================================================================
# Algorithm Classes
# ======================================================================================


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
class PySwarmsGlobalBestPSO(Algorithm, PSOCommonOptions):
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

    def _solve_internal_problem(
        self, problem: InternalOptimizationProblem, x0: NDArray[np.float64]
    ) -> InternalOptimizeResult:
        if not IS_PYSWARMS_INSTALLED:
            raise NotInstalledError(PYSWARMS_NOT_INSTALLED_ERROR)

        import pyswarms as ps

        pso_options_dict = {
            "c1": self.cognitive_parameter,
            "c2": self.social_parameter,
            "w": self.inertia_weight,
        }
        optimizer_kwargs = {"options": pso_options_dict}

        res = _pyswarms_internal(
            problem=problem,
            x0=x0,
            optimizer_class=ps.single.GlobalBestPSO,
            optimizer_kwargs=optimizer_kwargs,
            algo_options=self,
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
class PySwarmsLocalBestPSO(Algorithm, PSOCommonOptions):
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

    topology: RingTopology = RingTopology()
    """Configuration for the Ring topology.

    This algorithm uses a fixed ring topology where particles are connected to their
    local neighbors. This parameter allows customization of the number of neighbors,
    distance metric, and whether the topology remains static throughout optimization.

    """

    def _solve_internal_problem(
        self, problem: InternalOptimizationProblem, x0: NDArray[np.float64]
    ) -> InternalOptimizeResult:
        if not IS_PYSWARMS_INSTALLED:
            raise NotInstalledError(PYSWARMS_NOT_INSTALLED_ERROR)

        import pyswarms as ps

        pso_options_dict = {
            "c1": self.cognitive_parameter,
            "c2": self.social_parameter,
            "w": self.inertia_weight,
            "k": self.topology.k_neighbors,
            "p": self.topology.p_norm,
        }

        optimizer_kwargs = {
            "options": pso_options_dict,
            "static": self.topology.static,
        }

        res = _pyswarms_internal(
            problem=problem,
            x0=x0,
            optimizer_class=ps.single.LocalBestPSO,
            optimizer_kwargs=optimizer_kwargs,
            algo_options=self,
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
class PySwarmsGeneralPSO(Algorithm, PSOCommonOptions):
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

    topology: Literal["star", "ring", "vonneumann", "random", "pyramid"] | Topology = (
        "star"
    )
    """Topology structure for particle communication.

    The `topology` can be specified in two ways:

    1.  **By name (string):** e.g., ``"star"``, ``"ring"``. This uses the default
        parameter values for that topology.
    2.  **By dataclass instance:** e.g., ``RingTopology(k_neighbors=5, p_norm=1)``.
        This allows for detailed configuration of topology-specific parameters.

    Available topologies: ``StarTopology``, ``RingTopology``, ``VonNeumannTopology``,
    ``RandomTopology``, ``PyramidTopology``.

    """

    def _solve_internal_problem(
        self, problem: InternalOptimizationProblem, x0: NDArray[np.float64]
    ) -> InternalOptimizeResult:
        if not IS_PYSWARMS_INSTALLED:
            raise NotInstalledError(PYSWARMS_NOT_INSTALLED_ERROR)

        import pyswarms as ps

        pyswarms_topology, topology_options = _resolve_topology_config(self.topology)
        base_options = {
            "c1": self.cognitive_parameter,
            "c2": self.social_parameter,
            "w": self.inertia_weight,
        }
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
            algo_options=self,
        )

        return res


def _pyswarms_internal(
    problem: InternalOptimizationProblem,
    x0: NDArray[np.float64],
    optimizer_class: Any,
    optimizer_kwargs: dict[str, Any],
    algo_options: PSOCommonOptions,
) -> InternalOptimizeResult:
    """Internal function for PySwarms optimization.

    Args:
        problem: Internal optimization problem.
        x0: Initial parameter vector.
        optimizer_class: PySwarms optimizer class to use.
        optimizer_kwargs: Arguments for optimizer class.
        algo_options: The PySwarms common options.

    Returns:
        InternalOptimizeResult: Internal optimization result.

    """
    if algo_options.seed is not None:
        warnings.warn(
            "The 'seed' parameter only makes initial particle positions reproducible. "
            "PySwarms still uses NumPy's global random functions for generating "
            "velocities, updating coefficients, and handling other stochastic "
            "operations. For fully deterministic results, set a global seed with "
            "'np.random.seed()' before running the optimizer.",
            UserWarning,
        )

    rng = np.random.default_rng(algo_options.seed)

    velocity_clamp = _build_velocity_clamp(
        algo_options.velocity_clamp_min, algo_options.velocity_clamp_max
    )
    bounds = _get_pyswarms_bounds(problem.bounds)

    if algo_options.initial_positions is not None:
        init_pos = np.array(
            [
                problem.converter.params_to_internal(position)
                for position in algo_options.initial_positions
            ]
        )
    else:
        init_pos = _create_initial_positions(
            x0=x0,
            n_particles=algo_options.n_particles,
            bounds=bounds,
            center=algo_options.center_init,
            rng=rng,
        )

    optimizer = optimizer_class(
        n_particles=algo_options.n_particles,
        dimensions=len(x0),
        bounds=bounds,
        init_pos=init_pos,
        velocity_clamp=velocity_clamp,
        oh_strategy=algo_options.oh_strategy,
        bh_strategy=algo_options.boundary_strategy,
        vh_strategy=algo_options.velocity_strategy,
        ftol=algo_options.convergence_ftol_rel,
        ftol_iter=algo_options.convergence_ftol_iter,
        **optimizer_kwargs,
    )

    objective_wrapper = _create_batch_objective(problem, algo_options.n_cores)

    result = optimizer.optimize(
        objective_func=objective_wrapper,
        iters=algo_options.stopping_maxiter,
        verbose=algo_options.verbose,
    )

    res = _process_pyswarms_result(result=result, optimizer=optimizer)

    return res


def _resolve_topology_config(
    config: Literal["star", "ring", "vonneumann", "random", "pyramid"] | Topology,
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
        options = {"p": config.p_norm, "r": config.range_param}
    elif isinstance(config, RandomTopology):
        topology_instance = Random(static=config.static)
        options = {"k": config.k_neighbors}
    elif isinstance(config, PyramidTopology):
        topology_instance = Pyramid(static=config.static)
    else:
        raise TypeError(f"Unsupported topology configuration type: {type(config)}")

    return topology_instance, options


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

    init_pos[0] = x0
    init_pos = np.clip(init_pos, lower_bounds, upper_bounds)

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
