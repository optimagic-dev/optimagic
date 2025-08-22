"""Implement PySwarms particle swarm optimization algorithms.

This module provides optimagic-compatible wrappers for PySwarms particle swarm
optimization algorithms including global best, local best, and general PSO variants with
support for different topologies.

"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Literal

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


@dataclass(frozen=True)
class BasePSOOptions:
    """Base PSO parameters shared across all variants."""

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
    disable_history=True,
)
@dataclass(frozen=True)
class PySwarmsGlobalBestPSO(Algorithm):
    r"""Minimize a scalar function using Global Best Particle Swarm Optimization.

    This algorithm uses a swarm of particles that move through the search space,
    where each particle is attracted to both its personal best position and the
    global best position found by the entire swarm. It uses a star topology where
    all particles are connected to the global best particle.

    The position update follows:

    .. math::

        x_i(t+1) = x_i(t) + v_i(t+1)

    where the velocity update is:

    .. math::

        v_{ij}(t+1) = w \cdot v_{ij}(t) + c_1 r_{1j}(t)[y_{ij}(t) - x_{ij}(t)]
                      + c_2 r_{2j}(t)[\hat{y}_j(t) - x_{ij}(t)]

    Here :math:`c_1` and :math:`c_2` control the balance between personal experience
    and swarm knowledge, :math:`w` controls momentum, and :math:`r_{1j}`, :math:`r_{2j}`
    are random numbers from [0,1].

    """

    n_particles: PositiveInt = 50
    """Number of particles in the swarm.

    Typical values: 20-100.

    """

    cognitive_parameter: PositiveFloat = 0.5
    r"""Cognitive parameter :math:`c_1` controlling attraction to personal best."""

    social_parameter: PositiveFloat = 0.3
    r"""Social parameter :math:`c_2` controlling attraction to global best."""

    inertia_weight: PositiveFloat = 0.9
    r"""Inertia weight :math:`w` controlling momentum.

    Higher values promote exploration, lower values promote exploitation. Typical range:
    0.1-0.9.

    """

    convergence_ftol_rel: NonNegativeFloat = CONVERGENCE_FTOL_REL
    """Relative tolerance for convergence based on function value changes."""

    convergence_ftol_iter: PositiveInt = 1
    """Number of iterations to check for convergence."""

    stopping_maxiter: PositiveInt = STOPPING_MAXITER
    """Maximum number of iterations."""

    boundary_strategy: Literal[
        "periodic", "reflective", "shrink", "random", "intermediate"
    ] = "periodic"
    """Strategy for handling out-of-bounds particles: 'periodic' (wrap around),
    'reflective' (bounce), 'shrink' (move to boundary), 'random' (reposition),
    'intermediate' (place between current and boundary)."""

    velocity_strategy: Literal["unmodified", "adjust", "invert", "zero"] = "unmodified"
    """Strategy for out-of-bounds velocities: 'unmodified' (keep), 'adjust' (scale),
    'invert' (reverse), 'zero' (set to zero)."""

    velocity_clamp_min: float | None = None
    """Minimum velocity value for clamping.

    None to disable.

    """

    velocity_clamp_max: float | None = None
    """Maximum velocity value for clamping.

    None to disable.

    """

    n_processes: PositiveInt | None = None
    """Number of processes for parallel evaluation.

    None to disable parallelization.

    """

    center_init: PositiveFloat = 1.0
    """Scaling factor for initial particle positions around search space center."""

    verbose: bool = False
    """Print verbose output."""

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
            x0=x0, n_particles=self.n_particles, bounds=bounds
        )

        optimizer = ps.single.GlobalBestPSO(
            n_particles=self.n_particles,
            dimensions=len(x0),
            options=options,
            bounds=bounds,
            bh_strategy=self.boundary_strategy,
            velocity_clamp=velocity_clamp,
            vh_strategy=self.velocity_strategy,
            center=self.center_init,
            ftol=self.convergence_ftol_rel,
            ftol_iter=self.convergence_ftol_iter,
            init_pos=init_pos,
        )

        objective_wrapper = _create_objective_wrapper(problem)

        result = optimizer.optimize(
            objective_func=objective_wrapper,
            iters=self.stopping_maxiter,
            n_processes=self.n_processes,
            verbose=self.verbose,
        )

        return _process_pyswarms_result(
            result=result,
            n_particles=self.n_particles,
            n_iterations_run=self.stopping_maxiter,
        )

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


def _create_objective_wrapper(
    problem: InternalOptimizationProblem,
) -> Callable[[NDArray[np.float64]], NDArray[np.float64]]:
    """Create objective function wrapper for PySwarms 2D input format."""

    def objective_wrapper(x: NDArray[np.float64]) -> NDArray[np.float64]:
        """Objective wrapper for PySwarms format.

        Args:
            x: 2D array of shape (n_particles, n_dimensions) with particle positions.

        Returns:
            1D array of shape (n_particles,) with objective values.

        """
        return np.array([problem.fun(xi) for xi in x])

    return objective_wrapper


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
) -> NDArray[np.float64]:
    """Create initial population with x0 as first particle."""
    n_dimensions = len(x0)
    lower_bounds, upper_bounds = bounds

    # Generate random initial positions within the bounds
    init_pos = np.random.uniform(
        low=lower_bounds, high=upper_bounds, size=(n_particles, n_dimensions)
    )

    init_pos[0] = np.clip(x0, lower_bounds, upper_bounds)

    return init_pos
