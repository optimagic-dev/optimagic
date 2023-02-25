from typing import NamedTuple

import numpy as np

from estimagic.optimization.algo_options import (
    CONVERGENCE_ABSOLUTE_CRITERION_TOLERANCE,
    CONVERGENCE_ABSOLUTE_PARAMS_TOLERANCE,
    CONVERGENCE_MINIMAL_TRUSTREGION_RADIUS_TOLERANCE,
    CONVERGENCE_RELATIVE_CRITERION_TOLERANCE,
    CONVERGENCE_RELATIVE_GRADIENT_TOLERANCE,
    CONVERGENCE_RELATIVE_PARAMS_TOLERANCE,
)


class Bounds(NamedTuple):
    """Stopping criteria."""

    lower: np.ndarray
    upper: np.ndarray


class StopOptions(NamedTuple):
    """Criteria for stopping without successful convergence."""

    max_iter: int
    max_eval: int
    max_time: float


class ConvOptions(NamedTuple):
    """Criteria for successful convergence."""

    ftol_abs: float = CONVERGENCE_ABSOLUTE_CRITERION_TOLERANCE
    gtol_abs: float = 0.0
    xtol_abs: float = CONVERGENCE_ABSOLUTE_PARAMS_TOLERANCE
    ftol_rel: float = CONVERGENCE_RELATIVE_CRITERION_TOLERANCE
    gtol_rel: float = CONVERGENCE_RELATIVE_GRADIENT_TOLERANCE
    xtol_rel: float = CONVERGENCE_RELATIVE_PARAMS_TOLERANCE
    min_radius: float = CONVERGENCE_MINIMAL_TRUSTREGION_RADIUS_TOLERANCE


class RadiusOptions(NamedTuple):
    """Options for trust-region radius management."""

    min_radius: float = 1e-8
    max_radius: float = 1e6
    initial_radius: float = 0.1
    rho_decrease: float = 0.1
    rho_increase: float = 0.1
    shrinking_factor: float = 0.5
    expansion_factor: float = 2.0
    large_step: float = 0.5
    max_radius_to_step_ratio: float = np.inf


class Region(NamedTuple):
    center: np.ndarray
    radius: float
    shape: str


class HistorySearchOptions(NamedTuple):
    radius_type: str = "circumscribed"
    radius_factor: float = 1.5


class AcceptanceOptions(NamedTuple):
    radius_factor: float = 1e-5
    confidence_level: float = 0.8
    power_level: float = 0.8
    n_initial: int = 5
    n_min: int = 5
    n_max: int = 100
    min_improvement: float = 0.0
    sampler: str = "ball"


class StagnationOptions(NamedTuple):
    min_relative_step: float = 0.025
    sample_increment: int = 1
    max_trials: int = 1
    drop: bool = False
