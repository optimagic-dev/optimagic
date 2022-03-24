from typing import NamedTuple

import numpy as np


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

    pass


class RadiusOptions(NamedTuple):
    """Options for trust-region radius management"""

    min_radius: float = 1e-6
    max_radius: float = 1e6
    initial_radius: float = 0.1
    rho_decrease: float = 0.1
    rho_increase: float = 0.1
    shrinking_factor: float = 0.5
    expansion_factor: float = 2.0
    large_step: float = 0.5
    max_radius_to_step_ratio: float = np.inf


class TrustRegion(NamedTuple):
    center: np.ndarray
    radius: float
