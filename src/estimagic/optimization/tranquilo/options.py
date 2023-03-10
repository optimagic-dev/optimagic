from dataclasses import dataclass, replace
from typing import NamedTuple

import numpy as np

from estimagic.optimization.algo_options import (
    CONVERGENCE_RELATIVE_CRITERION_TOLERANCE,
    CONVERGENCE_RELATIVE_GRADIENT_TOLERANCE,
)
from estimagic.optimization.tranquilo.volume import (
    get_radius_of_cube_with_volume_of_sphere,
)


@dataclass
class Bounds:
    """Parameter bounds."""

    lower: np.ndarray
    upper: np.ndarray

    def __post_init__(self):
        self.has_any = _check_if_there_are_bounds(self.lower, self.upper)


@dataclass
class Region:
    """Trust region."""

    center: np.ndarray
    sphere_radius: float
    bounds: Bounds = None

    @property
    def cube_radius(self) -> float:
        return get_radius_of_cube_with_volume_of_sphere(
            sphere_radius=self.sphere_radius, dim=len(self.center), scaling_factor=1
        )

    @property
    def radius(self) -> float:
        return self.sphere_radius if self.shape == "sphere" else self.cube_radius

    @property
    def shape(self) -> str:
        any_bounds_binding = _any_bounds_binding(
            bounds=self.bounds, center=self.center, sphere_radius=self.sphere_radius
        )
        return "cube" if any_bounds_binding else "sphere"

    @property
    def effective_bounds(self) -> Bounds:
        return _get_effective_bounds(
            center=self.center, radius=self.radius, bounds=self.bounds
        )

    # make it behave like a NamedTuple
    def _replace(self, **kwargs):
        return replace(self, **kwargs)


def _get_effective_bounds(center, radius, bounds):
    lower_bounds = center - radius
    upper_bounds = center + radius

    if bounds is not None and bounds.lower is not None:
        lower_bounds = np.clip(lower_bounds, bounds.lower, np.inf)

    if bounds is not None and bounds.upper is not None:
        upper_bounds = np.clip(upper_bounds, -np.inf, bounds.upper)

    return Bounds(lower=lower_bounds, upper=upper_bounds)


def _any_bounds_binding(bounds, center, sphere_radius):
    if bounds is None or not bounds.has_any:
        out = False
    else:
        lower_binding = np.min(center - bounds.lower) <= sphere_radius
        upper_binding = np.min(bounds.upper - center) <= sphere_radius
        out = lower_binding or upper_binding
    return out


class StopOptions(NamedTuple):
    """Criteria for stopping without successful convergence."""

    max_iter: int
    max_eval: int
    max_time: float


class ConvOptions(NamedTuple):
    """Criteria for successful convergence."""

    ftol_abs: float = 0.0
    gtol_abs: float = 0.0
    xtol_abs: float = 0.0
    ftol_rel: float = CONVERGENCE_RELATIVE_CRITERION_TOLERANCE
    gtol_rel: float = CONVERGENCE_RELATIVE_GRADIENT_TOLERANCE
    xtol_rel: float = 1e-8
    min_radius: float = 0.0


class RadiusOptions(NamedTuple):
    """Options for trust-region radius management."""

    min_radius: float = 1e-6
    max_radius: float = 1e6
    initial_radius: float = 0.1
    rho_decrease: float = 0.1
    rho_increase: float = 0.1
    shrinking_factor: float = 0.5
    expansion_factor: float = 2.0
    large_step: float = 0.5
    max_radius_to_step_ratio: float = np.inf


class AcceptanceOptions(NamedTuple):
    confidence_level: float = 0.8
    power_level: float = 0.8
    n_initial: int = 5
    n_min: int = 5
    n_max: int = 100
    min_improvement: float = 0.0
    sampler: str = "ball"


class StagnationOptions(NamedTuple):
    min_relative_step_keep: float = 0.125
    min_relative_step: float = 0.05
    sample_increment: int = 1
    max_trials: int = 1
    drop: bool = True


def _check_if_there_are_bounds(lb, ub):
    out = False
    if lb is not None and np.isfinite(lb).any():
        out = True
    if ub is not None and np.isfinite(ub).any():
        out = True
    return out
