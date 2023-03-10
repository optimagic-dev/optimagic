from dataclasses import dataclass, replace

import numpy as np

from estimagic.optimization.tranquilo.bounds import Bounds
from estimagic.optimization.tranquilo.volume import (
    get_radius_of_cube_with_volume_of_sphere,
)


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
    out = False
    if bounds is not None and bounds.has_any:
        if bounds.lower is not None:
            lower_binding = np.min(center - bounds.lower) <= sphere_radius
        if bounds.upper is not None:
            upper_binding = np.min(bounds.upper - center) <= sphere_radius
        out = lower_binding or upper_binding
    return out
