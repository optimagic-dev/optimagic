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
    radius: float
    bounds: Bounds = None

    @property
    def shape(self) -> str:
        any_bounds_binding = _any_bounds_binding(
            bounds=self.bounds, center=self.center, radius=self.radius
        )
        return "cube" if any_bounds_binding else "sphere"

    @property
    def cube_bounds(self) -> Bounds:
        if self.shape == "sphere":
            radius = self.radius
        else:
            radius = get_radius_of_cube_with_volume_of_sphere(
                self.radius, len(self.center), scaling_factor=1.0
            )
        cube_bounds = _get_cube_bounds(
            center=self.center, radius=radius, bounds=self.bounds
        )
        return cube_bounds

    @property
    def cube_center(self) -> np.ndarray:
        if self.shape == "sphere":
            _center = self.center
        else:
            cube_bounds = self.cube_bounds
            _center = (cube_bounds.lower + cube_bounds.upper) / 2
        return _center

    def map_to_unit(self, x: np.ndarray) -> np.ndarray:
        """Map points inside the trustregion to the unit sphere or cube."""
        return _map_to_unit(self.cube_bounds, x=x)

    def map_from_unit(self, x: np.ndarray) -> np.ndarray:
        """Map points inside the unit sphere or cube to the trustregion."""
        return _map_from_unit(self.cube_bounds, x=x)

    # make it behave like a NamedTuple
    def _replace(self, **kwargs):
        return replace(self, **kwargs)


def _map_to_unit(cube_bounds, x):
    """Map points inside the trustregion to the unit sphere or cube."""
    out = 2 * (x - cube_bounds.lower) / (cube_bounds.upper - cube_bounds.lower) - 1
    return out


def _map_from_unit(cube_bounds, x):
    """Map points inside the unit sphere or cube to the trustregion."""
    out = (cube_bounds.upper - cube_bounds.lower) * (x + 1) / 2 + cube_bounds.lower
    return out


def _get_cube_bounds(center, radius, bounds):
    """Get new bounds that define the intersection of the trustregion and the bounds."""
    lower_bounds = center - radius
    upper_bounds = center + radius

    if bounds is not None and bounds.lower is not None:
        lower_bounds = np.clip(lower_bounds, bounds.lower, np.inf)

    if bounds is not None and bounds.upper is not None:
        upper_bounds = np.clip(upper_bounds, -np.inf, bounds.upper)

    return Bounds(lower=lower_bounds, upper=upper_bounds)


def _any_bounds_binding(bounds, center, radius):
    """Check if any bound is binding, i.e. inside the trustregion."""
    out = False
    if bounds is not None and bounds.has_any:
        if bounds.lower is not None:
            lower_binding = np.min(center - bounds.lower) <= radius
        if bounds.upper is not None:
            upper_binding = np.min(bounds.upper - center) <= radius
        out = lower_binding or upper_binding
    return out
