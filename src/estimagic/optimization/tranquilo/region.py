from typing import Union
from dataclasses import dataclass, replace

import numpy as np

from estimagic.optimization.tranquilo.bounds import Bounds
from estimagic.optimization.tranquilo.volume import (
    get_radius_of_cube_with_volume_of_sphere,
)


@dataclass(frozen=True)
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
            raise AttributeError(
                "The trustregion is a sphere, and thus has no cube bounds."
            )
        radius = get_radius_of_cube_with_volume_of_sphere(self.radius, len(self.center))
        bounds = _get_cube_bounds(center=self.center, radius=radius, bounds=self.bounds)
        return bounds

    @property
    def cube_center(self) -> np.ndarray:
        if self.shape == "sphere":
            raise AttributeError(
                "The trustregion is a sphere, and thus has no cube center."
            )
        center = _get_cube_center(bounds=self.cube_bounds)
        return center

    @property
    def effective_center(self) -> np.ndarray:
        center = self.center if self.shape == "sphere" else self.cube_center
        return center

    @property
    def effective_radius(self) -> Union[float, np.ndarray]:
        if self.shape == "sphere":
            radius = self.radius
        else:
            radius = _get_cube_radius(self.cube_bounds)
        return radius

    def map_to_unit(self, x: np.ndarray) -> np.ndarray:
        """Map points from the trustregion to the unit sphere or cube."""
        if self.shape == "sphere":
            out = _map_to_unit_sphere(x, center=self.center, radius=self.radius)
        else:
            out = _map_to_unit_cube(x, cube_bounds=self.cube_bounds)
        return out

    def map_from_unit(self, x: np.ndarray) -> np.ndarray:
        """Map points from the unit sphere or cube to the trustregion."""
        if self.shape == "sphere":
            out = _map_from_unit_sphere(x, center=self.center, radius=self.radius)
        else:
            cube_bounds = self.cube_bounds
            out = _map_from_unit_cube(x, cube_bounds=cube_bounds)
            # Bounds may not be satisfied exactly due to numerical inaccuracies.
            out = np.clip(out, cube_bounds.lower, cube_bounds.upper)
        return out

    # make it behave like a NamedTuple
    def _replace(self, **kwargs):
        return replace(self, **kwargs)


def _map_to_unit_cube(x, cube_bounds):
    """Map points from the trustregion to the unit cube."""
    out = 2 * (x - cube_bounds.lower) / (cube_bounds.upper - cube_bounds.lower) - 1
    return out


def _map_to_unit_sphere(x, center, radius):
    """Map points from the trustregion to the unit sphere."""
    out = (x - center) / radius
    return out


def _map_from_unit_cube(x, cube_bounds):
    """Map points from the unit cube to the trustregion."""
    out = (cube_bounds.upper - cube_bounds.lower) * (x + 1) / 2 + cube_bounds.lower
    return out


def _map_from_unit_sphere(x, center, radius):
    """Map points from the unit sphere to the trustregion."""
    out = x * radius + center
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


def _get_cube_center(bounds):
    """Get center of region defined by bounds."""
    center = (bounds.lower + bounds.upper) / 2
    return center


def _get_cube_radius(bounds):
    """Get radius of region defined by bounds."""
    radius = (bounds.upper - bounds.lower) / 2
    return radius


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
