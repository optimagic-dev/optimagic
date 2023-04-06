import numpy as np
from tranquilo.optimization.tranquilo.bounds import Bounds
from tranquilo.optimization.tranquilo.region import (
    Region,
    _any_bounds_binding,
    _get_shape,
    _get_cube_bounds,
    _get_cube_center,
    _get_effective_radius,
    _get_effective_center,
    _map_from_unit_cube,
    _map_from_unit_sphere,
    _map_to_unit_cube,
    _map_to_unit_sphere,
)
from numpy.testing import assert_array_equal
import pytest


def test_map_to_unit_sphere():
    got = _map_to_unit_sphere(np.ones(2), center=2 * np.ones(1), radius=2)
    assert_array_equal(got, -0.5 * np.ones(2))


def test_map_to_unit_cube():
    bounds = Bounds(lower=np.zeros(2), upper=2 * np.ones(2))
    got = _map_to_unit_cube(np.ones(2), cube_bounds=bounds)
    assert_array_equal(got, np.zeros(2))


def test_map_from_unit_sphere():
    got = _map_from_unit_sphere(-0.5 * np.ones(2), center=2 * np.ones(1), radius=2)
    assert_array_equal(got, np.ones(2))


def test_map_from_unit_cube():
    bounds = Bounds(lower=np.zeros(2), upper=2 * np.ones(2))
    got = _map_from_unit_cube(np.zeros(2), cube_bounds=bounds)
    assert_array_equal(got, np.ones(2))


def test_any_bounds_binding_true():
    bounds = Bounds(lower=-np.ones(2), upper=np.ones(2))
    out = _any_bounds_binding(bounds, center=np.zeros(2), radius=2)
    assert out


def test_any_bounds_binding_false():
    bounds = Bounds(lower=-np.ones(2), upper=np.ones(2))
    out = _any_bounds_binding(bounds, center=np.zeros(2), radius=0.5)
    assert not out


def test_get_shape_sphere():
    out = _get_shape(center=np.zeros(2), radius=1, bounds=None)
    assert out == "sphere"


def test_get_shape_cube():
    bounds = Bounds(lower=np.zeros(2), upper=np.ones(2))
    out = _get_shape(center=np.zeros(2), radius=1, bounds=bounds)
    assert out == "cube"


def test_get_cube_bounds():
    bounds = Bounds(lower=-np.ones(2), upper=np.ones(2))
    out = _get_cube_bounds(center=np.zeros(2), radius=1, bounds=bounds, shape="sphere")
    assert_array_equal(out.lower, bounds.lower)
    assert_array_equal(out.upper, bounds.upper)


def test_get_cube_bounds_no_bounds():
    bounds = Bounds(lower=None, upper=None)
    out = _get_cube_bounds(center=np.zeros(2), radius=1, bounds=bounds, shape="sphere")
    assert_array_equal(out.lower, -np.ones(2))
    assert_array_equal(out.upper, np.ones(2))


def test_get_cube_bounds_updated_upper_bounds():
    bounds = Bounds(lower=-2 * np.ones(2), upper=0.5 * np.ones(2))
    out = _get_cube_bounds(center=np.zeros(2), radius=1, bounds=bounds, shape="cube")
    np.all(out.lower > -np.ones(2))
    np.all(out.lower < np.zeros(2))
    np.all(out.upper == 0.5 * np.ones(2))


def test_get_cube_center():
    bounds = Bounds(lower=np.array([0, 0.5]), upper=np.array([1, 10]))
    out = _get_cube_center(cube_bounds=bounds)
    assert_array_equal(out, np.array([0.5, 5.25]))


def test_get_effective_radius():
    bounds = Bounds(lower=np.array([0, 0.5]), upper=np.array([1, 10]))
    out = _get_effective_radius(shape="cube", radius=None, cube_bounds=bounds)
    assert_array_equal(out, np.array([0.5, 4.75]))


def test_get_effective_center_sphere():
    out = _get_effective_center(shape="sphere", center=np.ones(2), cube_center=None)
    assert_array_equal(out, np.ones(2))


def test_get_effective_center_cube():
    out = _get_effective_center(shape="cube", center=None, cube_center=np.zeros(2))
    assert_array_equal(out, np.zeros(2))


def test_region_non_binding_bounds():
    region = Region(center=np.zeros(2), radius=1)
    assert region.shape == "sphere"
    assert region.radius == 1
    assert region.bounds is None
    with pytest.raises(AttributeError, match="The trustregion is a sphere"):
        region.cube_bounds  # noqa: B018
    with pytest.raises(AttributeError, match="The trustregion is a sphere"):
        region.cube_center  # noqa: B018


def test_region_binding_bounds():
    bounds = Bounds(lower=-np.ones(2), upper=0.5 * np.ones(2))
    region = Region(center=np.zeros(2), radius=1, bounds=bounds)
    assert region.shape == "cube"
    assert region.radius == 1
    assert region.bounds is bounds
    # shrinkage because cube radius is smaller than (spherical) radius
    assert np.all(bounds.lower - region.cube_bounds.lower < 0)
    assert_array_equal(region.cube_bounds.upper, bounds.upper)
