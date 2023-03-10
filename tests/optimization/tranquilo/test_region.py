import numpy as np
from estimagic.optimization.tranquilo.bounds import Bounds
from estimagic.optimization.tranquilo.region import (
    Region,
    _any_bounds_binding,
    _get_effective_bounds,
)
from numpy.testing import assert_array_equal


def test_any_bounds_binding_true():
    bounds = Bounds(lower=-np.ones(2), upper=np.ones(2))
    out = _any_bounds_binding(bounds, center=np.zeros(2), sphere_radius=2)
    assert out


def test_any_bounds_binding_false():
    bounds = Bounds(lower=-np.ones(2), upper=np.ones(2))
    out = _any_bounds_binding(bounds, center=np.zeros(2), sphere_radius=0.5)
    assert not out


def test_get_effective_bounds():
    bounds = Bounds(lower=-np.ones(2), upper=np.ones(2))
    out = _get_effective_bounds(center=np.zeros(2), radius=1, bounds=bounds)
    assert_array_equal(out.lower, bounds.lower)
    assert_array_equal(out.upper, bounds.upper)


def test_get_effective_bounds_no_bounds():
    bounds = Bounds(lower=None, upper=None)
    out = _get_effective_bounds(center=np.zeros(2), radius=1, bounds=bounds)
    assert_array_equal(out.lower, -np.ones(2))
    assert_array_equal(out.upper, np.ones(2))


def test_get_effective_bounds_updated_upper_bounds():
    bounds = Bounds(lower=-2 * np.ones(2), upper=0.5 * np.ones(2))
    out = _get_effective_bounds(center=np.zeros(2), radius=1, bounds=bounds)
    assert_array_equal(out.lower, -np.ones(2))
    assert_array_equal(out.upper, 0.5 * np.ones(2))


def test_region_non_binding_bounds():
    region = Region(center=np.zeros(2), sphere_radius=1)
    assert region.shape == "sphere"
    assert region.radius == 1
    assert region.cube_radius < 1
    assert region.bounds is None
    assert_array_equal(region.effective_bounds.lower, -np.ones(2))
    assert_array_equal(region.effective_bounds.upper, np.ones(2))


def test_region_binding_bounds():
    bounds = Bounds(lower=-np.ones(2), upper=0.5 * np.ones(2))
    region = Region(center=np.zeros(2), sphere_radius=1, bounds=bounds)
    assert region.shape == "cube"
    assert region.radius < 1
    assert region.cube_radius == region.radius
    assert region.bounds is bounds
    # shrinkage because of effective bounds because cube radius is smaller
    assert np.all(bounds.lower - region.effective_bounds.lower < 0)
    assert_array_equal(region.effective_bounds.upper, bounds.upper)
