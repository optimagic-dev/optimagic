import numpy as np
import pytest
from tranquilo.optimization.tranquilo.volume import (
    _cube_radius,
    _cube_volume,
    _sphere_radius,
    _sphere_volume,
    get_radius,
    get_radius_after_volume_scaling,
    get_radius_of_cube_with_volume_of_sphere,
    get_radius_of_sphere_with_volume_of_cube,
    get_volume,
)

dims = dims = [1, 2, 3, 4, 12, 13, 15]
coeffs = [
    2,
    np.pi,
    4 * np.pi / 3,
    np.pi**2 / 2,
    np.pi**6 / 720,
    128 * np.pi**6 / 135135,
    256 * np.pi**7 / 2027025,
]


@pytest.mark.parametrize("dim", dims)
def test_get_radius_of_sphere_with_volume_of_cube(dim):
    cube_radius = 1.5
    scaling_factor = 0.95
    vol = _cube_volume(cube_radius, dim) * scaling_factor
    expected = _sphere_radius(vol, dim)
    got = get_radius_of_sphere_with_volume_of_cube(cube_radius, dim, scaling_factor)
    assert np.allclose(got, expected)


@pytest.mark.parametrize("dim", dims)
def test_get_radius_of_cube_with_volume_of_sphere(dim):
    sphere_radius = 1.5
    scaling_factor = 0.95
    vol = _sphere_volume(sphere_radius, dim) * scaling_factor
    expected = _cube_radius(vol, dim)
    got = get_radius_of_cube_with_volume_of_sphere(sphere_radius, dim, scaling_factor)
    assert np.allclose(got, expected)


def test_get_radius_of_sphere_with_volume_of_cube_no_scaling():
    v1 = get_radius_of_sphere_with_volume_of_cube(2.0, 2, None)
    v2 = get_radius_of_sphere_with_volume_of_cube(2.0, 2, 1.0)
    assert v1 == v2


def test_get_radius_of_cube_with_volume_of_sphere_no_scaling():
    v1 = get_radius_of_cube_with_volume_of_sphere(2.0, 2, None)
    v2 = get_radius_of_cube_with_volume_of_sphere(2.0, 2, 1.0)
    assert v1 == v2


@pytest.mark.parametrize("dim", dims)
def test_radius_after_volume_rescaling_scaling_factor_sphere(dim):
    radius = 0.6
    scaling_factor = 0.9

    naive = _sphere_radius(_sphere_volume(radius, dim) * scaling_factor, dim)

    got = get_radius_after_volume_scaling(radius, dim, scaling_factor)

    assert np.allclose(got, naive)


@pytest.mark.parametrize("dim", dims)
def test_radius_after_volume_rescaling_scaling_factor_cube(dim):
    radius = 0.6
    scaling_factor = 0.9

    naive = _cube_radius(_cube_volume(radius, dim) * scaling_factor, dim)

    got = get_radius_after_volume_scaling(radius, dim, scaling_factor)

    assert np.allclose(got, naive)


@pytest.mark.parametrize("dim, coeff", list(zip(dims, coeffs)))
def test_shpere_volume_and_radius(dim, coeff):
    radius = 0.5
    expected_volume = coeff * radius**dim
    got_volume = get_volume(radius, dim, "sphere")
    assert np.allclose(got_volume, expected_volume)

    got_radius = get_radius(got_volume, dim, "sphere")
    assert np.allclose(got_radius, radius)


@pytest.mark.parametrize("dim", dims)
def test_cube_volume_and_radius(dim):
    radius = 0.6

    expected_volume = 1.2**dim

    got_volume = get_volume(radius, dim, "cube")
    assert np.allclose(got_volume, expected_volume)

    got_radius = get_radius(got_volume, dim, "cube")
    assert np.allclose(got_radius, radius)
