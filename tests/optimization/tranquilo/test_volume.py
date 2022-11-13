import numpy as np
import pytest
from estimagic.optimization.tranquilo.volume import _cube_radius
from estimagic.optimization.tranquilo.volume import _cube_volume
from estimagic.optimization.tranquilo.volume import _sphere_radius
from estimagic.optimization.tranquilo.volume import _sphere_volume
from estimagic.optimization.tranquilo.volume import get_radius
from estimagic.optimization.tranquilo.volume import get_radius_after_volume_scaling
from estimagic.optimization.tranquilo.volume import get_volume

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
