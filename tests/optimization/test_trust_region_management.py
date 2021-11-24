import numpy as np
from estimagic.optimization.trust_region_management import _create_upscaled_lhs_sample
from estimagic.optimization.trust_region_management import _scale_down_points
from estimagic.optimization.trust_region_management import _scale_up_points
from estimagic.optimization.trust_region_management import get_existing_points
from estimagic.optimization.trust_region_management import (
    get_next_trust_region_points_latin_hypercube,
)
from numpy.testing import assert_array_almost_equal as aaae


def get_upscaled_points(n_points, n_dim, n_designs):
    points = _create_upscaled_lhs_sample(n_dim, n_points, n_designs)
    if n_designs == 1:
        points = np.squeeze(points)
    return points


def test_scaling_bijection():
    params = {
        "n_points": 100,
        "n_dim": 20,
        "n_designs": 1,
    }
    center = np.ones(params["n_dim"])
    radius = 0.1

    points = get_upscaled_points(**params)
    downscaled = _scale_down_points(points, center, radius, params["n_points"])
    upscaled = _scale_up_points(downscaled, center, radius, params["n_points"])

    aaae(points, upscaled)


def test_scale_down_points():
    points = np.array([[0, 1], [1, 2], [2, 0]])
    n_points, n_dim = points.shape

    center = 0.5 * np.ones(n_dim)
    radius = 0.1

    downscaled = _scale_down_points(points, center, radius, n_points)
    expected = np.array(
        [[0.4, 0.46666667], [0.46666667, 0.53333333], [0.53333333, 0.4]]
    )

    aaae(expected, downscaled)


def test_scale_up_points():
    expected = np.array(
        [
            [1, 1],
            [2, 2],
            [3, 3],
        ]
    )
    points = expected / 3

    n_points, n_dim = points.shape

    center = 0.5 * np.ones(n_dim)
    radius = 0.5

    upscaled = _scale_up_points(points, center, radius, n_points)
    aaae(expected, upscaled)


def test_get_existing_points():
    old_sample = np.array(
        [
            [0.1, 0.1],
            [0.1, 0.2],
            [0.2, 0.2],
            [0.2, 0.3],
        ]
    )
    new_center = 0.3 * np.ones(2)
    new_radius = 0.1

    expected = np.array(
        [
            [0.2, 0.2],
            [0.2, 0.3],
        ]
    )

    got = get_existing_points(old_sample, new_center, new_radius)
    aaae(expected, got)


def test_get_existing_points_high_dim():
    """Check that the curse-of-dimensionality applies to our function."""
    n_dim = 100
    n_points = 25
    old_center = np.ones(n_dim)
    old_radius = 0.1
    old_sample, _ = get_next_trust_region_points_latin_hypercube(
        old_center, old_radius, n_points, n_iter=10
    )

    got = get_existing_points(old_sample, old_center * 1.05, old_radius)
    assert got is None


def test_latin_hypercube_property():
    """Check that for each single dimension the points are uniformly distributed."""
    n_dim, n_points = np.random.randint(2, 100, size=2)
    sample = _create_upscaled_lhs_sample(n_dim, n_points)

    index = np.arange(n_points)

    for j in range(n_dim):
        aaae(index, np.sort(sample[:, j]))


def test_get_empty_bin_info():
    pass


def test_extend_upscaled_lhs_sample():
    pass


def test_get_next_trust_region_points_latin_hypercube_single_use():
    pass


def test_get_next_trust_region_points_latin_hypercube_iteration():
    pass
