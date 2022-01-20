from itertools import product

import numpy as np
import pytest
from estimagic.optimization.trust_region_management import _create_upscaled_lhs_sample
from estimagic.optimization.trust_region_management import _scale_down_points
from estimagic.optimization.trust_region_management import _scale_up_points
from estimagic.optimization.trust_region_management import compute_optimality_criterion
from estimagic.optimization.trust_region_management import get_existing_points
from estimagic.optimization.trust_region_management import (
    get_next_trust_region_points_latin_hypercube,
)
from numpy.testing import assert_array_almost_equal as aaae


def test_scaling_bijection():
    params = {
        "n_points": 100,
        "n_dim": 20,
        "n_designs": 1,
    }
    center = np.ones(params["n_dim"])
    radius = 0.1

    points = _create_upscaled_lhs_sample(**params)
    points = np.squeeze(points)
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
    sample = _create_upscaled_lhs_sample(n_dim, n_points, n_designs=1)
    index = np.arange(n_points)
    for j in range(n_dim):
        aaae(index, np.sort(sample[0][:, j]))


@pytest.mark.parametrize(
    "center, radius, n_points, optimality_criterion, lhs_design, target, n_iter",
    product(
        [np.ones(d) for d in (2, 5)],
        [0.05, 0.1],
        [10, 100],
        ["a-optimal", "e-optimal", "d-optimal", "g-optimal", "maximin"],
        ["centered", "random"],
        ["linear", "quadratic"],
        [1, 100],
    ),
)
def test_get_next_trust_region_points_latin_hypercube_single_use(
    center, radius, n_points, optimality_criterion, lhs_design, target, n_iter
):
    """Check that function can be called with all arguments and that the resulting
    sampe fulfills some basic Latin-Hypercube condition.

    """
    sample, _ = get_next_trust_region_points_latin_hypercube(
        center=center,
        radius=radius,
        n_points=n_points,
        n_iter=n_iter,
        lhs_design=lhs_design,
        optimality_criterion=optimality_criterion,
    )

    decimal = 8 if lhs_design == "centered" else 2

    assert sample.shape == (n_points, len(center))
    aaae(sample.mean(axis=0), center, decimal=decimal)


@pytest.mark.parametrize(
    "optimality_criterion",
    ["a-optimal", "e-optimal", "d-optimal", "g-optimal", "maximin"],
)
def test_get_next_trust_region_points_latin_hypercube_optimality_criterion(
    optimality_criterion,
):
    """Check that the optimal sample is actually optimal."""
    sample, crit_vals_single = get_next_trust_region_points_latin_hypercube(
        center=np.ones(5),
        radius=0.1,
        n_points=10,
        n_iter=1,
        lhs_design="centered",
        optimality_criterion=optimality_criterion,
        target="linear",
    )
    crit_val_sample = compute_optimality_criterion(
        sample, optimality_criterion, target="linear"
    )

    optimized_sample, crit_vals_many = get_next_trust_region_points_latin_hypercube(
        center=np.ones(5),
        radius=0.1,
        n_points=10,
        n_iter=50_000,
        lhs_design="centered",
        optimality_criterion=optimality_criterion,
        target="linear",
    )
    crit_val_optimized_sample = compute_optimality_criterion(
        optimized_sample, optimality_criterion, target="linear"
    )

    if optimality_criterion != "maximin":
        # by design of the criterion only few values are possible
        assert len(np.unique(crit_vals_many)) > 10_000
    assert crit_val_sample > crit_val_optimized_sample
    assert crit_vals_single > np.min(crit_vals_many)
    assert crit_val_optimized_sample == np.min(crit_vals_many)
