from itertools import product

import numpy as np
import pytest
from estimagic.optimization.trust_region_sampling import _create_upscaled_lhs_sample
from estimagic.optimization.trust_region_sampling import _extend_upscaled_lhs_sample
from estimagic.optimization.trust_region_sampling import _get_empty_bin_info
from estimagic.optimization.trust_region_sampling import _scale_down_points
from estimagic.optimization.trust_region_sampling import _scale_up_points
from estimagic.optimization.trust_region_sampling import compute_optimality_criterion
from estimagic.optimization.trust_region_sampling import get_existing_points
from estimagic.optimization.trust_region_sampling import (
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
    sample fulfills some basic Latin-Hypercube condition.

    """
    sample = get_next_trust_region_points_latin_hypercube(
        center=center,
        radius=radius,
        n_points=n_points,
        n_iter=n_iter,
        lhs_design=lhs_design,
        optimality_criterion=optimality_criterion,
    )["points"]

    decimal = 8 if lhs_design == "centered" else 2

    assert sample.shape == (n_points, len(center))
    aaae(sample.mean(axis=0), center, decimal=decimal)


@pytest.mark.slow
@pytest.mark.parametrize(
    "optimality_criterion",
    ["a-optimal", "e-optimal", "d-optimal", "g-optimal", "maximin"],
)
def test_get_next_trust_region_points_latin_hypercube_optimality_criterion(
    optimality_criterion,
):
    """Check that the optimal sample is actually optimal."""
    sample = get_next_trust_region_points_latin_hypercube(
        center=np.ones(5),
        radius=0.1,
        n_points=10,
        n_iter=1,
        lhs_design="centered",
        optimality_criterion=optimality_criterion,
        target="linear",
    )

    optimized_sample = get_next_trust_region_points_latin_hypercube(
        center=np.ones(5),
        radius=0.1,
        n_points=10,
        n_iter=50_000,
        lhs_design="centered",
        optimality_criterion=optimality_criterion,
        target="linear",
    )
    crit_val_optimized_sample = compute_optimality_criterion(
        optimized_sample["points"], optimality_criterion, target="linear"
    )

    if optimality_criterion != "maximin":
        # by design of the criterion only few values are possible
        assert len(np.unique(optimized_sample["crit_vals"])) > 10_000
    assert sample["crit_vals"] > np.min(optimized_sample["crit_vals"])
    assert crit_val_optimized_sample == np.min(optimized_sample["crit_vals"])


def test_extend_upscaled_lhs_sample():
    """Test that existing points are correctly used.

    We want two draw 2 points with 1 existing point. Because of the center and
    radius there is only one possible point to draw in the upscaled space: (0, 0).

    """
    n_points = 2
    second_center = 0.4 * np.ones(2)
    second_radius = 0.1

    existing_points = np.array([[0.45, 0.45]])

    existing_scaled = _scale_up_points(
        existing_points, second_center, second_radius, n_points
    )
    empty_bins = _get_empty_bin_info(existing_scaled, n_points)
    new_points = _extend_upscaled_lhs_sample(
        empty_bins, n_points, n_designs=1, dtype=np.uint8
    )[0]

    assert np.all(new_points == np.array([[0, 0]]))


def test_get_empty_bin_info():
    """Test that correct empty bins are returned.

    In a 2-d Latin-Hypercube where the first two points are correctly placed over the
    first 2 dimensions, the only possible place for the 3rd point is at position (2, 2).

    """
    existing_upscaled_list = [np.array([[0, 0], [1, 1]]), np.array([[0, 1], [1, 0]])]

    empty_bins = []
    for existing_upscaled in existing_upscaled_list:
        empty_bins.append(_get_empty_bin_info(existing_upscaled, n_points=3))

    for empty_bin in empty_bins:
        assert np.all(empty_bin == np.array([[2, 2]]))


def test_get_empty_bin_info_multiple_points():
    """Test that correct number of empty bins is returned to create a Latin-Hypercube.

    This is due to the fact that two existing points fall into the same bin, and are
    therefore viewed as only one.

    """
    existing_upscaled = np.array(
        [
            [0.25, 0],
            [0.75, 0],
        ]
    )
    expected = np.array([[1, 1], [2, 2]])
    got = _get_empty_bin_info(existing_upscaled, n_points=3)

    assert np.all(expected == got)
