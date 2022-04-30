import numpy as np
import pytest
from estimagic.optimization.tranquilo.lhs_sampling import _create_upscaled_sample
from estimagic.optimization.tranquilo.lhs_sampling import _extend_upscaled_sample
from estimagic.optimization.tranquilo.lhs_sampling import _get_empty_bin_info
from estimagic.optimization.tranquilo.lhs_sampling import _scale_down_points
from estimagic.optimization.tranquilo.lhs_sampling import _scale_up_points
from estimagic.optimization.tranquilo.lhs_sampling import calculate_criterion
from estimagic.optimization.tranquilo.lhs_sampling import lhs_sampler
from numpy.testing import assert_array_almost_equal as aaae


def test_scaling_bijection():
    params = {
        "n_params": 20,
        "n_points": 100,
        "n_designs": 1,
        "seed": 0,
        "dtype": np.uint8,
    }
    lower_bounds = np.ones(params["n_params"])
    upper_bounds = 3 + lower_bounds

    points = _create_upscaled_sample(**params)
    points = np.squeeze(points)
    downscaled = _scale_down_points(
        points, params["n_points"], lower_bounds, upper_bounds
    )
    upscaled = _scale_up_points(
        downscaled, params["n_points"], lower_bounds, upper_bounds
    )

    aaae(points, upscaled)


def test_scale_down_points():
    points = np.array([[0, 1], [1, 2], [2, 0]])
    n_points, n_params = points.shape

    # center [0.5, ..., 0.5] with radius 0.1
    lower_bounds = 0.4 * np.ones(n_params)
    upper_bounds = 0.6 * np.ones(n_params)

    downscaled = _scale_down_points(points, n_points, lower_bounds, upper_bounds)
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

    n_points, n_params = points.shape

    # center [0.5, ..., 0.5] with radius 0.5
    lower_bounds = np.zeros(n_params)
    upper_bounds = np.ones(n_params)

    upscaled = _scale_up_points(points, n_points, lower_bounds, upper_bounds)
    aaae(expected, upscaled)


@pytest.mark.parametrize("n_params", [2, 5, 10, 50, 100])
def test_latin_hypercube_property(n_params):
    """Check that for each single dimension the points are uniformly distributed."""
    n_points = 50
    sample = _create_upscaled_sample(
        n_params, n_points, n_designs=1, seed=0, dtype=np.uint8
    )
    index = np.arange(n_points)
    for j in range(n_params):
        aaae(index, np.sort(sample[0][:, j]))


@pytest.mark.parametrize(
    "criterion",
    ["a-optimal", "e-optimal", "d-optimal", "g-optimal", "maximin"],
)
def test_optimality_criterion(
    criterion,
):
    """Check that the optimal sample is actually optimal."""
    sample = lhs_sampler(
        lower_bounds=0.4 * np.ones(5),
        upper_bounds=0.6 * np.ones(5),
        target_size=10,
        centered=True,
        criterion=criterion,
        quadratic_target=False,
        n_iter=1,
        return_crit_vals=True,
    )

    optimized_sample = lhs_sampler(
        lower_bounds=0.4 * np.ones(5),
        upper_bounds=0.6 * np.ones(5),
        target_size=10,
        centered=True,
        criterion=criterion,
        quadratic_target=False,
        n_iter=50_000,  # random search through 50_000 examples
        return_crit_vals=True,
    )

    crit_val_optimized_sample = calculate_criterion(
        optimized_sample["points"], criterion, quadratic_target=False
    )

    if criterion != "maximin":
        # by design of the maximin criterion only few values are possible
        assert len(np.unique(optimized_sample["crit_vals"])) > 10_000
    assert sample["crit_vals"] > np.min(optimized_sample["crit_vals"])
    assert crit_val_optimized_sample == np.min(optimized_sample["crit_vals"])


def test_extend_upscaled_lhs_sample():
    """Test that existing points are correctly used.

    We want two draw 2 points with 1 existing point. Because of the center and
    radius there is only one possible point to draw in the upscaled space: (0, 0).

    """
    n_points = 2
    lower_bounds = 0.3 * np.ones(2)
    upper_bounds = 0.5 * np.ones(2)

    existing_points = np.array([[0.45, 0.45]])

    existing_scaled = _scale_up_points(
        existing_points, n_points, lower_bounds, upper_bounds
    )
    empty_bins = _get_empty_bin_info(existing_scaled, n_points, dtype=np.uint8)
    new_points = _extend_upscaled_sample(
        empty_bins, n_points, n_designs=1, seed=0, dtype=np.uint8
    )
    new_points = np.squeeze(new_points)

    assert np.all(new_points == np.array([[0, 0]], dtype=np.uint8))


def test_get_empty_bin_info():
    """Test that correct empty bins are returned.

    In a 2-d Latin-Hypercube where the first two points are correctly placed over the
    first 2 dimensions, the only possible place for the 3rd point is at position (2, 2).

    """
    existing_upscaled_list = [np.array([[0, 0], [1, 1]]), np.array([[0, 1], [1, 0]])]

    empty_bins = []
    for existing_upscaled in existing_upscaled_list:
        empty_bins.append(
            _get_empty_bin_info(existing_upscaled, n_points=3, dtype=np.uint8)
        )

    for empty_bin in empty_bins:
        assert np.all(empty_bin == np.array([[2, 2]], dtype=np.uint8))


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
    expected = np.array([[1, 1], [2, 2]], dtype=np.uint8)
    got = _get_empty_bin_info(existing_upscaled, n_points=3, dtype=np.uint8)

    assert np.all(expected == got)
