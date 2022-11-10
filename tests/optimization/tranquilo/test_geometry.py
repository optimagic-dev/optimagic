import numpy as np
import pytest
from estimagic.optimization.tranquilo.geometry import get_geometry_checker_pair
from estimagic.optimization.tranquilo.geometry import (
    maximize_absolute_value_trust_trsbox,
)
from estimagic.optimization.tranquilo.models import ScalarModel
from estimagic.optimization.tranquilo.options import TrustRegion
from estimagic.optimization.tranquilo.sample_points import get_sampler
from numpy.testing import assert_array_almost_equal


def aaae(x, y, case=None):
    tolerance = {
        None: 8,
        "hessian": 3,
    }
    assert_array_almost_equal(x, y, decimal=tolerance[case])


def test_geometry_checker():

    rng = np.random.default_rng()
    sampler = get_sampler("sphere", bounds=None)
    trustregion = TrustRegion(center=np.zeros(2), radius=1)

    x = sampler(trustregion, target_size=10, rng=rng)
    x_scaled = x * 0.5

    quality_calculator, cutoff_simulator = get_geometry_checker_pair(
        "d_optimality", reference_sampler="ball", n_params=2, bounds=None
    )

    x_quality = quality_calculator(x, trustregion, bounds=None)
    x_scaled_quality = quality_calculator(x_scaled, trustregion, bounds=None)

    cutoff = cutoff_simulator(n_samples=10, rng=rng, n_simulations=1_000)

    assert x_quality > x_scaled_quality
    assert x_quality > cutoff


def test_geometry_checker_scale_invariance():

    rng = np.random.default_rng()
    sampler = get_sampler("sphere", bounds=None)

    trustregion = TrustRegion(center=np.zeros(2), radius=1)
    trustregion_scaled = TrustRegion(center=np.ones(2), radius=2)

    x = sampler(trustregion, target_size=10, rng=rng)
    x_scaled = 1 + 2 * x

    quality_calculator, _ = get_geometry_checker_pair(
        "d_optimality", reference_sampler="ball", n_params=2, bounds=None
    )

    x_quality = quality_calculator(x, trustregion, bounds=None)
    x_scaled_quality = quality_calculator(x_scaled, trustregion_scaled, bounds=None)

    assert x_quality == x_scaled_quality


# =====================================================================================


TEST_CASES = [
    (
        ScalarModel(
            intercept=-1,
            linear_terms=np.array([1.0, -1.0]),
            square_terms=np.zeros((2, 2)),
        ),
        TrustRegion(center=np.zeros(2), radius=2),
        np.array([-2.0, -2.0]),
        np.array([1.0, 2.0]),
        np.array([-np.sqrt(2.0), np.sqrt(2.0)]),
    ),
    (
        ScalarModel(
            intercept=-1,
            linear_terms=np.array([1.0, -1.0]),
            square_terms=np.zeros((2, 2)),
        ),
        TrustRegion(center=np.zeros(2), radius=5),
        np.array([-2.0, -2.0]),
        np.array([1.0, 2.0]),
        np.array([-2.0, 2.0]),
    ),
    (
        ScalarModel(
            intercept=3,
            linear_terms=np.array([1.0, -1.0]),
            square_terms=np.zeros((2, 2)),
        ),
        TrustRegion(center=np.ones(2), radius=5),
        np.array([-2.0, -2.0]) + 1,
        np.array([1.0, 2.0]) + 1,
        np.array([1, -2]) + 1,
    ),
    (
        ScalarModel(
            intercept=-1,
            linear_terms=np.array([-1.0, -1.0]),
            square_terms=np.zeros((2, 2)),
        ),
        TrustRegion(center=np.zeros(2), radius=np.sqrt(2)),
        np.array([-2.0, -2.0]),
        np.array([0.1, 0.9]),
        np.array([0.1, 0.9]),
    ),
    (
        ScalarModel(
            intercept=-1,
            linear_terms=np.array([-1.0, -1.0, -1.0]),
            square_terms=np.zeros((3, 3)),
        ),
        TrustRegion(center=np.zeros(3), radius=np.sqrt(3)),
        np.array([-2.0, -2.0, -2.0]),
        np.array([0.9, 0.1, 5.0]),
        np.array([0.9, 0.1, np.sqrt(3.0 - 0.81 - 0.01)]),
    ),
    (
        ScalarModel(
            intercept=0,
            linear_terms=np.array([0, -1.0]),
            square_terms=np.zeros((2, 2)),
        ),
        TrustRegion(center=np.zeros(2), radius=5),
        np.array([-2, -2]),
        np.array([1, 2]),
        np.array([0, 2]),
    ),
    (
        ScalarModel(
            intercept=0,
            linear_terms=np.array([1e-15, -1.0]),
            square_terms=np.zeros((2, 2)),
        ),
        TrustRegion(center=np.zeros(2), radius=5),
        np.array([-2, -2]),
        np.array([1, 2]),
        np.array([0, 2]),
    ),
    (
        ScalarModel(
            intercept=0,
            linear_terms=np.array([1e-15, 0]),
            square_terms=np.zeros((2, 2)),
        ),
        TrustRegion(center=np.zeros(2), radius=5),
        np.array([-2, -2]),
        np.array([1, 2]),
        np.array([-2, 0]),
    ),
    (
        ScalarModel(
            intercept=-1,
            linear_terms=np.array([1, 0, 1]),
            square_terms=np.array(
                [[-2.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]]
            ),
        ),
        TrustRegion(center=np.zeros(3), radius=5 / 12),
        np.ones(3) - 1e20,
        np.ones(3) * 1e20,
        np.array([-1.0 / 3.0, 0.0, -0.25]),
    ),
    (
        ScalarModel(
            intercept=1,
            linear_terms=np.array([1, 0, 1]),
            square_terms=np.array(
                [[-2.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]]
            ),
        ),
        TrustRegion(center=np.zeros(3), radius=5 / 12),
        np.ones(3) - 1e20,
        np.ones(3) * 1e20,
        np.array([0.25, 0.0, 1.0 / 3.0]),
    ),
]


@pytest.mark.parametrize(
    "scalar_model, trustregion, lower_bounds, upper_bounds, x_expected", TEST_CASES
)
def test_improve_geometry_trsbox_quadratic(
    scalar_model, trustregion, lower_bounds, upper_bounds, x_expected
):
    case = None
    if not np.all(scalar_model.square_terms == 0):
        case = "hessian"

    x_out = maximize_absolute_value_trust_trsbox(
        scalar_model, trustregion, lower_bounds, upper_bounds
    )
    aaae(x_out, x_expected, case)
