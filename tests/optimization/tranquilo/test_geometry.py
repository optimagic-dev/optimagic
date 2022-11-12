import numpy as np
from estimagic.optimization.tranquilo.geometry import get_geometry_checker_pair
from estimagic.optimization.tranquilo.options import TrustRegion
from estimagic.optimization.tranquilo.sample_points import get_sampler


def test_geometry_checker():

    rng = np.random.default_rng()
    sampler = get_sampler("sphere", bounds=None)
    trustregion = TrustRegion(center=np.zeros(2), radius=1)

    x = sampler(trustregion, n_points=10, rng=rng)
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

    x = sampler(trustregion, n_points=10, rng=rng)
    x_scaled = 1 + 2 * x

    quality_calculator, _ = get_geometry_checker_pair(
        "d_optimality", reference_sampler="ball", n_params=2, bounds=None
    )

    x_quality = quality_calculator(x, trustregion, bounds=None)
    x_scaled_quality = quality_calculator(x_scaled, trustregion_scaled, bounds=None)

    assert x_quality == x_scaled_quality
