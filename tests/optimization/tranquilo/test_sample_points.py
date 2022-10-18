from collections import namedtuple

import numpy as np
import pytest
from estimagic.optimization.tranquilo.options import Bounds
from estimagic.optimization.tranquilo.options import TrustRegion
from estimagic.optimization.tranquilo.sample_points import get_sampler


@pytest.mark.parametrize(
    "sampler", ["naive", "box", "optimal_box", "sphere", "optimal_sphere"]
)
def test_integration_of_get_sampler_and_refercen_sampler(sampler):
    sampler = get_sampler(
        sampler=sampler,
        bounds=namedtuple("Bounds", ["lower", "upper"])(-np.ones(3), np.ones(3)),
    )

    calculated = sampler(
        trustregion=TrustRegion(center=0.5 * np.ones(3), radius=1),
        target_size=5,
        rng=np.random.default_rng(1234),
    )

    assert calculated.shape == (5, 3)
    assert (calculated <= 1).all()
    assert (calculated >= -1).all()


def test_bounds():
    bounds = Bounds(lower=-2 * np.ones(2), upper=np.array([0.25, 0.25]))
    trustregion = TrustRegion(center=np.zeros(2), radius=1)
    sampler = get_sampler("optimal_sphere", bounds)
    rng = np.random.default_rng()
    sample = sampler(trustregion, target_size=8, rng=rng)

    lower = np.full_like(sample, bounds.lower)
    upper = np.full_like(sample, bounds.upper)

    assert np.all(lower <= sample)
    assert np.all(sample <= upper)
