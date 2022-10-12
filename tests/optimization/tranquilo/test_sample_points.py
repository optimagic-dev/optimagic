from collections import namedtuple

import numpy as np
import pytest
from estimagic.optimization.tranquilo.options import TrustRegion
from estimagic.optimization.tranquilo.sample_points import get_sampler


@pytest.mark.parametrize("sampler", ["naive", "sphere", "optimal_sphere"])
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
