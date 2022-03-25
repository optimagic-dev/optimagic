from collections import namedtuple

import numpy as np
from estimagic.optimization.tranquilo.sample_points import get_sampler


def test_integration_of_get_sampler_and_refercen_sampler():
    sampler = get_sampler(
        sampler="naive",
        bounds=namedtuple("Bounds", ["lower", "upper"])(-np.ones(3), np.ones(3)),
    )

    calculated, info = sampler(
        trustregion=namedtuple("TrustRegion", ["center", "radius"])(
            0.5 * np.ones(3), 1
        ),
        target_size=5,
    )

    assert calculated.shape == (5, 3)
    assert (calculated <= 1).all()
    assert (calculated >= -1).all()
