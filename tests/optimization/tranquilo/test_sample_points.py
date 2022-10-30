import numpy as np
import pytest
from estimagic.optimization.tranquilo.options import Bounds
from estimagic.optimization.tranquilo.options import TrustRegion
from estimagic.optimization.tranquilo.sample_points import _pairwise_distance_on_hull
from estimagic.optimization.tranquilo.sample_points import _project_onto_unit_hull
from estimagic.optimization.tranquilo.sample_points import get_sampler
from numpy.testing import assert_array_almost_equal as aaae
from scipy.spatial.distance import pdist


SAMPLERS = ["naive", "cube", "sphere", "optimal_cube", "optimal_sphere"]


@pytest.mark.parametrize("sampler", SAMPLERS)
def test_bounds_are_satisfied(sampler):
    bounds = Bounds(lower=-2 * np.ones(2), upper=np.array([0.25, 0.5]))
    sampler = get_sampler(sampler, bounds)
    sample = sampler(
        trustregion=TrustRegion(center=np.zeros(2), radius=1),
        target_size=5,
        rng=np.random.default_rng(1234),
    )

    lower = np.full_like(sample, bounds.lower)
    upper = np.full_like(sample, bounds.upper)

    assert np.all(lower <= sample)
    assert np.all(sample <= upper)


@pytest.mark.parametrize("sampler", SAMPLERS)
def test_enough_existing_points(sampler):
    # test that if enough existing points exist an empty array is returned
    sampler = get_sampler(
        sampler=sampler,
        bounds=Bounds(lower=-np.ones(3), upper=np.ones(3)),
    )
    calculated = sampler(
        trustregion=TrustRegion(center=np.zeros(3), radius=1),
        target_size=5,
        existing_xs=np.empty((5, 3)),
        rng=np.random.default_rng(1234),
    )

    assert calculated.size == 0


@pytest.mark.parametrize("sampler", ["optimal_cube", "optimal_sphere"])
def test_optimization_ignores_existing_points(sampler):
    # test that existing points behave as constants in the optimal sampling
    sampler = get_sampler(
        sampler=sampler,
        bounds=Bounds(lower=-np.ones(3), upper=np.ones(3)),
    )
    calculated = sampler(
        trustregion=TrustRegion(center=np.zeros(3), radius=1),
        target_size=5,
        existing_xs=np.ones((2, 3)),  # same point implies min distance of zero always
        rng=np.random.default_rng(1234),
    )

    assert pdist(calculated).min() > 0


@pytest.mark.parametrize("sampler", ["sphere", "cube"])
def test_optimality(sampler):
    # test that optimal versions of hull samplers produce better criterion value
    standard_sampler = get_sampler(
        sampler=sampler,
        bounds=Bounds(lower=-np.ones(3), upper=np.ones(3)),
    )
    optimal_sampler = get_sampler(
        sampler="optimal_" + sampler,
        bounds=Bounds(lower=-np.ones(3), upper=np.ones(3)),
    )

    distances = []
    for sampler in [standard_sampler, optimal_sampler]:
        sample = sampler(
            trustregion=TrustRegion(center=np.zeros(3), radius=1),
            target_size=5,
            rng=np.random.default_rng(1234),
        )
        distances.append(pdist(sample).min())

    assert distances[1] > distances[0]


@pytest.mark.parametrize("ord", [2, np.inf])
def test_pairwise_distance_on_hull(ord):  # noqa: A002

    # equal points imply zero distance
    value = _pairwise_distance_on_hull(x=np.ones((2, 2)), existing_xs=None, ord=ord)
    assert value == 0

    # non-equal points imply positive distance
    value = _pairwise_distance_on_hull(
        x=np.arange(4).reshape(2, 2), existing_xs=None, ord=ord
    )
    assert value > 0


@pytest.mark.parametrize("ord", [2, np.inf])
def test_project_onto_unit_hull(ord):  # noqa: A002

    rng = np.random.default_rng(1234)
    old = rng.uniform(-1, 1, size=10).reshape(5, 2)
    new = _project_onto_unit_hull(old, ord)

    norm = np.linalg.norm(old, axis=1, ord=ord)
    with pytest.raises(AssertionError):
        aaae(1, norm)

    norm = np.linalg.norm(new, axis=1, ord=ord)
    aaae(1, norm)
