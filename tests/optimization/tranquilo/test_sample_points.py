import numpy as np
import pytest
from tranquilo.optimization.tranquilo.bounds import Bounds
from tranquilo.optimization.tranquilo.region import Region
from tranquilo.optimization.tranquilo.sample_points import (
    _draw_from_distribution,
    _minimal_pairwise_distance_on_hull,
    _project_onto_unit_hull,
    get_sampler,
)
from numpy.testing import assert_array_almost_equal as aaae
from scipy.spatial.distance import pdist

SAMPLERS = ["random_interior", "random_hull", "optimal_hull"]


@pytest.mark.parametrize("sampler", SAMPLERS)
def test_samplers(sampler):
    _sampler = get_sampler(sampler)
    trustregion = Region(center=np.array([0.0, 0]), radius=1.5, bounds=None)
    sample = _sampler(
        trustregion=trustregion,
        n_points=5,
        rng=np.random.default_rng(1234),
    )
    assert len(sample) == 5
    assert np.all(-1.5 <= sample)
    assert np.all(sample <= 1.5)


@pytest.mark.parametrize("sampler", SAMPLERS)
def test_bounds_are_satisfied(sampler):
    bounds = Bounds(lower=np.array([-2.0, -2.0]), upper=np.array([0.25, 0.5]))
    _sampler = get_sampler(sampler)
    trustregion = Region(center=np.array([0.0, 0]), radius=1.5, bounds=bounds)
    sample = _sampler(
        trustregion=trustregion,
        n_points=5,
        rng=np.random.default_rng(1234),
    )
    lower = np.full_like(sample, bounds.lower)
    upper = np.full_like(sample, bounds.upper)
    assert np.all(lower <= sample)
    assert np.all(sample <= upper)


@pytest.mark.parametrize("sampler", SAMPLERS)
def test_enough_existing_points(sampler):
    # test that if enough existing points exist an empty array is returned
    sampler = get_sampler(sampler=sampler)
    bounds = Bounds(lower=-np.ones(3), upper=np.ones(3))
    calculated = sampler(
        trustregion=Region(center=np.zeros(3), radius=1, bounds=bounds),
        n_points=0,
        existing_xs=np.empty((5, 3)),
        rng=np.random.default_rng(1234),
    )

    assert calculated.size == 0


def test_optimization_ignores_existing_points():
    # test that existing points behave as constants in the optimal sampling
    sampler = get_sampler(sampler="optimal_hull")
    bounds = Bounds(lower=-np.ones(3), upper=np.ones(3))
    calculated = sampler(
        trustregion=Region(center=np.zeros(3), radius=1, bounds=bounds),
        n_points=3,
        existing_xs=np.ones((2, 3)),  # same point implies min distance of zero always
        rng=np.random.default_rng(1234),
    )

    assert pdist(calculated).min() > 0


def test_optimality():
    # test that optimal versions of hull samplers produce better criterion value
    standard_sampler = get_sampler(sampler="random_hull")
    optimal_sampler = get_sampler(sampler="optimal_hull")
    bounds = Bounds(lower=-np.ones(3), upper=np.ones(3))
    distances = []
    for sampler in [standard_sampler, optimal_sampler]:
        sample = sampler(
            trustregion=Region(center=np.zeros(3), radius=1, bounds=bounds),
            n_points=5,
            rng=np.random.default_rng(1234),
        )
        distances.append(pdist(sample).min())

    assert distances[1] > distances[0]


@pytest.mark.parametrize("n_points_randomsearch", [1, 2, 5, 10])
def test_randomsearch(n_points_randomsearch):
    # test that initial randomsearch of hull samplers produce better fekete values

    bounds = Bounds(lower=-np.ones(3), upper=np.ones(3))

    _sampler = get_sampler("optimal_hull")

    # optimal sampling without randomsearch
    _, info = _sampler(
        trustregion=Region(center=np.zeros(3), radius=1, bounds=bounds),
        n_points=5,
        rng=np.random.default_rng(0),
        return_info=True,
    )

    # optimal sampling with randomsearch
    _, info_randomsearch = _sampler(
        trustregion=Region(center=np.zeros(3), radius=1, bounds=bounds),
        n_points=5,
        rng=np.random.default_rng(0),
        n_points_randomsearch=n_points_randomsearch,
        return_info=True,
    )

    for key in ["start_fekete", "opt_fekete"]:
        statement = info_randomsearch[key] >= info[key] or np.isclose(
            info_randomsearch[key], info[key], rtol=1e-3
        )
        assert statement


@pytest.mark.parametrize("trustregion_shape", ("sphere", "cube"))
def test_pairwise_distance_on_hull(trustregion_shape):
    # equal points imply zero distance
    value = _minimal_pairwise_distance_on_hull(
        x=np.ones(4),
        existing_xs=None,
        hardness=1,
        trustregion_shape=trustregion_shape,
        n_params=2,
    )
    assert value == 0

    # non-equal points imply positive distance
    value = _minimal_pairwise_distance_on_hull(
        x=np.arange(4),
        existing_xs=None,
        hardness=1,
        trustregion_shape=trustregion_shape,
        n_params=2,
    )
    assert value > 0


@pytest.mark.parametrize("trustregion_shape", ("sphere", "cube"))
def test_project_onto_unit_hull(trustregion_shape):
    rng = np.random.default_rng(1234)
    old = rng.uniform(-1, 1, size=10).reshape(5, 2)
    new = _project_onto_unit_hull(old, trustregion_shape=trustregion_shape)

    order = 2 if trustregion_shape == "sphere" else np.inf

    norm = np.linalg.norm(old, axis=1, ord=order)
    with pytest.raises(AssertionError):
        aaae(1, norm)

    norm = np.linalg.norm(new, axis=1, ord=order)
    aaae(1, norm)


@pytest.mark.parametrize("distribution", ["normal", "uniform"])
def test_draw_from_distribution(distribution):
    rng = np.random.default_rng()
    draw = _draw_from_distribution(distribution, rng=rng, size=(3, 2))
    if distribution == "uniform":
        assert (-1 <= draw).all()
        assert (draw <= 1).all()
    assert draw.shape == (3, 2)
