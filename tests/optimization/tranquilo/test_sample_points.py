import numpy as np
import pytest
from estimagic.optimization.tranquilo.options import Bounds, Region
from estimagic.optimization.tranquilo.sample_points import (
    _determinant_on_hull,
    _draw_from_distribution,
    _minimal_pairwise_distance_on_hull,
    _project_onto_unit_hull,
    get_sampler,
)
from numpy.testing import assert_array_almost_equal as aaae
from scipy.spatial.distance import pdist

SAMPLERS = ["box", "ball", "cube", "sphere", "optimal_cube", "optimal_sphere"]


@pytest.mark.parametrize("sampler", SAMPLERS)
def test_bounds_are_satisfied(sampler):
    bounds = Bounds(lower=-2 * np.ones(2), upper=np.array([0.25, 0.5]))
    sampler = get_sampler(sampler, bounds)
    sample = sampler(
        trustregion=Region(center=np.zeros(2), radius=1),
        n_points=5,
        rng=np.random.default_rng(1234),
    )
    lower = np.full_like(sample, bounds.lower)
    upper = np.full_like(sample, bounds.upper)
    assert np.all(lower <= sample)
    assert np.all(sample <= upper)


@pytest.mark.parametrize("order", [3, 10, 100])
def test_bounds_are_satisfied_general_hull_sampler(order):
    bounds = Bounds(lower=-2 * np.ones(2), upper=np.array([0.25, 0.5]))
    sampler = get_sampler("hull_sampler", bounds, user_options={"order": order})
    sample = sampler(
        trustregion=Region(center=np.zeros(2), radius=1),
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
    sampler = get_sampler(
        sampler=sampler,
        bounds=Bounds(lower=-np.ones(3), upper=np.ones(3)),
    )
    calculated = sampler(
        trustregion=Region(center=np.zeros(3), radius=1),
        n_points=0,
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
        model_info=None,
    )
    calculated = sampler(
        trustregion=Region(center=np.zeros(3), radius=1),
        n_points=3,
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
            trustregion=Region(center=np.zeros(3), radius=1),
            n_points=5,
            rng=np.random.default_rng(1234),
        )
        distances.append(pdist(sample).min())

    assert distances[1] > distances[0]


@pytest.mark.parametrize("sampler", ["sphere", "cube"])
@pytest.mark.filterwarnings("ignore::RuntimeWarning")
def test_multistart(sampler):
    # test that multistart versions of hull samplers produce better fekete value

    _sampler = get_sampler(
        sampler="optimal_" + sampler,
        bounds=Bounds(lower=-np.ones(3), upper=np.ones(3)),
    )

    # optimal sampling without multistart
    sample = _sampler(
        trustregion=Region(center=np.zeros(3), radius=1),
        n_points=5,
        rng=np.random.default_rng(1234),
    )

    # optimal sampling with multistart
    sample_multistart = _sampler(
        trustregion=Region(center=np.zeros(3), radius=1),
        n_points=5,
        rng=np.random.default_rng(1234),
        multistart=True,
        multistart_options={"n_samples": 2, "share_optimizations": 0.5},
    )

    criterion_kwargs = {
        "existing_xs": None,
        "order": 2 if sampler == "sphere" else np.inf,
        "n_params": 3,
    }

    distances = [
        _determinant_on_hull(_sample, **criterion_kwargs)
        for _sample in [sample, sample_multistart]
    ]

    assert distances[1] >= distances[0]


@pytest.mark.parametrize("order", [2, np.inf])
def test_pairwise_distance_on_hull(order):
    # equal points imply zero distance
    value = _minimal_pairwise_distance_on_hull(
        x=np.ones(4), existing_xs=None, hardness=1, order=order, n_params=2
    )
    assert value == 0

    # non-equal points imply positive distance
    value = _minimal_pairwise_distance_on_hull(
        x=np.arange(4), existing_xs=None, hardness=1, order=order, n_params=2
    )
    assert value > 0


@pytest.mark.parametrize("order", [2, np.inf])
def test_project_onto_unit_hull(order):
    rng = np.random.default_rng(1234)
    old = rng.uniform(-1, 1, size=10).reshape(5, 2)
    new = _project_onto_unit_hull(old, order)

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
