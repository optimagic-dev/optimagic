import numpy as np
from estimagic.optimization.tranquilo.clustering import cluster
from numpy.testing import assert_array_equal as aae


def test_cluster_lollipop():
    rng = np.random.default_rng(123456)
    center = np.array([0.25, 0.25])
    radius = 0.05

    x = np.array(
        [
            center,
            *(center + rng.uniform(low=-radius, high=radius, size=(6, 2))).tolist(),
            [0.5, 0.5],
            [0.75, 0.75],
        ],
    )

    clusters, centers = cluster(x, epsilon=0.1)
    assert len(centers) == 3
    aae(np.unique(clusters), np.arange(3))


def test_cluster_grid():
    base_grid = np.linspace(-1, 1, 11)
    a, b = np.meshgrid(base_grid, base_grid)
    x = np.column_stack([a.flatten(), b.flatten()])

    clusters, centers = cluster(x, epsilon=0.1)

    assert len(centers) == len(x)
    aae(np.sort(clusters), np.arange(len(x)))
    aae(np.sort(centers), np.arange(len(x)))
