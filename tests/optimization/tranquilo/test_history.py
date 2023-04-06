"""Test the history class for least-squares optimizers."""
import numpy as np
import pytest
from tranquilo.optimization.tranquilo.history import History
from tranquilo.optimization.tranquilo.region import Region
from numpy.testing import assert_array_almost_equal as aaae


XS = [
    np.arange(3),
    np.arange(3).tolist(),
    np.arange(3).reshape(1, 3),
    np.arange(3).reshape(1, 3).tolist(),
]


@pytest.mark.parametrize("xs", XS)
def test_add_xs_not_initialized(xs):
    history = History(functype="least_squares")

    new_indices = history.add_xs(xs)

    if len(xs) == 1:
        aaae(new_indices, np.array([0]))
    else:
        assert new_indices == 0

    assert isinstance(history.xs, np.ndarray)
    aaae(history.xs[0], np.arange(3))

    assert history.index_mapper == {0: []}
    assert history.n_xs == 1
    assert history.n_fun == 0


@pytest.mark.parametrize("xs", XS)
def test_add_xs_initialized_with_space(xs):
    history = History(functype="least_squares")

    history.add_xs(np.ones((20, 3)))
    new_indices = history.add_xs(xs)

    if len(xs) == 1:
        aaae(new_indices, np.array([20]))
    else:
        assert new_indices == 20

    assert isinstance(history.xs, np.ndarray)
    aaae(history.xs[:21], np.vstack([np.ones((20, 3)), np.arange(3)]))

    assert history.index_mapper == {i: [] for i in range(21)}
    assert history.n_xs == 21
    assert history.n_fun == 0


@pytest.mark.parametrize("xs", XS)
def test_add_xs_initialized_extension_needed(xs):
    history = History(functype="least_squares")

    history.add_xs(np.ones(3))
    initial_size = len(history.xs)
    history.add_xs(np.ones((initial_size - 1, 3)))
    history.add_xs(xs)

    assert len(history.xs) > initial_size

    aaae(history.xs[initial_size], np.arange(3))

    assert history.n_xs == initial_size + 1
    assert history.n_fun == 0


EVALS = [
    (0, np.arange(5)),
    ([0], [np.arange(5)]),
    (np.array([0]), np.arange(5).reshape(1, 5)),
]


@pytest.mark.parametrize("x_indices, evals", EVALS)
def test_add_evals_not_initialized(x_indices, evals):
    history = History(functype="least_squares")
    history.add_xs(np.arange(3))

    history.add_evals(x_indices, evals)

    assert history.get_n_fun() == 1
    assert history.get_n_xs() == 1

    aaae(history.fvecs[0], np.arange(5))
    aaae(history.fvals[0], 30.0)

    assert history.index_mapper == {0: [0]}


@pytest.mark.parametrize("evals", [tup[1] for tup in EVALS])
def test_add_evals_initialized_with_space(evals):
    history = History(functype="least_squares")
    history.add_xs(np.arange(6).reshape(2, 3))
    history.add_evals([0] * 20, np.ones((20, 5)))

    history.add_evals(1, evals)

    assert history.get_n_fun() == 21
    assert history.get_n_xs() == 2

    aaae(history.fvecs[:21], np.vstack([np.ones((20, 5)), np.arange(5)]))
    aaae(history.fvals[20], 30.0)

    assert history.index_mapper == {0: list(range(20)), 1: [20]}


def test_get_indices_in_trustregion():
    history = History(functype="least_squares")
    xs = [[1, 1], [1.1, 1.2], [1.5, 1], [0.9, 0.9]]
    fvecs = np.zeros((4, 3))
    indices = history.add_xs(xs)
    history.add_evals(indices, fvecs)

    trustregion = Region(
        center=np.ones(2),
        radius=0.3,
    )

    indices = history.get_x_indices_in_region(trustregion)

    aaae(indices, np.array([0, 1, 3]))


@pytest.fixture()
def history():
    history = History(functype="least_squares")
    xs = np.arange(15).reshape(5, 3)
    fvecs = np.arange(25).reshape(5, 5)
    indices = history.add_xs(xs)
    history.add_evals(indices, fvecs)
    return history


def test_get_xs_no_indices(history):
    xs = history.get_xs()
    aaae(xs, np.arange(15).reshape(5, 3))


def test_get_xs_with_indices(history):
    xs = history.get_xs([0, 2, 4])
    aaae(xs, np.arange(15).reshape(5, 3)[[0, 2, 4]])


def test_get_xs_scalar_index(history):
    xs = history.get_xs(0)
    aaae(xs, np.arange(3))


def test_add_eval_for_invalid_x(history):
    with pytest.raises(ValueError):
        history.add_evals(5, np.arange(5))


def test_get_fvecs_scalar_index(history):
    fvecs = history.get_fvecs(0)
    aaae(fvecs, np.arange(5).reshape(1, 5))


def test_get_fvecs_with_indices(history):
    fvecs = history.get_fvecs([0])
    assert isinstance(fvecs, dict)
    assert len(fvecs) == 1
    assert 0 in fvecs
    aaae(fvecs[0], np.arange(5).reshape(1, 5))


def test_get_fvals_scalar_index(history):
    fvals = history.get_fvals(0)
    aaae(fvals, 30.0)


def test_get_fvals_with_indices(history):
    fvals = history.get_fvals([0])
    assert isinstance(fvals, dict)
    assert len(fvals) == 1
    assert 0 in fvals
    aaae(fvals[0], 30.0)


@pytest.mark.parametrize("average", [True, False])
def test_get_model_data_trivial_averaging(history, average):
    got_xs, got_fvecs = history.get_model_data(
        x_indices=[0, 1],
        average=average,
    )

    aaae(got_xs, np.arange(6).reshape(2, 3))
    aaae(got_fvecs, np.arange(10).reshape(2, 5))


def test_get_model_data_no_averaging(history):
    got_xs, got_fvecs = history.get_model_data(x_indices=[0, 1])
    aaae(got_xs, np.arange(6).reshape(2, 3))
    aaae(got_fvecs, np.arange(10).reshape(2, 5))


@pytest.fixture()
def noisy_history():
    history = History(functype="least_squares")
    history.add_xs(np.arange(6).reshape(2, 3))
    fvecs = np.arange(25).reshape(5, 5)
    history.add_evals([0, 0, 1, 1, 1], fvecs)
    return history


@pytest.mark.parametrize("average", [True, False])
def test_get_model_data_with_repeated_evaluations(noisy_history, average):
    got_xs, got_fvecs = noisy_history.get_model_data(
        x_indices=[0, 1],
        average=average,
    )

    if average:
        aaae(got_xs, np.arange(6).reshape(2, 3))
        expected_fvecs = np.array(
            [
                np.arange(10).reshape(2, 5).mean(axis=0),
                np.arange(10, 25).reshape(3, 5).mean(axis=0),
            ]
        )
        aaae(got_fvecs, expected_fvecs)
    else:
        aaae(got_xs, np.arange(6).reshape(2, 3).repeat([2, 3], axis=0))
        aaae(got_fvecs, np.arange(25).reshape(5, 5))
