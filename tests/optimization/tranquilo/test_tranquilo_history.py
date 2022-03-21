"""Test the history class for least-squares optimizers."""
import numpy as np
import pytest
from estimagic.optimization.tranquilo.tranquilo_history import History
from numpy.testing import assert_array_almost_equal as aaae


ENTRIES = [
    (np.arange(3), [np.arange(5)]),
    ([np.arange(3)], list(range(5))),
    (np.arange(3).reshape(1, 3), np.arange(5).reshape(1, 5)),
]

TEST_CASES = []
for entries in ENTRIES:
    # leave it in in case centered stuff comes back
    for is_center in [False]:
        TEST_CASES.append((entries, is_center))


@pytest.mark.parametrize("entries, is_center", TEST_CASES)
def test_add_entries_not_initialized(entries, is_center):
    history = History(functype="least_squares")

    if is_center:
        c_info = {"x": np.zeros(3), "fvecs": np.zeros(5), "radius": 1}
        history.add_centered_entries(*entries, c_info)
    else:
        history.add_entries(*entries)

    xs, fvecs, fvals = history.get_entries()
    xs_sinlge = history.get_xs()
    fvecs_sinlge = history.get_fvecs()
    fvals_sinlge = history.get_fvals()

    for entry in xs, fvecs, fvals:
        assert isinstance(entry, np.ndarray)

    aaae(xs, np.arange(3).reshape(1, 3))
    aaae(xs_sinlge, np.arange(3).reshape(1, 3))
    aaae(fvecs, np.arange(5).reshape(1, 5))
    aaae(fvecs_sinlge, np.arange(5).reshape(1, 5))
    aaae(fvals, np.array([30.0]))
    aaae(fvals_sinlge, np.array([30.0]))


@pytest.mark.parametrize("entries, is_center", TEST_CASES)
def test_add_entries_initialized_with_space(entries, is_center):
    history = History(functype="least_squares")
    history.add_entries(np.ones((4, 3)), np.zeros((4, 5)))

    if is_center:
        c_info = {"x": np.zeros(3), "fvecs": np.zeros(5), "radius": 1}
        history.add_centered_entries(*entries, c_info)
    else:
        history.add_entries(*entries)

    xs, fvecs, fvals = history.get_entries(index=-1)
    xs_sinlge = history.get_xs(index=-1)
    fvecs_sinlge = history.get_fvecs(index=-1)
    fvals_sinlge = history.get_fvals(index=-1)

    for entry in xs, fvecs:
        assert isinstance(entry, np.ndarray)

    aaae(xs, np.arange(3))
    aaae(xs_sinlge, np.arange(3))
    aaae(fvecs, np.arange(5))
    aaae(fvecs_sinlge, np.arange(5))
    assert fvals == 30
    assert fvals_sinlge == 30


def test_add_entries_initialized_extension_needed():
    history = History(functype="least_squares")
    history.add_entries(np.ones((4, 3)), np.zeros((4, 5)))
    history.xs = history.xs[:5]
    history.fvecs = history.fvecs[:5]
    history.fvals = history.fvals[:5]

    history.add_entries(np.arange(12).reshape(4, 3), np.arange(20).reshape(4, 5))

    assert len(history.xs) == 10
    assert len(history.fvecs) == 10
    assert len(history.fvals) == 10

    xs, fvecs, _ = history.get_entries(index=-1)
    xs_sinlge = history.get_xs(index=-1)
    fvecs_sinlge = history.get_fvecs(index=-1)

    for entry in xs, xs_sinlge, fvecs, fvecs_sinlge:
        assert isinstance(entry, np.ndarray)

    assert history.get_n_fun() == 8
