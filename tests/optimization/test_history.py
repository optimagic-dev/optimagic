import numpy as np
import pytest
from estimagic.optimization.history import LeastSquaresHistory
from numpy.testing import assert_array_almost_equal as aaae


ENTRIES = [
    (np.arange(3), [np.arange(5)]),
    ([np.arange(3)], list(range(5))),
    (np.arange(3).reshape(1, 3), np.arange(5).reshape(1, 5)),
]

TEST_CASES = []
for entries in ENTRIES:
    for is_center in True, False:
        TEST_CASES.append((entries, is_center))


@pytest.mark.parametrize("entries, is_center", TEST_CASES)
def test_add_entries_not_initialized(entries, is_center):
    history = LeastSquaresHistory()

    if is_center:
        c_info = {"x": np.zeros(3), "residuals": np.zeros(5), "radius": 1}
        history.add_centered_entries(*entries, c_info)
    else:
        history.add_entries(*entries)

    xs, residuals, critvals = history.get_entries()

    for entry in xs, residuals, critvals:
        assert isinstance(entry, np.ndarray)

    aaae(xs, np.arange(3).reshape(1, 3))
    aaae(residuals, np.arange(5).reshape(1, 5))
    aaae(critvals, np.array([30.0]))


@pytest.mark.parametrize("entries, is_center", TEST_CASES)
def test_add_entries_initialized_with_space(entries, is_center):
    history = LeastSquaresHistory()
    history.add_entries(np.ones((4, 3)), np.zeros((4, 5)))

    if is_center:
        c_info = {"x": np.zeros(3), "residuals": np.zeros(5), "radius": 1}
        history.add_centered_entries(*entries, c_info)
    else:
        history.add_entries(*entries)

    xs, residuals, critvals = history.get_entries(index=-1)

    for entry in xs, residuals:
        assert isinstance(entry, np.ndarray)

    aaae(xs, np.arange(3))
    aaae(residuals, np.arange(5))
    assert critvals == 30


def test_add_entries_initialized_extension_needed():
    history = LeastSquaresHistory()
    history.add_entries(np.ones((4, 3)), np.zeros((4, 5)))
    history.xs = history.xs[:5]
    history.residuals = history.residuals[:5]
    history.critvals = history.critvals[:5]

    history.add_entries(np.arange(12).reshape(4, 3), np.arange(20).reshape(4, 5))

    assert len(history.xs) == 10
    assert len(history.residuals) == 10
    assert len(history.critvals) == 10

    xs, residuals, _ = history.get_entries(index=-1)

    for entry in xs, residuals:
        assert isinstance(entry, np.ndarray)

    assert history.get_n_fun() == 8


@pytest.mark.xfail
def test_add_centered_entries():
    assert 1 == 2


@pytest.mark.xfail
def test_get_centered_entries():
    assert 1 == 2
