import pytest
from numpy.testing import assert_array_almost_equal as aaae

from optimagic.optimization.history import History, HistoryEntry
from optimagic.typing import Direction, EvalTask


@pytest.fixture
def history_entries():
    return [
        HistoryEntry(params=[1, 2, 3], fun=1, time=0.1, task=EvalTask.FUN),
        HistoryEntry(params=[4, 5, 6], fun=2, time=0.2, task=EvalTask.FUN),
        HistoryEntry(params=[7, 8, 9], fun=3, time=0.3, task=EvalTask.FUN),
    ]


def test_history_add_entry(history_entries):
    history = History(Direction.MINIMIZE)
    for entry in history_entries:
        history.add_entry(entry)

    assert history.params == [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    assert history.fun == [1, 2, 3]
    assert history.task == [EvalTask.FUN, EvalTask.FUN, EvalTask.FUN]
    assert history.batches == [0, 1, 2]
    assert history.direction == Direction.MINIMIZE
    aaae(history.time, [0.0, 0.1, 0.2])


def test_history_add_batch(history_entries):
    history = History(Direction.MAXIMIZE)
    history.add_batch(history_entries)

    assert history.params == [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    assert history.fun == [1, 2, 3]
    assert history.task == [EvalTask.FUN, EvalTask.FUN, EvalTask.FUN]
    assert history.batches == [0, 0, 0]
    assert history.direction == Direction.MAXIMIZE
    aaae(history.time, [0.0, 0.1, 0.2])
