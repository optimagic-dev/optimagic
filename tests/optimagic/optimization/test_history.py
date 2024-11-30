import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal as aaae
from numpy.testing import assert_array_equal

from optimagic.optimization.history import History, HistoryEntry
from optimagic.typing import Direction, EvalTask


@pytest.fixture
def history_entries():
    return [
        HistoryEntry(params={"a": 1, "b": [2, 3]}, fun=1, time=0.1, task=EvalTask.FUN),
        HistoryEntry(params={"a": 4, "b": [5, 6]}, fun=3, time=0.2, task=EvalTask.FUN),
        HistoryEntry(params={"a": 7, "b": [8, 9]}, fun=2, time=0.3, task=EvalTask.FUN),
    ]


def test_history_add_entry(history_entries):
    history = History(Direction.MINIMIZE)
    for entry in history_entries:
        history.add_entry(entry)

    assert history.direction == Direction.MINIMIZE

    assert history.params == [
        {"a": 1, "b": [2, 3]},
        {"a": 4, "b": [5, 6]},
        {"a": 7, "b": [8, 9]},
    ]
    assert history.fun == [1, 3, 2]
    assert history.task == [EvalTask.FUN, EvalTask.FUN, EvalTask.FUN]
    assert history.batches == [0, 1, 2]
    aaae(history.time, [0.0, 0.1, 0.2])

    assert_array_equal(history.fun_array, np.array([1, 3, 2], dtype=np.float64))
    assert_array_equal(history.monotone_fun, np.array([1, 1, 1], dtype=np.float64))
    assert_array_equal(
        history.flat_params_array, np.arange(1, 10, dtype=np.float64).reshape(3, 3)
    )
    aaae(history.time_array, np.array([0.0, 0.1, 0.2]))


def test_history_add_batch(history_entries):
    history = History(Direction.MAXIMIZE)
    history.add_batch(history_entries)

    assert history.direction == Direction.MAXIMIZE

    assert history.params == [
        {"a": 1, "b": [2, 3]},
        {"a": 4, "b": [5, 6]},
        {"a": 7, "b": [8, 9]},
    ]
    assert history.fun == [1, 3, 2]
    assert history.task == [EvalTask.FUN, EvalTask.FUN, EvalTask.FUN]
    assert history.batches == [0, 0, 0]
    aaae(history.time, [0.0, 0.1, 0.2])

    assert_array_equal(history.fun_array, np.array([1, 3, 2], dtype=np.float64))
    assert_array_equal(history.monotone_fun, np.array([1, 3, 3], dtype=np.float64))
    assert_array_equal(
        history.flat_params_array, np.arange(1, 10, dtype=np.float64).reshape(3, 3)
    )
    aaae(history.time_array, np.array([0.0, 0.1, 0.2]))


def test_history_from_data():
    params = [{"a": 1, "b": [2, 3]}, {"a": 4, "b": [5, 6]}, {"a": 7, "b": [8, 9]}]
    fun = [1, 3, 2]
    task = [EvalTask.FUN, EvalTask.FUN, EvalTask.FUN]
    batches = [0, 0, 0]
    time = [0.0, 0.1, 0.2]

    history = History(
        direction=Direction.MAXIMIZE,
        fun=fun,
        params=params,
        task=task,
        batches=batches,
        start_time=time,
    )

    assert history.direction == Direction.MAXIMIZE

    assert history.params == params
    assert history.fun == fun
    assert history.task == task
    assert history.batches == batches
    aaae(history.time, time)

    assert_array_equal(history.fun_array, np.array(fun, dtype=np.float64))
    assert_array_equal(history.monotone_fun, np.array([1, 3, 3], dtype=np.float64))
    assert_array_equal(
        history.flat_params_array, np.arange(1, 10, dtype=np.float64).reshape(3, 3)
    )
    aaae(history.time_array, np.array(time))
