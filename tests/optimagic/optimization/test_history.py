import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_array_almost_equal as aaae
from numpy.testing import assert_array_equal

from optimagic.optimization.history import (
    History,
    HistoryEntry,
    _calculate_monotone_sequence,
    _get_flat_param_names,
    _get_flat_params,
    _is_1d_array,
    _task_as_categorical,
    _validate_args_are_all_none_or_lists_of_same_length,
)
from optimagic.typing import Direction, EvalTask

# ======================================================================================
# Test histories add entries and batches methods
# ======================================================================================


@pytest.fixture
def history_entries():
    return [
        HistoryEntry(
            params={"a": 1, "b": [2, 3]},
            fun=1,
            start_time=0.1,
            stop_time=0.2,
            task=EvalTask.FUN,
        ),
        HistoryEntry(
            params={"a": 4, "b": [5, 6]},
            fun=3,
            start_time=0.2,
            stop_time=0.3,
            task=EvalTask.FUN,
        ),
        HistoryEntry(
            params={"a": 7, "b": [8, 9]},
            fun=2,
            start_time=0.3,
            stop_time=0.4,
            task=EvalTask.FUN,
        ),
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
    aaae(history.start_time, [0.1, 0.2, 0.3])
    aaae(history.stop_time, [0.2, 0.3, 0.4])

    assert_array_equal(history.monotone_fun, np.array([1, 1, 1], dtype=np.float64))
    assert_array_equal(
        history.flat_params, np.arange(1, 10, dtype=np.float64).reshape(3, 3)
    )


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
    aaae(history.start_time, [0.1, 0.2, 0.3])
    aaae(history.stop_time, [0.2, 0.3, 0.4])

    assert_array_equal(history.monotone_fun, np.array([1, 3, 3], dtype=np.float64))
    assert_array_equal(
        history.flat_params, np.arange(1, 10, dtype=np.float64).reshape(3, 3)
    )


# ======================================================================================
# Test history from data method
# ======================================================================================


@pytest.fixture
def history_data():
    return {
        "params": [{"a": 1, "b": [2, 3]}, {"a": 4, "b": [5, 6]}, {"a": 7, "b": [8, 9]}],
        "fun": [1, 3, 2],
        "task": [EvalTask.FUN, EvalTask.FUN, EvalTask.FUN],
        "batches": [0, 0, 0],
        "start_time": [0.0, 0.15, 0.3],
        "stop_time": [0.1, 0.25, 0.4],
    }


def test_history_from_data(history_data):
    history = History(
        direction=Direction.MAXIMIZE,
        **history_data,
    )

    assert history.direction == Direction.MAXIMIZE

    assert history.params == history_data["params"]
    assert history.fun == history_data["fun"]
    assert history.task == history_data["task"]
    assert history.batches == history_data["batches"]
    aaae(history.start_time, history_data["start_time"])
    aaae(history.stop_time, history_data["stop_time"])

    assert_array_equal(history.monotone_fun, np.array([1, 3, 3], dtype=np.float64))
    assert_array_equal(
        history.flat_params, np.arange(1, 10, dtype=np.float64).reshape(3, 3)
    )


# ======================================================================================
# Unit tests
# ======================================================================================


def test_is_1d_array():
    assert _is_1d_array(np.arange(2)) is True
    assert _is_1d_array(np.eye(2)) is False
    assert _is_1d_array([0, 1]) is False


def test_get_flat_params_pytree():
    params = [
        {"a": 1, "b": [0, 1], "c": np.arange(2)},
        {"a": 2, "b": [1, 2], "c": np.arange(2)},
    ]
    got = _get_flat_params(params)
    exp = [
        [1, 0, 1, 0, 1],
        [2, 1, 2, 0, 1],
    ]
    assert_array_equal(got, exp)


def test_get_flat_params_fast_path():
    params = [np.arange(2)]
    got = _get_flat_params(params)
    exp = [[0, 1]]
    assert_array_equal(got, exp)


def test_get_flat_param_names():
    got = _get_flat_param_names(param={"a": 0, "b": [0, 1], "c": np.arange(2)})
    exp = ["a", "b_0", "b_1", "c_0", "c_1"]
    assert got == exp


def test_calculate_monotone_sequence_maximize():
    sequence = [0, 1, 0, 0, 2, 10, 0]
    exp = [0, 1, 1, 1, 2, 10, 10]
    got = _calculate_monotone_sequence(sequence, direction=Direction.MAXIMIZE)
    assert_array_equal(exp, got)


def test_calculate_monotone_sequence_minimize():
    sequence = [10, 11, 8, 12, 0, 5]
    exp = [10, 10, 8, 8, 0, 0]
    got = _calculate_monotone_sequence(sequence, direction=Direction.MINIMIZE)
    assert_array_equal(exp, got)


def test_validate_args_are_all_none_or_lists_of_same_length():
    _validate_args_are_all_none_or_lists_of_same_length(None, None)
    _validate_args_are_all_none_or_lists_of_same_length([1], [1])

    with pytest.raises(ValueError, match="All list arguments must have the same"):
        _validate_args_are_all_none_or_lists_of_same_length([1], [1, 2])

    with pytest.raises(ValueError, match="All arguments must be lists of the same"):
        _validate_args_are_all_none_or_lists_of_same_length(None, [1])


def test_task_as_categorical():
    task = [EvalTask.FUN, EvalTask.JAC, EvalTask.FUN_AND_JAC]
    got = _task_as_categorical(task)
    assert got.tolist() == ["fun", "jac", "fun_and_jac"]
    assert isinstance(got.dtype, pd.CategoricalDtype)


@pytest.fixture
def history():
    data = {
        "fun": [10, None, 9, None, 5],
        "task": [
            EvalTask.FUN,
            EvalTask.JAC,
            EvalTask.FUN,
            EvalTask.JAC,
            EvalTask.FUN,
        ],
        "start_time": [0, 2, 5, 7, 10],
        "stop_time": [1, 4, 6, 9, 11],
        "params": [3, 3, 2, 2, 1],
        "batches": [0, 1, 2, 3, 4],
    }

    return History(direction=Direction.MINIMIZE, **data)
