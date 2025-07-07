import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_array_almost_equal as aaae
from numpy.testing import assert_array_equal
from pandas.testing import assert_frame_equal
from pybaum import tree_map

import optimagic as om
from optimagic.optimization.history import (
    History,
    HistoryEntry,
    _apply_reduction_to_batches,
    _calculate_monotone_sequence,
    _get_batch_starts_and_stops,
    _get_flat_param_names,
    _get_flat_params,
    _is_1d_array,
    _task_to_categorical,
    _validate_args_are_all_none_or_lists_of_same_length,
)
from optimagic.typing import Direction, EvalTask

# ======================================================================================
# Test methods to add data to History (add_entry, add_batch, init)
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
    assert history.task == [EvalTask.FUN, EvalTask.FUN, EvalTask.FUN]
    assert history.batches == [0, 1, 2]
    aaae(history.fun, [1, 3, 2])
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
    assert history.task == [EvalTask.FUN, EvalTask.FUN, EvalTask.FUN]
    assert history.batches == [0, 0, 0]
    aaae(history.fun, [1, 3, 2])
    aaae(history.start_time, [0.1, 0.2, 0.3])
    aaae(history.stop_time, [0.2, 0.3, 0.4])

    assert_array_equal(history.monotone_fun, np.array([1, 3, 3], dtype=np.float64))
    assert_array_equal(
        history.flat_params, np.arange(1, 10, dtype=np.float64).reshape(3, 3)
    )


def test_history_from_data():
    data = {
        "params": [{"a": 1, "b": [2, 3]}, {"a": 4, "b": [5, 6]}, {"a": 7, "b": [8, 9]}],
        "fun": [1, 3, 2],
        "task": [EvalTask.FUN, EvalTask.FUN, EvalTask.FUN],
        "batches": [0, 0, 0],
        "start_time": [0.0, 0.15, 0.3],
        "stop_time": [0.1, 0.25, 0.4],
    }

    history = History(
        direction=Direction.MAXIMIZE,
        **data,
    )

    assert history.direction == Direction.MAXIMIZE

    assert history.params == data["params"]
    assert history.task == data["task"]
    assert history.batches == data["batches"]
    aaae(history.fun, data["fun"])
    aaae(history.start_time, data["start_time"])
    aaae(history.stop_time, data["stop_time"])

    assert_array_equal(history.monotone_fun, np.array([1, 3, 3], dtype=np.float64))
    assert_array_equal(
        history.flat_params, np.arange(1, 10, dtype=np.float64).reshape(3, 3)
    )


# ======================================================================================
# Test functionality of History
# ======================================================================================


@pytest.fixture
def params():
    params_tree = {"a": None, "b": {"c": None, "d": (None, None)}}
    return [
        tree_map(lambda _: k, params_tree, is_leaf=lambda leaf: leaf is None)  # noqa: B023
        for k in range(6)
    ]


@pytest.fixture
def history_data(params):
    return {
        "fun": [10, 9, None, None, 2, 5],
        "task": [
            EvalTask.FUN,
            EvalTask.FUN,
            EvalTask.JAC,
            EvalTask.JAC,
            EvalTask.FUN,
            EvalTask.FUN_AND_JAC,
        ],
        "start_time": [0, 2, 5, 7, 10, 12],
        "stop_time": [1, 4, 6, 9, 11, 14],
        "params": params,
        "batches": [0, 1, 2, 3, 4, 5],
    }


@pytest.fixture
def history(history_data):
    return History(direction=Direction.MINIMIZE, **history_data)


@pytest.fixture
def history_parallel(history_data):
    data = history_data.copy()
    data["batches"] = [0, 0, 1, 1, 2, 3]
    return History(direction=Direction.MINIMIZE, **data)


# Function data, function value, and monotone function value
# --------------------------------------------------------------------------------------


def test_history_fun_data_with_fun_evaluations_cost_model(history: History):
    got = history.fun_data(
        cost_model=om.timing.fun_evaluations,
        monotone=False,
    )
    exp = pd.DataFrame(
        {
            "fun": [10, 9, np.nan, np.nan, 2, 5],
            "time": [1, 2, 2, 2, 3, 4],
            "task": [
                "fun",
                "fun",
                "jac",
                "jac",
                "fun",
                "fun_and_jac",
            ],
        }
    )
    assert_frame_equal(got, exp, check_dtype=False, check_categorical=False)


def test_history_fun_data_with_fun_evaluations_cost_model_and_monotone(
    history: History,
):
    got = history.fun_data(
        cost_model=om.timing.fun_evaluations,
        monotone=True,
    )
    exp = pd.DataFrame(
        {
            "fun": [10, 9, np.nan, np.nan, 2, 2],
            "time": [1, 2, 2, 2, 3, 4],
            "task": [
                "fun",
                "fun",
                "jac",
                "jac",
                "fun",
                "fun_and_jac",
            ],
        }
    )
    assert_frame_equal(got, exp, check_dtype=False, check_categorical=False)


def test_history_fun_data_with_fun_batches_cost_model(history_parallel: History):
    got = history_parallel.fun_data(
        cost_model=om.timing.fun_batches,
        monotone=False,
    )
    exp = pd.DataFrame(
        {
            "fun": [9, np.nan, 2, 5],
            "time": [1.0, 1.0, 2.0, 3.0],
            "task": [
                "fun",
                "jac",
                "fun",
                "fun_and_jac",
            ],
        }
    )
    assert_frame_equal(got, exp, check_dtype=False, check_categorical=False)


def test_history_fun_data_with_evaluation_time_cost_model(history: History):
    got = history.fun_data(
        cost_model=om.timing.evaluation_time,
        monotone=False,
    )
    exp = pd.DataFrame(
        {
            "fun": [10, 9, np.nan, np.nan, 2, 5],
            "time": [1, 3, 4, 6, 7, 9],
            "task": [
                "fun",
                "fun",
                "jac",
                "jac",
                "fun",
                "fun_and_jac",
            ],
        }
    )
    assert_frame_equal(got, exp, check_dtype=False, check_categorical=False)


def test_fun_property(history: History):
    assert_array_equal(history.fun, [10, 9, None, None, 2, 5])


def test_monotone_fun_property(history: History):
    assert_array_equal(history.monotone_fun, np.array([10, 9, np.nan, np.nan, 2, 2]))


# Acceptance
# --------------------------------------------------------------------------------------


def test_is_accepted_property(history: History):
    got = history.is_accepted
    exp = np.array([True, True, False, False, True, False])
    assert_array_equal(got, exp)


# Parameter data, params, flat params, and flat params names
# --------------------------------------------------------------------------------------


def test_params_data_fun_evaluations_cost_model(history: History):
    got = history.params_data()
    exp = (
        pd.DataFrame(
            {
                "counter": np.tile(np.arange(6), reps=4),
                "name": np.repeat(
                    [
                        "a",
                        "b_c",
                        "b_d_0",
                        "b_d_1",
                    ],
                    6,
                ),
                "value": np.tile(list(range(6)), 4),
                "task": np.tile(
                    [
                        "fun",
                        "fun",
                        "jac",
                        "jac",
                        "fun",
                        "fun_and_jac",
                    ],
                    4,
                ),
                "fun": np.tile(
                    [
                        10,
                        9,
                        np.nan,
                        np.nan,
                        2,
                        5,
                    ],
                    4,
                ),
            }
        )
        .set_index(["counter", "name"])
        .sort_index()
    )
    assert_frame_equal(got, exp, check_categorical=False, check_dtype=False)


def test_params_data_fun_evaluations_cost_model_parallel(history_parallel: History):
    got = history_parallel.params_data()
    exp = (
        pd.DataFrame(
            {
                "counter": np.tile(np.arange(6), reps=4),
                "name": np.repeat(
                    [
                        "a",
                        "b_c",
                        "b_d_0",
                        "b_d_1",
                    ],
                    6,
                ),
                "value": np.tile(list(range(6)), 4),
                "task": np.tile(
                    [
                        "fun",
                        "fun",
                        "jac",
                        "jac",
                        "fun",
                        "fun_and_jac",
                    ],
                    4,
                ),
                "fun": np.tile(
                    [
                        10,
                        9,
                        np.nan,
                        np.nan,
                        2,
                        5,
                    ],
                    4,
                ),
            }
        )
        .set_index(["counter", "name"])
        .sort_index()
    )
    assert_frame_equal(got, exp, check_categorical=False, check_dtype=False)


def test_params_data_fun_evaluations_cost_model_parallel_collapse_batches(
    history_parallel: History,
):
    got = history_parallel.params_data(collapse_batches=True)
    exp = (
        pd.DataFrame(
            {
                "counter": np.tile([0, 1, 2, 3], reps=4),
                "name": np.repeat(
                    [
                        "a",
                        "b_c",
                        "b_d_0",
                        "b_d_1",
                    ],
                    4,
                ),
                "value": np.tile([1, 2, 4, 5], 4),
                "task": np.tile(["fun", "jac", "fun", "fun_and_jac"], 4),
                "fun": np.tile([9, np.nan, 2, 5], 4),
            }
        )
        .set_index(["counter", "name"])
        .sort_index()
    )
    assert_frame_equal(got, exp, check_categorical=False, check_dtype=False)


def test_params_property(history, params):
    assert history.params == params


def test_flat_params_property(history: History):
    got = history.flat_params
    assert_array_equal(got, [[k for _ in range(4)] for k in range(6)])


def test_flat_param_names(history: History):
    assert history.flat_param_names == ["a", "b_c", "b_d_0", "b_d_1"]


# Time
# --------------------------------------------------------------------------------------


def test_get_total_timings_per_task_fun(history: History):
    got = history._get_timings_per_task(EvalTask.FUN, cost_factor=1)
    exp = np.array([1, 1, 0, 0, 1, 0])
    assert_array_equal(got, exp)


def test_get_total_timings_per_task_jac_cost_factor_none(history: History):
    got = history._get_timings_per_task(EvalTask.JAC, cost_factor=None)
    exp = np.array([0, 0, 1, 2, 0, 0])
    assert_array_equal(got, exp)


def test_get_total_timings_per_task_fun_and_jac(history: History):
    got = history._get_timings_per_task(EvalTask.FUN_AND_JAC, cost_factor=-0.5)
    exp = np.array([0, 0, 0, 0, 0, -0.5])
    assert_array_equal(got, exp)


def test_get_total_timings_custom_cost_model(history: History):
    cost_model = om.timing.CostModel(
        fun=0.5, jac=1, fun_and_jac=2, label="test", aggregate_batch_time=sum
    )
    got = history._get_total_timings(cost_model)
    exp = np.array(
        [
            0.5,
            0.5,
            1,
            1,
            0.5,
            2,
        ]
    )
    assert_array_equal(got, exp)


def test_get_total_timings_fun_evaluations(history: History):
    got = history._get_total_timings(cost_model=om.timing.fun_evaluations)
    exp = np.array([1, 1, 0, 0, 1, 1])
    assert_array_equal(got, exp)


def test_get_total_timings_fun_batches(history: History):
    got = history._get_total_timings(cost_model=om.timing.fun_batches)
    exp = np.array([1, 1, 0, 0, 1, 1])
    assert_array_equal(got, exp)


def test_get_total_timings_fun_batches_parallel(history_parallel: History):
    got = history_parallel._get_total_timings(cost_model=om.timing.fun_batches)
    exp = np.array([1, 1, 0, 0, 1, 1])
    assert_array_equal(got, exp)


def test_get_total_timings_evaluation_time(history: History):
    got = history._get_total_timings(cost_model=om.timing.evaluation_time)
    exp = np.array([1, 2, 1, 2, 1, 2])
    assert_array_equal(got, exp)


def test_get_total_timings_wall_time(history: History):
    got = history._get_total_timings(cost_model="wall_time")
    exp = np.array([1, 4, 6, 9, 11, 14])
    assert_array_equal(got, exp)


def test_get_total_timings_invalid_cost_model(history: History):
    with pytest.raises(
        TypeError, match="cost_model must be a CostModel or 'wall_time'."
    ):
        history._get_total_timings(cost_model="invalid")


def test_start_time_property(history: History):
    assert history.start_time == [0, 2, 5, 7, 10, 12]


def test_stop_time_property(history: History):
    assert history.stop_time == [1, 4, 6, 9, 11, 14]


# Batches
# --------------------------------------------------------------------------------------


def test_batches_property(history: History):
    assert history.batches == [0, 1, 2, 3, 4, 5]


# Tasks
# --------------------------------------------------------------------------------------


def test_task_property(history: History):
    assert history.task == [
        EvalTask.FUN,
        EvalTask.FUN,
        EvalTask.JAC,
        EvalTask.JAC,
        EvalTask.FUN,
        EvalTask.FUN_AND_JAC,
    ]


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


def test_get_flat_param_names_pytree():
    got = _get_flat_param_names(param={"a": 0, "b": [0, 1], "c": np.arange(2)})
    exp = ["a", "b_0", "b_1", "c_0", "c_1"]
    assert got == exp


def test_get_flat_param_names_fast_path():
    got = _get_flat_param_names(param=np.arange(2))
    exp = ["0", "1"]
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
    got = _task_to_categorical(task)
    assert got.tolist() == ["fun", "jac", "fun_and_jac"]
    assert isinstance(got.dtype, pd.CategoricalDtype)


def test_get_batch_starts_and_stops():
    batches = [0, 0, 1, 1, 1, 2, 2, 3]
    got_starts, got_stops = _get_batch_starts_and_stops(batches)
    assert got_starts == [0, 2, 5, 7]
    assert got_stops == [2, 5, 7, 8]


def test_apply_to_batch_sum():
    data = np.array([0, 1, 2, 3, 4])
    batch_ids = [0, 0, 1, 1, 2]
    exp = np.array([1, 5, 4])
    got = _apply_reduction_to_batches(data, batch_ids, sum)
    assert_array_equal(exp, got)


def test_apply_to_batch_max():
    data = np.array([0, 1, 2, 3, 4])
    batch_ids = [0, 0, 1, 1, 2]
    exp = np.array([1, 3, 4])
    got = _apply_reduction_to_batches(data, batch_ids, max)
    assert_array_equal(exp, got)


def test_apply_to_batch_broken_func():
    data = np.array([0, 1, 2, 3, 4])
    batch_ids = [0, 0, 1, 1, 2]
    with pytest.raises(ValueError, match="Calling function <lambda> on batch [0, 0]"):
        _apply_reduction_to_batches(data, batch_ids, reduction_function=lambda _: 1 / 0)


def test_apply_to_batch_func_with_non_scalar_return():
    data = np.array([0, 1, 2, 3, 4])
    batch_ids = [0, 0, 1, 1, 2]
    with pytest.raises(ValueError, match="Function <lambda> did not return a scalar"):
        _apply_reduction_to_batches(
            data, batch_ids, reduction_function=lambda _list: _list
        )
