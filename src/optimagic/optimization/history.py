import warnings
from dataclasses import dataclass
from functools import partial
from typing import Any, Callable, Iterable, Literal

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from pybaum import leaf_names, tree_just_flatten

from optimagic.parameters.tree_registry import get_registry
from optimagic.timing import CostModel
from optimagic.typing import Direction, EvalTask, PyTree


@dataclass(frozen=True)
class HistoryEntry:
    params: PyTree
    fun: float | None
    start_time: float
    stop_time: float
    task: EvalTask


class History:
    # TODO: add counters for the relevant evaluations
    def __init__(
        self,
        direction: Direction,
        params: list[PyTree] | None = None,
        fun: list[float | None] | None = None,
        start_time: list[float] | None = None,
        stop_time: list[float] | None = None,
        batches: list[int] | None = None,
        task: list[EvalTask] | None = None,
    ) -> None:
        """Initialize a history.

        The history must know the direction of the optimization problem in order to
        correctly return monotone sequences. The history can be initialized empty, for
        example for usage during an optimization process, or with data, for example to
        recover a history from a log.

        """
        _validate_args_are_all_none_or_lists_of_same_length(
            params, fun, start_time, stop_time, batches, task
        )

        self.direction = direction
        self._params = params if params is not None else []
        self._fun = fun if fun is not None else []
        self._start_time = start_time if start_time is not None else []
        self._stop_time = stop_time if stop_time is not None else []
        self._batches = batches if batches is not None else []
        self._task = task if task is not None else []

    # ==================================================================================
    # Methods to add entries to the history
    # ==================================================================================

    def add_entry(self, entry: HistoryEntry, batch_id: int | None = None) -> None:
        if batch_id is None:
            batch_id = self._get_next_batch_id()
        self._params.append(entry.params)
        self._fun.append(entry.fun)
        self._start_time.append(entry.start_time)
        self._stop_time.append(entry.stop_time)
        self._batches.append(batch_id)
        self._task.append(entry.task)

    def add_batch(
        self, batch: list[HistoryEntry], batch_size: int | None = None
    ) -> None:
        # The naming is complicated here:
        # batch refers to the entries to be added to the history in one go
        # batch_size is a property of a parallelizing algorithm that influences how
        # the batch_ids are assigned. It is not the same as the length of the batch.
        if batch_size is None:
            batch_size = len(batch)

        start = self._get_next_batch_id()
        n_batches = int(np.ceil(len(batch) / batch_size))
        ids = np.repeat(np.arange(start, start + n_batches), batch_size)[: len(batch)]

        for entry, id in zip(batch, ids, strict=False):
            self.add_entry(entry, id)

    def _get_next_batch_id(self) -> int:
        if not self._batches:
            batch = 0
        else:
            batch = self._batches[-1] + 1
        return batch

    # ==================================================================================
    # Properties and methods to access the history
    # ==================================================================================

    # Function data, function value, and monotone function value
    # ----------------------------------------------------------------------------------

    def fun_data(self, cost_model: CostModel, monotone: bool = False) -> pd.DataFrame:
        """Return the function value data.

        Args:
            cost_model: The cost model that is used to calculate the time measure.
            monotone: Whether to return the monotone function values. Defaults to False.

        Returns:
            pd.DataFrame: The function value data. The columns are: 'fun', 'time' and
                'task'. If monotone is False, value is the fun value, otherwise the
                monotone function value. If dropna is True, rows with missing values
                are dropped.

        """
        if monotone:
            fun = self.monotone_fun
        else:
            fun = np.array(self.fun, dtype=np.float64)  # converts None to nan

        timings = self._get_total_timings(cost_model)
        task = _task_to_categorical(self.task)

        if not self._is_serial():
            # In the non-serial case, we take the batching into account and reduce
            # timings and fun to one value per batch.
            timings = _apply_reduction_to_batches(
                data=timings,
                batch_ids=self.batches,
                reduction_function=cost_model.aggregate_batch_time,
            )

            min_or_max = (
                np.nanmin if self.direction == Direction.MINIMIZE else np.nanmax
            )
            fun = _apply_reduction_to_batches(
                data=fun,
                batch_ids=self.batches,
                reduction_function=min_or_max,  # type: ignore[arg-type]
            )

            # Verify that tasks are homogeneous in each batch, and select first if true.
            tasks_and_batches = pd.DataFrame({"task": task, "batches": self.batches})
            grouped_tasks = tasks_and_batches.groupby("batches")["task"]
            if not grouped_tasks.nunique().eq(1).all():
                raise ValueError("Tasks are not homogeneous in each batch.")

            task = grouped_tasks.first().reset_index(drop=True)

        time = np.cumsum(timings)
        return pd.DataFrame({"fun": fun, "time": time, "task": task})

    @property
    def fun(self) -> list[float | None]:
        return self._fun

    @property
    def monotone_fun(self) -> NDArray[np.float64]:
        """The monotone function value of the history.

        If the value is None, the output at that position is nan.

        """
        return _calculate_monotone_sequence(self.fun, direction=self.direction)

    # Acceptance
    # ----------------------------------------------------------------------------------

    @property
    def is_accepted(self) -> NDArray[np.bool_]:
        """Boolean indicator whether a function value is accepted.

        A function value is accepted if it is smaller (or equal) than the monotone
        function value counterpart in the case of minimization, or larger (or equal) in
        the case of maximization. If the value is None, the output at that position is
        False.

        """
        fun_arr = np.array(self.fun, dtype=np.float64)
        if self.direction == Direction.MINIMIZE:
            return fun_arr <= self.monotone_fun
        elif self.direction == Direction.MAXIMIZE:
            return fun_arr >= self.monotone_fun

    # Parameter data, params, flat params, and flat params names
    # ----------------------------------------------------------------------------------

    def params_data(
        self, dropna: bool = False, collapse_batches: bool = False
    ) -> pd.DataFrame:
        """Return the parameter data.

        Args:
            dropna: Whether to drop rows with missing function values. These correspond
                to parameters that were used to calculate pure jacobians. Defaults to
                False.
            collapse_batches: Whether to collapse the batches and only keep the
                parameters that led to the minimal (or maximal) function value in each
                batch. Defaults to False.

        Returns:
            pd.DataFrame: The parameter data. The columns are: 'name' (the parameter
                names), 'value' (the parameter values), 'task' (the task for which the
                parameter was used), and 'counter' (a counter that is unique for each
                row).

        """
        wide = pd.DataFrame(self.flat_params, columns=self.flat_param_names)
        wide["task"] = _task_to_categorical(self.task)
        wide["fun"] = self.fun

        # If requested, we collapse the batches and only keep the parameters that led to
        # the minimal (or maximal) function value in each batch.
        if collapse_batches and not self._is_serial():
            wide["batches"] = self.batches

            # Verify that tasks are homogeneous in each batch
            if not wide.groupby("batches")["task"].nunique().eq(1).all():
                raise ValueError("Tasks are not homogeneous in each batch.")

            # We fill nans with inf or -inf to make sure that the idxmin/idxmax is
            # well-defined, since there is the possibility that all fun values are nans
            # in a batch.
            if self.direction == Direction.MINIMIZE:
                loc = (
                    wide.assign(fun_without_nan=wide["fun"].fillna(np.inf))
                    .groupby("batches")["fun_without_nan"]
                    .idxmin()
                )
            elif self.direction == Direction.MAXIMIZE:
                loc = (
                    wide.assign(fun_without_nan=wide["fun"].fillna(-np.inf))
                    .groupby("batches")["fun_without_nan"]
                    .idxmax()
                )

            wide = wide.loc[loc].drop(columns="batches")

        # We drop rows with missing values if requested. These correspond to parameters
        # that were used to calculate pure jacobians. This step must be done before
        # dropping the fun column and before setting the counter.
        if dropna:
            wide = wide.dropna(subset="fun")

        wide["counter"] = np.arange(len(wide))

        long = pd.melt(
            wide,
            var_name="name",
            value_name="value",
            id_vars=["task", "counter", "fun"],
        )

        data = long.reindex(columns=["counter", "name", "value", "task", "fun"])

        return data.set_index(["counter", "name"]).sort_index()

    @property
    def params(self) -> list[PyTree]:
        return self._params

    @property
    def flat_params(self) -> list[list[float]]:
        return _get_flat_params(self._params)

    @property
    def flat_param_names(self) -> list[str]:
        return _get_flat_param_names(param=self._params[0])

    # Time
    # ----------------------------------------------------------------------------------

    def _get_total_timings(
        self, cost_model: CostModel | Literal["wall_time"]
    ) -> NDArray[np.float64]:
        """Return the total timings across all tasks.

        Args:
            cost_model: The cost model that is used to calculate the time measure. If
                "wall_time", the wall time is returned.

        Returns:
            np.ndarray: The sum of the timings across all tasks.

        """
        if not isinstance(cost_model, CostModel) and cost_model != "wall_time":
            raise TypeError("cost_model must be a CostModel or 'wall_time'.")

        if cost_model == "wall_time":
            return np.array(self.stop_time, dtype=np.float64) - self.start_time[0]

        fun_time = self._get_timings_per_task(
            task=EvalTask.FUN, cost_factor=cost_model.fun
        )
        jac_time = self._get_timings_per_task(
            task=EvalTask.JAC, cost_factor=cost_model.jac
        )
        fun_and_jac_time = self._get_timings_per_task(
            task=EvalTask.FUN_AND_JAC, cost_factor=cost_model.fun_and_jac
        )

        return fun_time + jac_time + fun_and_jac_time

    def _get_timings_per_task(
        self, task: EvalTask, cost_factor: float | None
    ) -> NDArray[np.float64]:
        """Return the time measure per task.

        Args:
            task: The task for which the time is calculated.
            cost_factor: The cost factor used to calculate the time. If None, the time
                is the difference between the start and stop time, otherwise the time
                is given by the cost factor.

        Returns:
            np.ndarray: The time per task. For entries where the task is not the
                requested task, the time is 0.

        """
        task_mask = np.array([1 if t == task else 0 for t in self.task])
        factor: float | NDArray[np.float64]
        if cost_factor is None:
            factor = np.array(self.stop_time, dtype=np.float64) - np.array(
                self.start_time, dtype=np.float64
            )
        else:
            factor = cost_factor

        return factor * task_mask

    @property
    def start_time(self) -> list[float]:
        return self._start_time

    @property
    def stop_time(self) -> list[float]:
        return self._stop_time

    # Batches and fast_path
    # ----------------------------------------------------------------------------------

    @property
    def batches(self) -> list[int]:
        return self._batches

    def _is_serial(self) -> bool:
        return np.array_equal(self.batches, np.arange(len(self.batches)))

    # Tasks
    # ----------------------------------------------------------------------------------

    @property
    def task(self) -> list[EvalTask]:
        return self._task

    # ==================================================================================
    # Add deprecated dict access
    # ==================================================================================

    @property
    def time(self) -> list[float]:
        msg = (
            "The attribute `time` of History will be deprecated soon. Use the "
            "`start_time` method instead."
        )
        warnings.warn(msg, FutureWarning)
        arr = np.array(self._start_time)
        return (arr - arr[0]).tolist()

    @property
    def criterion(self) -> list[float | None]:
        msg = "The attribute `criterion` of History is deprecated. Use `fun` instead."
        warnings.warn(msg, FutureWarning)
        return self.fun

    @property
    def runtime(self) -> list[float]:
        msg = (
            "The attribute `runtime` of History will be deprecated soon. Use the "
            "`start_time` method instead."
        )
        warnings.warn(msg, FutureWarning)
        return self.time

    def __getitem__(self, key: str) -> Any:
        msg = "dict-like access to History is deprecated. Use attribute access instead."
        warnings.warn(msg, FutureWarning)
        return getattr(self, key)


# ======================================================================================
# Functions directly used in History methods
# ======================================================================================


def _get_flat_params(params: list[PyTree]) -> list[list[float]]:
    fast_path = len(params) > 0 and _is_1d_array(params[0])
    if fast_path:
        flatten = lambda x: x.tolist()
    else:
        registry = get_registry(extended=True)
        flatten = partial(tree_just_flatten, registry=registry)

    return [flatten(p) for p in params]


def _get_flat_param_names(param: PyTree) -> list[str]:
    fast_path = _is_1d_array(param)
    if fast_path:
        # Mypy raises an error here because .tolist() returns a str for zero-dimensional
        # arrays, but the fast path is only taken for 1d arrays, so it can be ignored.
        return np.arange(param.size).astype(str).tolist()

    registry = get_registry(extended=True)
    return leaf_names(param, registry=registry)


def _is_1d_array(param: PyTree) -> bool:
    return isinstance(param, np.ndarray) and param.ndim == 1


def _calculate_monotone_sequence(
    sequence: list[float | None], direction: Direction
) -> NDArray[np.float64]:
    sequence_arr = np.array(sequence, dtype=np.float64)  # converts None to nan
    nan_mask = np.isnan(sequence_arr)

    if direction == Direction.MINIMIZE:
        sequence_arr[nan_mask] = np.inf
        out = np.minimum.accumulate(sequence_arr)
    elif direction == Direction.MAXIMIZE:
        sequence_arr[nan_mask] = -np.inf
        out = np.maximum.accumulate(sequence_arr)

    out[nan_mask] = np.nan
    return out


# ======================================================================================
# Misc
# ======================================================================================


def _validate_args_are_all_none_or_lists_of_same_length(
    *args: list[Any] | None,
) -> None:
    all_none = all(arg is None for arg in args)
    all_list = all(isinstance(arg, list) for arg in args)

    if not all_none:
        if all_list:
            unique_list_lengths = set(map(len, args))  # type: ignore[arg-type]

            if len(unique_list_lengths) != 1:
                raise ValueError("All list arguments must have the same length.")

        else:
            raise ValueError("All arguments must be lists of the same length or None.")


def _task_to_categorical(task: list[EvalTask]) -> "pd.Series[str]":
    EvalTaskDtype = pd.CategoricalDtype(categories=[t.value for t in EvalTask])
    return pd.Series([t.value for t in task], dtype=EvalTaskDtype)


def _apply_reduction_to_batches(
    data: NDArray[np.float64],
    batch_ids: list[int],
    reduction_function: Callable[[Iterable[float]], float],
) -> NDArray[np.float64]:
    """Apply a reduction operator on batches of data.

    This function assumes that batch_ids are non-empty and sorted.

    Args:
        data: 1d array with data.
        batch_ids: A list with batch ids whose length is equal to the size of data.
            Values need to be sorted and can be repeated.
        reduction_function: A reduction function that takes an iterable of floats as
            input (e.g., a numpy.ndarray or list of floats) and returns a scalar. The
            function must be able to handle NaN's.

    Returns:
        The transformed data. Has one entry per unique batch id, equal to the result of
        applying the reduction function to the data of that batch.

    """
    batch_starts, batch_stops = _get_batch_starts_and_stops(batch_ids)

    batch_results: list[float] = []

    for start, stop in zip(batch_starts, batch_stops, strict=True):
        batch_data = data[start:stop]
        batch_id = batch_ids[start]

        try:
            if np.isnan(batch_data).all():
                reduced = np.nan
            else:
                reduced = reduction_function(batch_data)
        except Exception as e:
            msg = (
                f"Calling function {reduction_function.__name__} on batch {batch_id} "
                "of the History raised an Exception. Please verify that "
                f"{reduction_function.__name__} is well-defined, takes an iterable of "
                "floats as input and returns a scalar. The function must be able to "
                "handle NaN's."
            )
            raise ValueError(msg) from e

        if not np.isscalar(reduced):
            msg = (
                f"Function {reduction_function.__name__} did not return a scalar for "
                f"batch {batch_id}. Please verify that {reduction_function.__name__} "
                "returns a scalar when called on an iterable of floats. The function "
                "must be able to handle NaN's."
            )
            raise ValueError(msg)

        batch_results.append(reduced)  # type: ignore[arg-type]

    return np.array(batch_results, dtype=np.float64)


def _get_batch_starts_and_stops(batch_ids: list[int]) -> tuple[list[int], list[int]]:
    """Get start and stop indices of batches.

    This function assumes that batch_ids are non-empty and sorted.

    """
    ids_arr = np.array(batch_ids, dtype=np.int64)
    indices = np.where(ids_arr[:-1] != ids_arr[1:])[0] + 1
    list_indices: list[int] = indices.tolist()
    starts = [0, *list_indices]
    stops = [*starts[1:], len(batch_ids)]
    return starts, stops
