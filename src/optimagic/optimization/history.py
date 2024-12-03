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
    # Properties to access the history
    # ==================================================================================

    # Function data, function value, and monotone function value
    # ----------------------------------------------------------------------------------

    def fun_data(self, cost_model: CostModel, monotone: bool) -> pd.DataFrame:
        """Return the function value data.

        Args:
            cost_model: The cost model that is used to calculate the time measure.
            monotone: Whether to return the monotone function values.

        Returns:
            pd.DataFrame: The function value data. The columns are: 'task', 'time' and
                'value'. If monotone is False, value is the fun value, otherwise the
                monotone function value.

        """
        if monotone:
            fun: list[float | None] | NDArray[np.float64] = self.monotone_fun
        else:
            fun = self.fun
        task = _task_as_categorical(self.task)
        time = self._get_time(cost_model)
        return pd.DataFrame({"fun": fun, "task": task, "time": time})

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

    def params_data(self, cost_model: CostModel) -> pd.DataFrame:
        """Return the parameter data.

        Args:
            cost_model: The cost model that is used to calculate the time measure.

        Returns:
            pd.DataFrame: The parameter data. The columns are: 'name' (the parameter
                names), 'value' (the parameter values), 'task', and 'time'.

        """
        wide = pd.DataFrame(self.flat_params, columns=self.flat_param_names)
        wide["task"] = _task_as_categorical(self.task)
        wide["time"] = self._get_time(cost_model)
        data = pd.melt(
            wide, var_name="name", value_name="value", id_vars=["task", "time"]
        )
        return data.reindex(columns=["name", "value", "task", "time"])

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

    def _get_time(
        self, cost_model: CostModel | Literal["wall_time"]
    ) -> NDArray[np.float64]:
        """Return the cumulative time measure.

        Args:
            cost_model: The cost model that is used to calculate the time measure. If
                "wall_time", the wall time is returned.

        Returns:
            np.ndarray: The time measure.

        """
        if not isinstance(cost_model, CostModel) and cost_model != "wall_time":
            raise ValueError("cost_model must be a CostModel or 'wall_time'.")

        if cost_model == "wall_time":
            return np.array(self.stop_time, dtype=np.float64) - self.start_time[0]

        fun_time = self._get_time_per_task(
            task=EvalTask.FUN, cost_factor=cost_model.fun
        )
        jac_time = self._get_time_per_task(
            task=EvalTask.JAC, cost_factor=cost_model.jac
        )
        fun_and_jac_time = self._get_time_per_task(
            task=EvalTask.FUN_AND_JAC, cost_factor=cost_model.fun_and_jac
        )

        time = fun_time + jac_time + fun_and_jac_time
        batch_time = _batch_apply(
            data=time,
            batch_ids=self.batches,
            func=cost_model.aggregate_batch_time,
        )
        return np.cumsum(batch_time)

    def _get_time_per_task(
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
        dummy_task = np.array([1 if t == task else 0 for t in self.task])
        if cost_factor is None:
            factor: float | NDArray[np.float64] = np.array(
                self.stop_time, dtype=np.float64
            ) - np.array(self.start_time, dtype=np.float64)
        else:
            factor = cost_factor

        return factor * dummy_task

    @property
    def start_time(self) -> list[float]:
        return self._start_time

    @property
    def stop_time(self) -> list[float]:
        return self._stop_time

    # Batches
    # ----------------------------------------------------------------------------------

    @property
    def batches(self) -> list[int]:
        return self._batches

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
# Methods
# ======================================================================================


def _get_flat_params(params: list[PyTree]) -> list[list[float]]:
    if len(params) > 0 and _is_1d_array(params[0]):
        # fast path
        flatten = lambda x: x.tolist()
    else:
        registry = get_registry(extended=True)
        flatten = partial(tree_just_flatten, registry=registry)

    return [flatten(p) for p in params]


def _get_flat_param_names(param: PyTree) -> list[str]:
    if _is_1d_array(param):
        # fast path
        return np.arange(param.size).astype(str).tolist()

    registry = get_registry(extended=True)
    return leaf_names(param, registry=registry)


def _is_1d_array(param: PyTree) -> bool:
    return isinstance(param, np.ndarray) and param.ndim == 1


def _calculate_monotone_sequence(
    sequence: list[float | None], direction: Direction
) -> NDArray[np.float64]:
    sequence_arr = np.array(sequence, dtype=np.float64)  # converts None to nan
    none_mask = np.isnan(sequence_arr)

    if direction == Direction.MINIMIZE:
        sequence_arr[none_mask] = np.inf
        out = np.minimum.accumulate(sequence_arr)
    elif direction == Direction.MAXIMIZE:
        sequence_arr[none_mask] = -np.inf
        out = np.maximum.accumulate(sequence_arr)

    out[none_mask] = np.nan
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


def _task_as_categorical(task: list[EvalTask]) -> pd.Categorical:
    return pd.Categorical(
        [t.value for t in task], categories=[t.value for t in EvalTask]
    )


def _batch_apply(
    data: NDArray[np.float64],
    batch_ids: list[int],
    func: Callable[[Iterable[float]], float],
) -> NDArray[np.float64]:
    """Apply a reduction operator on batches of data.

    Args:
        data: 1d array with data.
        batch_ids: A list whose length is equal to the size of data. Values need to be
            sorted and can be repeated.
        func: A reduction function that takes an iterable of floats as input (e.g., a
            numpy array or a list) and returns a scalar.

    Returns:
        The transformed data. Has the same length as data. For each batch, the result of
        the reduction operation is stored at the first index of that batch, and all
        other values of that batch are set to zero.

    """
    batch_starts = _get_batch_start(batch_ids)
    batch_stops = [*batch_starts, len(data)][1:]

    batch_results = []
    for batch, (start, stop) in zip(
        batch_ids, zip(batch_starts, batch_stops, strict=False), strict=False
    ):
        try:
            batch_data = data[start:stop]
            reduced = func(batch_data)
            batch_results.append(reduced)
        except Exception as e:
            msg = (
                f"Calling function {func.__name__} on batch {batch} of the History "
                f"History raised an Exception. Please verify that {func.__name__} is "
                "properly defined."
            )
            raise ValueError(msg) from e

    out = np.zeros_like(data)
    out[batch_starts] = batch_results
    return out


def _get_batch_start(batch_ids: list[int]) -> list[int]:
    """Get start indices of batch.

    This function assumes that batch_ids non-empty and sorted.

    """
    ids_arr = np.array(batch_ids, dtype=np.int64)
    indices = np.where(ids_arr[:-1] != ids_arr[1:])[0] + 1
    return np.insert(indices, 0, 0).tolist()
