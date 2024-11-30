import warnings
from dataclasses import dataclass
from functools import partial
from typing import Any

import numpy as np
from numpy.typing import NDArray
from pybaum import get_registry, tree_just_flatten

from optimagic.typing import Direction, EvalTask, PyTree


@dataclass(frozen=True)
class HistoryEntry:
    params: PyTree
    fun: float | None
    time: float
    task: EvalTask


class History:
    # TODO: add counters for the relevant evaluations
    def __init__(
        self,
        direction: Direction,
        params: list[PyTree] | None = None,
        fun: list[float | None] | None = None,
        time: list[float] | None = None,
        batches: list[int] | None = None,
        task: list[EvalTask] | None = None,
    ) -> None:
        """Initialize a history.

        The history must know the direction of the optimization problem in order to
        correctly return monotone sequences. The history can be initialized empty, for
        example for usage during an optimization process, or with data, for example to
        recover a history from a log.

        """
        if direction not in [Direction.MINIMIZE, Direction.MAXIMIZE]:
            raise ValueError(f"Invalid direction: {direction}.")
        self.direction = direction
        self._params = params if params is not None else []
        self._fun = fun if fun is not None else []
        self._time = time if time is not None else []
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
        self._time.append(entry.time)
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

    # Function value and monotone function value
    # ----------------------------------------------------------------------------------

    @property
    def fun(self) -> list[float | None]:
        return self._fun

    @property
    def fun_array(self) -> NDArray[np.float64]:
        return np.array(self._valid_fun, dtype=np.float64)

    @property
    def _valid_fun(self) -> list[float]:
        return [f for f in self.fun if f is not None]

    @property
    def monotone_fun(self) -> NDArray[np.float64]:
        if self.direction == Direction.MINIMIZE:
            return np.minimum.accumulate(self.fun_array)
        elif self.direction == Direction.MAXIMIZE:
            return np.maximum.accumulate(self.fun_array)

    # Acceptance
    # ----------------------------------------------------------------------------------

    @property
    def is_accepted(self) -> NDArray[np.bool_]:
        if self.direction == Direction.MINIMIZE:
            return self.fun_array <= self.monotone_fun
        elif self.direction == Direction.MAXIMIZE:
            return self.fun_array >= self.monotone_fun

    # Parameters
    # ----------------------------------------------------------------------------------

    @property
    def params(self) -> list[PyTree]:
        return self._params

    @property
    def flat_params_array(self) -> NDArray[np.float64]:
        return np.array(_flatten_params(self._valid_params), dtype=np.float64)

    @property
    def _valid_params(self) -> list[PyTree]:
        return [p for p, f in zip(self.params, self.fun, strict=True) if f is not None]

    # Time
    # ----------------------------------------------------------------------------------

    @property
    def time(self) -> list[float]:
        arr = np.array(self._time)
        return (arr - arr[0]).tolist()

    @property
    def time_array(self) -> NDArray[np.float64]:
        return np.array(self._valid_time, dtype=np.float64)

    @property
    def _valid_time(self) -> list[float]:
        return [t for t, f in zip(self.time, self.fun, strict=True) if f is not None]

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
    def criterion(self) -> list[float | None]:
        msg = "The attribute `criterion` of History is deprecated. Use `fun` instead."
        warnings.warn(msg, FutureWarning)
        return self.fun

    @property
    def runtime(self) -> list[float]:
        msg = "The attribute `runtime` of History is deprecated. Use `time` instead."
        warnings.warn(msg, FutureWarning)
        return self.time

    def __getitem__(self, key: str) -> Any:
        msg = "dict-like access to History is deprecated. Use attribute access instead."
        warnings.warn(msg, FutureWarning)
        return getattr(self, key)


def _flatten_params(valid_params: list[PyTree]) -> list[list[float]]:
    is_flat = (
        len(valid_params) > 0
        and isinstance(valid_params[0], np.ndarray)
        and valid_params[0].ndim == 1
    )

    if is_flat:
        flatten = lambda x: x.tolist()
    else:
        registry = get_registry(extended=True)
        flatten = partial(tree_just_flatten, registry=registry)

    return [flatten(p) for p in valid_params]
