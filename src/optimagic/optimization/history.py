from dataclasses import dataclass

import numpy as np

from optimagic.typing import EvalTask, PyTree


@dataclass(frozen=True)
class HistoryEntry:
    params: PyTree
    fun: float | None
    time: float
    task: EvalTask


class History:
    # TODO: add counters for the relevant evaluations
    def __init__(self) -> None:
        self._params: list[PyTree] = []
        self._fun: list[float | None] = []
        self._time: list[float] = []
        self._batches: list[int] = []
        self._task: list[EvalTask] = []

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

    @property
    def params(self) -> list[PyTree]:
        return self._params

    @property
    def fun(self) -> list[float | None]:
        return self._fun

    @property
    def time(self) -> list[float]:
        arr = np.array(self._time)
        return (arr - arr[0]).tolist()

    @property
    def batches(self) -> list[int]:
        return self._batches

    @property
    def task(self) -> list[EvalTask]:
        return self._task

    def _get_next_batch_id(self) -> int:
        if not self._batches:
            batch = 0
        else:
            batch = self._batches[-1] + 1
        return batch
