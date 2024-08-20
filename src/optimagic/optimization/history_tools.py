from dataclasses import dataclass
from functools import partial

import numpy as np
from numpy.typing import NDArray
from pybaum import tree_just_flatten

from optimagic.optimization.history import History
from optimagic.parameters.tree_registry import get_registry
from optimagic.typing import Direction


@dataclass(frozen=True)
class HistoryArrays:
    fun: NDArray[np.float64]
    params: NDArray[np.float64]
    time: NDArray[np.float64]
    monotone_fun: NDArray[np.float64]
    is_accepted: NDArray[np.bool_]

    @property
    def criterion(self) -> NDArray[np.float64]:
        return self.fun

    @property
    def monotone_criterion(self) -> NDArray[np.float64]:
        return self.monotone_fun

    def __getitem__(self, key: str) -> NDArray[np.float64] | NDArray[np.bool_]:
        return getattr(self, key)


def get_history_arrays(history: History, direction: Direction) -> HistoryArrays:
    # ==================================================================================
    # Handle deprecations for now
    # ==================================================================================
    assert direction in [Direction.MINIMIZE, Direction.MAXIMIZE]

    if isinstance(history, dict):
        parhist = history["params"]
        funhist = history["criterion"]
        timehist = history["runtime"]

    else:
        parhist = history.params
        funhist = history.fun
        timehist = history.time

    # ==================================================================================
    # Filter out evaluations that do not have a `fun` value
    # ==================================================================================

    parhist = [p for p, f in zip(parhist, funhist, strict=False) if f is not None]
    timehist = [t for t, f in zip(timehist, funhist, strict=False) if f is not None]
    funhist = [f for f in funhist if f is not None]

    # ==================================================================================

    is_flat = (
        len(parhist) > 0 and isinstance(parhist[0], np.ndarray) and parhist[0].ndim == 1
    )
    if is_flat:
        to_internal = lambda x: x.tolist()
    else:
        registry = get_registry(extended=True)
        to_internal = partial(tree_just_flatten, registry=registry)

    critvals = np.array(funhist)

    params = np.array([to_internal(p) for p in parhist])

    runtimes = np.array(timehist)

    if direction == Direction.MINIMIZE:
        monotone = np.minimum.accumulate(critvals)
        is_accepted = critvals <= monotone
    elif direction == Direction.MAXIMIZE:
        monotone = np.maximum.accumulate(critvals)
        is_accepted = critvals >= monotone

    out = HistoryArrays(
        fun=critvals,
        params=params,
        time=runtimes,
        monotone_fun=monotone,
        is_accepted=is_accepted,
    )
    return out
