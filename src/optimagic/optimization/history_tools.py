import warnings
from dataclasses import dataclass
from functools import partial
from typing import Literal

import numpy as np
from numpy.typing import NDArray
from pybaum import tree_just_flatten

from optimagic.optimization.internal_optimization_problem import History
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
        msg = "The criterion attribute is deprecated in HistoryArrays."
        warnings.warn(msg, FutureWarning)
        return self.fun

    @property
    def monotone_criterion(self) -> NDArray[np.float64]:
        msg = "The monotone_criterion attribute is deprecated in HistoryArrays."
        warnings.warn(msg, FutureWarning)
        return self.monotone_fun

    def __getitem__(self, key: str) -> NDArray[np.float64] | NDArray[np.bool_]:
        msg = "Dict access for HistoryArrays is deprecated."
        warnings.warn(msg, FutureWarning)
        return getattr(self, key)


def get_history_arrays(
    history: History, direction: Direction | Literal["minimize", "maximize"]
) -> HistoryArrays:
    # ==================================================================================
    # Handle deprecations for now
    # ==================================================================================
    if direction not in ["minimize", "maximize"]:
        msg = "Strings as direction argument are deprecated in get_history_arrays."
        warnings.warn(msg, FutureWarning)

    if direction == "minimize":
        direction = Direction.MINIMIZE
    elif direction == "maximize":
        direction = Direction.MAXIMIZE

    if isinstance(history, dict):
        msg = "Dict input for history argument is deprecated in get_history_arrays."
        warnings.warn(msg, FutureWarning)

        parhist = history["params"]
        funhist = history["criterion"]
        timehist = history["runtime"]

    else:
        parhist = history.params
        funhist = history.fun
        timehist = history.time

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
