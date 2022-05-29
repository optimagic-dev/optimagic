from functools import partial

import numpy as np
from estimagic.parameters.tree_registry import get_registry
from pybaum import tree_just_flatten


def get_history_arrays(history, direction, converter=None):

    if converter is not None:
        to_internal = converter.params_to_internal
    else:
        registry = get_registry(extended=True)
        to_internal = partial(tree_just_flatten, registry=registry)

    critvals = np.array(history["criterion"])

    params = np.array([to_internal(p) for p in history["params"]])

    runtimes = np.array(history["runtime"])

    if direction == "minimize":
        monotone = np.minimum.accumulate(critvals)
        is_accepted = critvals <= monotone
    elif direction == "maximize":
        monotone = np.maximum.accumulate(critvals)
        is_accepted = critvals >= monotone

    out = {
        "criterion": critvals,
        "params": params,
        "runtimes": runtimes,
        "monotone_criterion": monotone,
        "is_accepted": is_accepted,
    }
    return out
