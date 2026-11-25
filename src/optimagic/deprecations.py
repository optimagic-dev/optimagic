"""Code that keeps deprecated functionality of optimagic working.

All deprecated functionality lives in this module, except for deprecated options of
algorithms, which are declared next to the algorithm. Code in other modules only calls
the functions defined here. When a deprecation expires, its function, the lines that
call it and its tests in ``tests/optimagic/test_deprecations.py`` are deleted.

The contributor guide describes the deprecation policy.

"""

import warnings

import numpy as np

# ======================================================================================
# Deprecated in 0.6.0, remove in 0.7.0
# ======================================================================================


def get_history_time(start_time: list[float], attribute: str) -> list[float]:
    """Warn that History.time or History.runtime is deprecated and return its value.

    Args:
        start_time: The start times of all evaluations in the history.
        attribute: The name of the deprecated attribute, "time" or "runtime".

    Returns:
        The start times, measured from the first start time.

    """
    msg = (
        f"The attribute `{attribute}` of History is deprecated and will be removed in "
        "optimagic version 0.7.0. Use the `start_time` attribute instead. "
        f"`{attribute}` equals `start_time` minus its first entry."
    )
    # stacklevel 3 points to the line that accessed the attribute
    warnings.warn(msg, FutureWarning, stacklevel=3)
    arr = np.array(start_time)
    return (arr - arr[0]).tolist()
