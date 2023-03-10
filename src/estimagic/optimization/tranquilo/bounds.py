from dataclasses import dataclass, replace

import numpy as np


@dataclass
class Bounds:
    """Parameter bounds."""

    lower: np.ndarray
    upper: np.ndarray

    def __post_init__(self):
        self.has_any = _any_finite(self.lower, self.upper)

    # make it behave like a NamedTuple
    def _replace(self, **kwargs):
        return replace(self, **kwargs)


def _any_finite(lb, ub):
    out = False
    if lb is not None and np.isfinite(lb).any():
        out = True
    if ub is not None and np.isfinite(ub).any():
        out = True
    return out
