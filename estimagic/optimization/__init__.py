import inspect

from estimagic.optimization import scipy_optimizers
from estimagic.optimization import tao_optimizers

AVAILABLE_ALGORITHMS = {
    **dict(inspect.getmembers(scipy_optimizers, inspect.isfunction)),
    **dict(inspect.getmembers(tao_optimizers, inspect.isfunction)),
}

AVAILABLE_ALGORITHMS = {
    key: val for key, val in AVAILABLE_ALGORITHMS.items() if not key.startswith("_")
}
