import inspect

from estimagic.config import IS_PETSC4PY_INSTALLED
from estimagic.optimization import estimagic_optimizers
from estimagic.optimization import scipy_optimizers
from estimagic.optimization import tao_optimizers

AVAILABLE_ALGORITHMS = {
    **dict(inspect.getmembers(scipy_optimizers, inspect.isfunction)),
    **dict(inspect.getmembers(estimagic_optimizers, inspect.isfunction)),
}

if IS_PETSC4PY_INSTALLED:
    AVAILABLE_ALGORITHMS.update(
        **dict(inspect.getmembers(tao_optimizers, inspect.isfunction))
    )


AVAILABLE_ALGORITHMS = {
    key: val for key, val in AVAILABLE_ALGORITHMS.items() if not key.startswith("_")
}
