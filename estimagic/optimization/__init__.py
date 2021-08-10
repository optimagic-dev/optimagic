import inspect

from estimagic.config import IS_CYIPOPT_INSTALLED
from estimagic.config import IS_DFOLS_INSTALLED
from estimagic.config import IS_NLOPT_INSTALLED
from estimagic.config import IS_PETSC4PY_INSTALLED
from estimagic.config import IS_PYBOBYQA_INSTALLED
from estimagic.optimization import cyipopt_optimizers
from estimagic.optimization import nag_optimizers
from estimagic.optimization import nlopt_optimizers
from estimagic.optimization import scipy_optimizers
from estimagic.optimization import tao_optimizers


COLLECTED_FUNCTIONS = {
    **dict(inspect.getmembers(scipy_optimizers, inspect.isfunction)),
}

if IS_PETSC4PY_INSTALLED:
    COLLECTED_FUNCTIONS.update(
        **dict(inspect.getmembers(tao_optimizers, inspect.isfunction))
    )

if IS_NLOPT_INSTALLED:
    COLLECTED_FUNCTIONS.update(
        **dict(inspect.getmembers(nlopt_optimizers, inspect.isfunction))
    )
# drop private and helper functions
AVAILABLE_ALGORITHMS = {}
for k, v in COLLECTED_FUNCTIONS.items():
    if not k.startswith("_") and k != "calculate_trustregion_initial_radius":
        AVAILABLE_ALGORITHMS[k] = v

if IS_PYBOBYQA_INSTALLED:
    AVAILABLE_ALGORITHMS["nag_pybobyqa"] = nag_optimizers.nag_pybobyqa

if IS_DFOLS_INSTALLED:
    AVAILABLE_ALGORITHMS["nag_dfols"] = nag_optimizers.nag_dfols

if IS_CYIPOPT_INSTALLED:
    AVAILABLE_ALGORITHMS["ipopt"] = cyipopt_optimizers.ipopt
